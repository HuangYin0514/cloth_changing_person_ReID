"""
train_pendulum_fixed.py - 修正版冲击摆训练（修复维度+数据类型错误）
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import torch

# ============================================================
# 物理参数
# ============================================================
g = 9.81
L = 1.0
theta0 = np.pi / 4
e = 0.8
t_final = 2.0
omega0 = np.sqrt(g / L)
t0 = np.pi / (2 * omega0)

print("=" * 60)
print("修正版冲击摆训练")
print("=" * 60)
print(f"冲击时间: t0 = {t0:.4f} s")


# ============================================================
# 网络定义
# ============================================================
class PendulumNet(torch.nn.Module):
    def __init__(self, hidden=3, neurons=64):
        super().__init__()
        layers = [torch.nn.Linear(1, neurons), torch.nn.Tanh()]
        for _ in range(hidden - 1):
            layers.append(torch.nn.Linear(neurons, neurons))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(neurons, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)

    def derivatives(self, t):
        t.requires_grad_(True)
        q = self.net(t)
        q_dot = torch.autograd.grad(q, t, torch.ones_like(q), create_graph=True, retain_graph=True)[0]
        q_ddot = torch.autograd.grad(q_dot, t, torch.ones_like(q_dot), create_graph=True)[0]
        return q, q_dot, q_ddot


def pde_residual(theta, theta_dot, theta_ddot):
    return theta_ddot + (g / L) * torch.sin(theta)


# ============================================================
# 训练函数（修复版）
# ============================================================
def train_pendulum_fixed(epochs=15000, n_coll=800, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # 关键修复1: 统一数据类型为float32
    dtype = torch.float32

    net_left = PendulumNet().to(device, dtype=dtype)
    net_right = PendulumNet().to(device, dtype=dtype)

    optimizer = torch.optim.Adam(list(net_left.parameters()) + list(net_right.parameters()), lr=lr, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2000)

    # 损失权重
    lambda_pde = 0.1
    lambda_ic = 10.0
    lambda_jump = 100.0

    history = {"loss": [], "pde": [], "ic": [], "jump": []}

    print(f"\n训练配置:")
    print(f"  训练轮数: {epochs}")
    print(f"  配点数: {n_coll}")
    print(f"  学习率: {lr}")
    print(f"  损失权重: PDE={lambda_pde}, IC={lambda_ic}, Jump={lambda_jump}")
    print("-" * 60)

    start_time = time.time()

    for epoch in range(epochs):
        # ===== 生成配点（修复维度+数据类型问题）=====
        # 左子域配点
        n_left_normal = n_coll // 2
        n_left_fine = n_coll // 2

        # 关键修复2: 生成float32类型的张量
        t_left_normal = torch.rand(n_left_normal, 1, device=device, dtype=dtype) * t0
        t_left_fine = torch.rand(n_left_fine, 1, device=device, dtype=dtype) * 0.1 + (t0 - 0.05)
        t_left_fine = t_left_fine[t_left_fine[:, 0] >= 0].reshape(-1, 1)

        # 如果过滤后数量太少，补充一些点
        if t_left_fine.shape[0] < n_left_fine // 2:
            t_left_fine = torch.rand(n_left_fine, 1, device=device, dtype=dtype) * 0.08 + (t0 - 0.04)
            t_left_fine = t_left_fine[t_left_fine[:, 0] >= 0].reshape(-1, 1)

        t_left = torch.cat([t_left_normal, t_left_fine], dim=0)

        # 右子域配点
        n_right_normal = n_coll // 2
        n_right_fine = n_coll // 2

        t_right_normal = torch.rand(n_right_normal, 1, device=device, dtype=dtype) * (t_final - t0) + t0
        t_right_fine = torch.rand(n_right_fine, 1, device=device, dtype=dtype) * 0.1 + t0
        t_right_fine = t_right_fine[t_right_fine[:, 0] <= t_final].reshape(-1, 1)

        if t_right_fine.shape[0] < n_right_fine // 2:
            t_right_fine = torch.rand(n_right_fine, 1, device=device, dtype=dtype) * 0.08 + t0
            t_right_fine = t_right_fine[t_right_fine[:, 0] <= t_final].reshape(-1, 1)

        t_right = torch.cat([t_right_normal, t_right_fine], dim=0)

        # 左子域损失
        theta_left, theta_dot_left, theta_ddot_left = net_left.derivatives(t_left)
        loss_pde_left = torch.mean(pde_residual(theta_left, theta_dot_left, theta_ddot_left) ** 2)

        # 右子域损失
        theta_right, theta_dot_right, theta_ddot_right = net_right.derivatives(t_right)
        loss_pde_right = torch.mean(pde_residual(theta_right, theta_dot_right, theta_ddot_right) ** 2)
        loss_pde = (loss_pde_left + loss_pde_right) / 2

        # 初始条件
        t_ic = torch.zeros(1, 1, device=device, dtype=dtype)
        theta_ic, theta_dot_ic, _ = net_left.derivatives(t_ic)
        loss_ic_theta = torch.mean((theta_ic - theta0) ** 2)
        loss_ic_vel = torch.mean(theta_dot_ic**2)
        loss_ic = loss_ic_theta + loss_ic_vel

        # 冲击条件
        # 关键修复3: t0_tensor指定为float32类型
        t0_tensor = torch.tensor([[t0]], device=device, dtype=dtype)
        theta_left_t0, theta_dot_left_t0, _ = net_left.derivatives(t0_tensor)
        theta_right_t0, theta_dot_right_t0, _ = net_right.derivatives(t0_tensor)

        loss_pos_jump = torch.mean((theta_left_t0 - theta_right_t0) ** 2) * 10.0
        loss_vel_jump = torch.mean((theta_dot_right_t0 + e * theta_dot_left_t0) ** 2) * 10.0
        loss_jump = loss_pos_jump + loss_vel_jump

        # 总损失
        loss_total = lambda_pde * loss_pde + lambda_ic * loss_ic + lambda_jump * loss_jump

        # 反向传播
        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(net_left.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(net_right.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss_total)

        # 记录
        history["loss"].append(loss_total.item())
        history["pde"].append(loss_pde.item())
        history["ic"].append(loss_ic.item())
        history["jump"].append(loss_jump.item())

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1:6d}: Loss={loss_total.item():.4e}, " f"PDE={loss_pde.item():.4e}, IC={loss_ic.item():.4e}, Jump={loss_jump.item():.4e}")

            if (epoch + 1) % 5000 == 0:
                print(f"        冲击点: θ_left={theta_left_t0.item():.4f}, θ_right={theta_right_t0.item():.4f}")

    elapsed = time.time() - start_time
    print(f"\n训练完成，用时 {elapsed:.2f} 秒")

    # 保存
    torch.save(net_left.state_dict(), "pendulum_left_fixed.pth")
    torch.save(net_right.state_dict(), "pendulum_right_fixed.pth")
    np.save("pendulum_history_fixed.npy", history)
    print("模型已保存: pendulum_left_fixed.pth, pendulum_right_fixed.pth")

    return net_left, net_right, history


# ============================================================
# 测试函数
# ============================================================
def test_model(net_left, net_right):
    """测试模型"""
    device = next(net_left.parameters()).device
    dtype = next(net_left.parameters()).dtype  # 获取模型使用的数据类型

    t_left = torch.linspace(0, t0, 500, device=device, dtype=dtype).reshape(-1, 1)
    t_right = torch.linspace(t0, t_final, 500, device=device, dtype=dtype).reshape(-1, 1)

    net_left.eval()
    net_right.eval()

    theta_left, _, _ = net_left.derivatives(t_left)
    theta_right, _, _ = net_right.derivatives(t_right)

    t_np = np.concatenate([t_left.detach().cpu().numpy().flatten(), t_right.detach().cpu().numpy().flatten()])
    theta_pinn = np.concatenate([theta_left.detach().cpu().numpy().flatten(), theta_right.detach().cpu().numpy().flatten()])

    # 精确解
    def exact(t_val):
        if t_val < t0:
            return theta0 * np.cos(omega0 * t_val)
        else:
            dt = t_val - t0
            return -e * theta0 * np.sin(omega0 * dt)

    theta_exact = np.array([exact(tt) for tt in t_np])
    error = np.abs(theta_pinn - theta_exact)

    l2_error = np.sqrt(np.mean(error**2))
    max_error = np.max(error)

    print(f"\n测试结果:")
    print(f"  L2误差: {l2_error:.4e} rad ({l2_error*180/np.pi:.2f}°)")
    print(f"  最大误差: {max_error:.4e} rad ({max_error*180/np.pi:.2f}°)")

    # 检查冲击点
    t0_tensor = torch.tensor([[t0]], device=device, dtype=dtype)
    theta_left_t0, _, _ = net_left.derivatives(t0_tensor)
    theta_right_t0, _, _ = net_right.derivatives(t0_tensor)
    print(f"  冲击点 θ_left: {theta_left_t0.item():.4f} rad ({theta_left_t0.item()*180/np.pi:.2f}°)")
    print(f"  冲击点 θ_right: {theta_right_t0.item():.4f} rad ({theta_right_t0.item()*180/np.pi:.2f}°)")

    return t_np, theta_pinn, theta_exact, error, l2_error


# ============================================================
# 绘图函数
# ============================================================
def plot_results(net_left, net_right, history):
    """绘制结果"""
    t_np, theta_pinn, theta_exact, error, l2_error = test_model(net_left, net_right)

    t_exact_plot = np.linspace(0, t_final, 1000)
    theta_exact_plot = np.array([theta0 * np.cos(omega0 * tt) if tt < t0 else -e * theta0 * np.sin(omega0 * (tt - t0)) for tt in t_exact_plot])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 角度对比
    axes[0, 0].plot(t_exact_plot, theta_exact_plot, "k-", linewidth=2, label="Exact")
    axes[0, 0].plot(t_np, theta_pinn, "r--", linewidth=1.5, label="PINN")
    axes[0, 0].axvline(x=t0, color="b", linestyle=":", label=f"Impact at t={t0:.3f}s")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Angle (rad)")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_title("Pendulum Motion with Impact")

    # 误差
    axes[0, 1].semilogy(t_np, error, "g-")
    axes[0, 1].axvline(x=t0, color="b", linestyle=":")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Error (rad)")
    axes[0, 1].set_title(f"Error (L₂ = {l2_error:.2e} rad)")
    axes[0, 1].grid(True)

    # 训练损失
    axes[1, 0].semilogy(history["loss"], label="Total")
    axes[1, 0].semilogy(history["pde"], label="PDE")
    axes[1, 0].semilogy(history["ic"], label="IC")
    axes[1, 0].semilogy(history["jump"], label="Jump")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_title("Training Loss")

    # 冲击点附近的细节
    near_mask = (t_np > t0 - 0.1) & (t_np < t0 + 0.1)
    t_near = t_np[near_mask]
    theta_near = theta_pinn[near_mask]
    exact_near = theta_exact[near_mask]

    axes[1, 1].plot(t_near, theta_near, "r-", label="PINN")
    axes[1, 1].plot(t_near, exact_near, "k--", label="Exact")
    axes[1, 1].axvline(x=t0, color="b", linestyle=":")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Angle (rad)")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_title("Near Impact Region")

    plt.tight_layout()
    plt.savefig("pendulum_fixed_results.png", dpi=300)
    plt.show()
    print("\n图片已保存: pendulum_fixed_results.png")


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    # 训练
    net_left, net_right, history = train_pendulum_fixed(epochs=15000, n_coll=800, lr=1e-3)

    # 测试并绘图
    plot_results(net_left, net_right, history)

    print("\n完成！")
