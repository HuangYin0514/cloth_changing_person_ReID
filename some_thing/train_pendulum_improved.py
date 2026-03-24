"""
train_pendulum_improved.py - 改进版冲击摆训练
使用单个网络 + 时间变换 + 强制约束
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
t0 = np.pi / (2 * omega0)  # 0.5015 s

print("=" * 60)
print("改进版冲击摆训练（单个网络 + 时间变换）")
print("=" * 60)
print(f"冲击时间: t0 = {t0:.4f} s")
print(f"自然频率: ω0 = {omega0:.4f} rad/s")


# ============================================================
# 时间变换函数
# ============================================================
def transform_time(t, t0):
    """
    时间变换：将分段函数转化为连续函数
    冲击前：τ = t
    冲击后：τ = t + α
    其中 α 使得变换后函数在冲击点处连续
    """
    # 简单方案：直接使用分段网络，这里用条件判断
    # 在损失函数中分别处理前后区间
    return t


# ============================================================
# 网络定义（更深、更宽）
# ============================================================
class ImprovedPendulumNet(torch.nn.Module):
    def __init__(self, hidden_layers=4, neurons=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, neurons),
            torch.nn.Tanh(),
            torch.nn.Linear(neurons, neurons),
            torch.nn.SiLU(),
            torch.nn.Linear(neurons, neurons),
            torch.nn.SiLU(),
            torch.nn.Linear(neurons, neurons),
        )
        self.output = torch.nn.Linear(neurons, 1)

    def forward(self, t):
        x = self.net(t)
        return self.output(torch.sin(x))

    def derivatives(self, t):
        t.requires_grad_(True)
        t = t.float()
        q = self.forward(t)
        q_dot = torch.autograd.grad(q, t, torch.ones_like(q), create_graph=True, retain_graph=True)[0]
        q_ddot = torch.autograd.grad(q_dot, t, torch.ones_like(q_dot), create_graph=True)[0]
        return q, q_dot, q_ddot


def pde_residual(theta, theta_dot, theta_ddot):
    """PDE残差: θ̈ + (g/L) sinθ = 0"""
    return theta_ddot + (g / L) * torch.sin(theta)


def exact_solution(t):
    """精确解（分段）"""
    result = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < t0:
            result[i] = theta0 * np.cos(omega0 * ti)
        else:
            dt = ti - t0
            result[i] = -e * theta0 * np.sin(omega0 * dt)
    return result


# ============================================================
# 训练函数
# ============================================================
def train_improved(epochs=20000, n_coll=1000, lr=1e-3, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")

    net = ImprovedPendulumNet(hidden_layers=4, neurons=128).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2000)

    # 损失权重（逐步增加冲击权重）
    lambda_pde = 1.0
    lambda_ic = 10.0

    history = {"loss": [], "pde": [], "ic": [], "jump": [], "t0_pred": []}

    print("\n训练配置:")
    print(f"  训练轮数: {epochs}")
    print(f"  配点数: {n_coll}")
    print(f"  学习率: {lr}")
    print("-" * 60)

    start_time = time.time()

    for epoch in range(epochs):
        # 动态调整冲击权重
        lambda_jump = min(100.0, (epoch / 5000) * 100.0)

        # ===== 配点生成 =====
        # 普通配点
        t_normal = torch.rand(n_coll, 1, device=device) * t_final

        # 冲击点附近加密
        t_fine = torch.rand(n_coll // 2, 1, device=device) * 0.2 + (t0 - 0.1)
        t_fine = torch.clamp(t_fine, min=0, max=t_final)

        # 极靠近冲击点的点
        t_near = torch.rand(200, 1, device=device) * 0.02 + (t0 - 0.01)
        t_near = torch.clamp(t_near, min=0, max=t_final)

        t = torch.cat([t_normal, t_fine, t_near], dim=0)

        # ===== 前向计算 =====
        theta, theta_dot, theta_ddot = net.derivatives(t)

        # ===== PDE损失（所有点）=====
        pde = pde_residual(theta, theta_dot, theta_ddot)
        loss_pde = torch.mean(pde**2)

        # ===== 初始条件损失 =====
        t_ic = torch.zeros(1, 1, device=device)
        theta_ic, theta_dot_ic, _ = net.derivatives(t_ic)
        loss_ic_theta = torch.mean((theta_ic - theta0) ** 2)
        loss_ic_vel = torch.mean(theta_dot_ic**2)
        loss_ic = loss_ic_theta + loss_ic_vel

        # ===== 冲击条件损失 =====
        # 在t0点强制位置为0
        t0_tensor = torch.tensor([[t0]], device=device)
        theta_t0, theta_dot_t0, _ = net.derivatives(t0_tensor)
        loss_pos_jump = torch.mean(theta_t0**2) * 10.0

        # 速度跳变：用前后差分计算
        eps = 1e-4
        t_before = torch.tensor([[t0 - eps]], device=device)
        t_after = torch.tensor([[t0 + eps]], device=device)

        theta_before, theta_dot_before, _ = net.derivatives(t_before)
        theta_after, theta_dot_after, _ = net.derivatives(t_after)

        # 速度跳变条件：θ̇_after = -e * θ̇_before
        loss_vel_jump = torch.mean((theta_dot_after + e * theta_dot_before) ** 2) * 10.0

        # 在冲击点附近强制连续性
        t_left_near = torch.linspace(t0 - 0.02, t0, 50, device=device).reshape(-1, 1)
        t_right_near = torch.linspace(t0, t0 + 0.02, 50, device=device).reshape(-1, 1)

        theta_left_near, _, _ = net.derivatives(t_left_near)
        theta_right_near, _, _ = net.derivatives(t_right_near)

        loss_continuity = torch.mean((theta_left_near[-1] - theta_right_near[0]) ** 2) * 0.0

        loss_jump = loss_pos_jump + loss_vel_jump + loss_continuity

        # ===== 总损失 =====
        loss_total = lambda_pde * loss_pde + lambda_ic * loss_ic + lambda_jump * loss_jump

        # ===== 反向传播 =====
        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss_total)

        # ===== 记录 =====
        history["loss"].append(loss_total.item())
        history["pde"].append(loss_pde.item())
        history["ic"].append(loss_ic.item())
        history["jump"].append(loss_jump.item())
        history["t0_pred"].append(theta_t0.item())

        if verbose and (epoch + 1) % 1000 == 0:
            print(
                f"Epoch {epoch+1:6d}: Loss={loss_total.item():.4e}, "
                f"PDE={loss_pde.item():.4e}, IC={loss_ic.item():.4e}, "
                f"Jump={loss_jump.item():.4e}, θ(t0)={theta_t0.item():.4f} rad"
            )

    elapsed = time.time() - start_time
    print(f"\n训练完成，用时 {elapsed:.2f} 秒")

    # 保存模型
    torch.save(net.state_dict(), "pendulum_improved.pth")
    np.save("pendulum_improved_history.npy", history)
    print("模型已保存: pendulum_improved.pth")

    return net, history


# ============================================================
# 测试函数
# ============================================================
def test_model(net):
    """测试模型精度"""
    device = next(net.parameters()).device

    t_test = torch.linspace(0, t_final, 1000, device=device).reshape(-1, 1)
    theta_pred, _, _ = net.derivatives(t_test)

    t_np = t_test.detach().cpu().numpy().flatten()
    theta_pred_np = theta_pred.detach().cpu().numpy().flatten()
    theta_exact_np = exact_solution(t_np)

    error = np.abs(theta_pred_np - theta_exact_np)
    l2_error = np.sqrt(np.mean(error**2))
    max_error = np.max(error)

    # 检查冲击点
    t0_tensor = torch.tensor([[t0]], device=device)
    theta_t0, theta_dot_t0, _ = net.derivatives(t0_tensor)

    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    print(f"冲击点 θ(t0): {theta_t0.item():.6f} rad ({theta_t0.item()*180/np.pi:.4f}°)")
    print(f"冲击点 θ̇(t0): {theta_dot_t0.item():.4f} rad/s")
    print(f"L2误差: {l2_error:.4e} rad ({l2_error*180/np.pi:.4f}°)")
    print(f"最大误差: {max_error:.4e} rad ({max_error*180/np.pi:.4f}°)")

    # 判断是否成功
    if abs(theta_t0.item()) < 0.01 and l2_error < 0.01:
        print("\n✅ 精度达标！冲击点误差 < 0.01 rad，L2误差 < 0.01 rad")
    else:
        print("\n⚠️ 精度不足，建议增加训练轮数或调整参数")

    return t_np, theta_pred_np, theta_exact_np, error, l2_error, theta_t0.item()


# ============================================================
# 绘图函数
# ============================================================
def plot_results(net, history):
    """绘制结果"""
    t_np, theta_pred, theta_exact, error, l2_error, theta_t0 = test_model(net)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 角度对比
    axes[0, 0].plot(t_np, theta_exact, "k-", linewidth=2, label="Exact")
    axes[0, 0].plot(t_np, theta_pred, "r--", linewidth=1.5, label="PINN")
    axes[0, 0].axvline(x=t0, color="b", linestyle=":", label=f"Impact at t={t0:.3f}s")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Angle (rad)")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_title("Pendulum Motion with Impact")

    # 2. 误差
    axes[0, 1].semilogy(t_np, error, "g-", linewidth=1.5)
    axes[0, 1].axvline(x=t0, color="b", linestyle=":")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Error (rad)")
    axes[0, 1].set_title(f"Error (L₂ = {l2_error:.2e} rad)")
    axes[0, 1].grid(True)

    # 3. 训练损失
    axes[1, 0].semilogy(history["loss"], label="Total")
    axes[1, 0].semilogy(history["pde"], label="PDE")
    axes[1, 0].semilogy(history["ic"], label="IC")
    axes[1, 0].semilogy(history["jump"], label="Jump")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_title("Training Loss")

    # 4. 冲击点预测值变化
    axes[1, 1].plot(history["t0_pred"], "b-", linewidth=1.5)
    axes[1, 1].axhline(y=0, color="r", linestyle="--", label="Target (0 rad)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("θ(t0) (rad)")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_title("Impact Point Prediction")

    plt.tight_layout()
    plt.savefig("pendulum_improved_results.png", dpi=300)
    plt.show()
    print("\n图片已保存: pendulum_improved_results.png")


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    # 训练
    net, history = train_improved(epochs=6000, n_coll=800, lr=1e-3)

    # 测试并绘图
    plot_results(net, history)

    print("\n完成！")
