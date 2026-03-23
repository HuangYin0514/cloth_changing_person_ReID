"""
train_and_save_pendulum.py - 训练并保存冲击摆模型
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# 物理参数
g = 9.81
L = 1.0
theta0 = np.pi / 4
e = 0.8
t_final = 2.0
t0 = 0.5015  # 冲击时间


# 网络定义
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
        q_dot = torch.autograd.grad(q, t, torch.ones_like(q), create_graph=True)[0]
        q_ddot = torch.autograd.grad(q_dot, t, torch.ones_like(q_dot), create_graph=True)[0]
        return q, q_dot, q_ddot


def pde_residual(theta, theta_dot, theta_ddot):
    return theta_ddot + (g / L) * torch.sin(theta)


def train_pendulum(epochs=10000, n_coll=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    net_left = PendulumNet().to(device)
    net_right = PendulumNet().to(device)

    optimizer = torch.optim.Adam(list(net_left.parameters()) + list(net_right.parameters()), lr=1e-3)

    lambda_pde = 0.01
    lambda_ic = 1.0
    lambda_jump = 1.0

    history = {"loss": [], "pde": [], "ic": [], "jump": []}

    print(f"\n训练参数:")
    print(f"  冲击时间: t0 = {t0} s")
    print(f"  训练轮数: {epochs}")
    print(f"  配点数: {n_coll}")
    print("-" * 50)

    for epoch in range(epochs):
        # 配点
        t_left = torch.rand(n_coll, 1, device=device) * t0
        t_right = torch.rand(n_coll, 1, device=device) * (t_final - t0) + t0

        # 左子域
        theta_left, theta_dot_left, theta_ddot_left = net_left.derivatives(t_left)
        loss_pde_left = torch.mean(pde_residual(theta_left, theta_dot_left, theta_ddot_left) ** 2)

        # 右子域
        theta_right, theta_dot_right, theta_ddot_right = net_right.derivatives(t_right)
        loss_pde_right = torch.mean(pde_residual(theta_right, theta_dot_right, theta_ddot_right) ** 2)
        loss_pde = (loss_pde_left + loss_pde_right) / 2

        # 初始条件
        t_ic = torch.zeros(1, 1, device=device)
        theta_ic, theta_dot_ic, _ = net_left.derivatives(t_ic)
        loss_ic = torch.mean((theta_ic - theta0) ** 2) + torch.mean(theta_dot_ic**2)

        # 冲击条件
        t0_tensor = torch.tensor([[t0]], device=device)
        theta_left_t0, theta_dot_left_t0, _ = net_left.derivatives(t0_tensor)
        theta_right_t0, theta_dot_right_t0, _ = net_right.derivatives(t0_tensor)

        loss_pos_jump = torch.mean((theta_left_t0 - theta_right_t0) ** 2)
        loss_vel_jump = torch.mean((theta_dot_right_t0 + e * theta_dot_left_t0) ** 2)
        loss_jump = loss_pos_jump + loss_vel_jump

        # 总损失
        loss_total = lambda_pde * loss_pde + lambda_ic * loss_ic + lambda_jump * loss_jump

        # 反向传播
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        # 记录
        history["loss"].append(loss_total.item())
        history["pde"].append(loss_pde.item())
        history["ic"].append(loss_ic.item())
        history["jump"].append(loss_jump.item())

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1:5d}: Loss={loss_total.item():.4e}, " f"PDE={loss_pde.item():.4e}, IC={loss_ic.item():.4e}, Jump={loss_jump.item():.4e}")

    # 保存模型和历史
    torch.save(net_left.state_dict(), "pendulum_left.pth")
    torch.save(net_right.state_dict(), "pendulum_right.pth")
    np.save("pendulum_history.npy", history)

    print("\n模型已保存: pendulum_left.pth, pendulum_right.pth")
    print("训练历史已保存: pendulum_history.npy")

    return net_left, net_right, history


def compute_exact_solution(t, theta0, e, omega0, t0):
    theta = np.zeros_like(t)
    mask_before = t < t0
    theta[mask_before] = theta0 * np.cos(omega0 * t[mask_before])
    mask_after = t >= t0
    dt = t[mask_after] - t0
    theta_max_after = e * theta0
    theta[mask_after] = -theta_max_after * np.sin(omega0 * dt)
    return theta


def plot_results(net_left, net_right, history):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    omega0 = np.sqrt(g / L)

    t_left = torch.linspace(0, t0, 500, device=device).reshape(-1, 1)
    t_right = torch.linspace(t0, t_final, 500, device=device).reshape(-1, 1)

    net_left.eval()
    net_right.eval()

    with torch.no_grad():
        theta_left, _, _ = net_left.derivatives(t_left)
        theta_right, _, _ = net_right.derivatives(t_right)

    t_np = np.concatenate([t_left.cpu().numpy().flatten(), t_right.cpu().numpy().flatten()])
    theta_pinn = np.concatenate([theta_left.cpu().numpy().flatten(), theta_right.cpu().numpy().flatten()])

    # 精确解
    t_exact = np.linspace(0, t_final, 1000)
    theta_exact = compute_exact_solution(t_exact, theta0, e, omega0, t0)

    # 插值计算误差
    theta_exact_interp = np.interp(t_np, t_exact, theta_exact)
    error = np.abs(theta_pinn - theta_exact_interp)
    l2_error = np.sqrt(np.mean(error**2))
    max_error = np.max(error)

    print(f"\n误差分析:")
    print(f"  L2误差: {l2_error:.4e} rad")
    print(f"  最大误差: {max_error:.4e} rad")

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 角度对比
    axes[0, 0].plot(t_exact, theta_exact, "k-", linewidth=2, label="Exact")
    axes[0, 0].plot(t_np, theta_pinn, "r--", linewidth=1.5, label="PINN")
    axes[0, 0].axvline(x=t0, color="b", linestyle=":", label=f"Impact at t={t0:.3f}s")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Angle (rad)")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_title("Pendulum Motion with Impact")

    # 误差
    axes[0, 1].semilogy(t_np, error, "g-", linewidth=1.5)
    axes[0, 1].axvline(x=t0, color="b", linestyle=":")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Absolute Error (rad)")
    axes[0, 1].set_title(f"Error (L₂ = {l2_error:.2e}, Max = {max_error:.2e})")
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
    axes[1, 0].set_title("Training Loss History")

    # 损失细节
    axes[1, 1].semilogy(history["pde"], label="PDE")
    axes[1, 1].semilogy(history["ic"], label="IC")
    axes[1, 1].semilogy(history["jump"], label="Jump")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_title("Loss Components")

    plt.tight_layout()
    plt.savefig("pendulum_results.pdf", dpi=300, bbox_inches="tight")
    plt.show()
    print("\n图片已保存: pendulum_results.pdf")


if __name__ == "__main__":
    print("=" * 60)
    print("训练冲击摆模型")
    print("=" * 60)

    # 训练
    net_left, net_right, history = train_pendulum(epochs=10000, n_coll=500)

    # 绘图
    plot_results(net_left, net_right, history)

    print("\n完成!")
