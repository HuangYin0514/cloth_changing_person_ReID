import matplotlib.pyplot as plt
import numpy as np
import torch

# 物理参数
g = 9.81
L = 1.0
theta0 = np.pi / 4
e = 0.8
t_final = 3.0


# 网络定义（与训练时相同）
class SimplePendulumNet(torch.nn.Module):
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


def compute_exact_solution(t, theta0, e, g, L):
    """计算精确解（分段）"""
    omega0 = np.sqrt(g / L)

    # 小角度近似（用于演示）
    # 实际应该用数值积分，这里用近似
    theta = np.zeros_like(t)

    # 冲击时间
    t_impact = np.pi / (2 * omega0)

    for i, ti in enumerate(t):
        if ti < t_impact:
            # 冲击前
            theta[i] = theta0 * np.cos(omega0 * ti)
        else:
            # 冲击后：振幅减小
            dt = ti - t_impact
            theta_max = e * theta0  # 近似
            theta[i] = -theta_max * np.sin(omega0 * dt)

    return theta


def plot_results(net_left, net_right, t0, save_path="pendulum_final.png"):
    """绘制对比图"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_left.to(device)
    net_right.to(device)
    net_left.eval()
    net_right.eval()

    # 时间点
    t_left = torch.linspace(0, t0, 500, device=device).reshape(-1, 1)
    t_right = torch.linspace(t0, t_final, 500, device=device).reshape(-1, 1)

    # 预测值
    theta_left, _, _ = net_left.derivatives(t_left)
    theta_right, _, _ = net_right.derivatives(t_right)
    theta_left_np = theta_left.detach().cpu().numpy().flatten()
    theta_right_np = theta_right.detach().cpu().numpy().flatten()

    t_np = np.concatenate([t_left.detach().cpu().numpy().flatten(), t_right.detach().cpu().numpy().flatten()])
    theta_pinn = np.concatenate([theta_left_np, theta_right_np])

    # 计算精确解
    theta_exact = compute_exact_solution(t_np, theta0, e, g, L)

    # 计算误差
    error = np.abs(theta_pinn - theta_exact)
    l2_error = np.sqrt(np.mean(error**2))
    max_error = np.max(error)

    print(f"\n误差分析:")
    print(f"  L2误差: {l2_error:.4e}")
    print(f"  最大误差: {max_error:.4e}")

    # 绘图
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 上子图：角度对比
    axes[0].plot(t_np, theta_exact, "k-", linewidth=2, label="Exact Solution")
    axes[0].plot(t_np, theta_pinn, "r--", linewidth=1.5, label="PINN Prediction")
    axes[0].axvline(x=t0, color="b", linestyle=":", linewidth=1.5, label=f"Impact at t={t0:.3f}s")
    axes[0].set_xlabel("Time (s)", fontsize=12)
    axes[0].set_ylabel("Angle (rad)", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Pendulum Motion with Impact", fontsize=14)

    # 下子图：误差
    axes[1].semilogy(t_np, error, "g-", linewidth=1.5)
    axes[1].axvline(x=t0, color="b", linestyle=":", linewidth=1.5)
    axes[1].set_xlabel("Time (s)", fontsize=12)
    axes[1].set_ylabel("Absolute Error (rad)", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title(f"Error (L2 = {l2_error:.2e}, Max = {max_error:.2e})", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"\n图片已保存至: {save_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("绘制带冲击摆的PINN结果")
    print("=" * 60)

    # 加载训练好的模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net_left = SimplePendulumNet(hidden=3, neurons=64)
    net_right = SimplePendulumNet(hidden=3, neurons=64)

    net_left.load_state_dict(torch.load("pendulum_left.pth", map_location=device))
    net_right.load_state_dict(torch.load("pendulum_right.pth", map_location=device))

    net_left.to(device)
    net_right.to(device)

    # 冲击时间（与训练时相同）
    omega0 = np.sqrt(g / L)
    t0 = np.pi / (2 * omega0)
    print(f"冲击时间: {t0:.4f} s")

    # 绘图
    plot_results(net_left, net_right, t0, save_path="pendulum_final.png")
