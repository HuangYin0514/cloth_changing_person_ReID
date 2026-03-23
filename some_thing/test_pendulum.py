"""
test_pendulum.py - 测试训练好的冲击摆模型
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

# ============================================================
# 物理参数（必须与训练时完全一致）
# ============================================================
g = 9.81
L = 1.0
theta0 = np.pi / 4
e = 0.8
t_final = 2.0
omega0 = np.sqrt(g / L)
t0 = np.pi / (2 * omega0)  # 0.5015 s

print("=" * 60)
print("测试冲击摆模型")
print("=" * 60)
print(f"冲击时间: t0 = {t0:.4f} s")
print(f"自然频率: ω0 = {omega0:.4f} rad/s")


# ============================================================
# 网络定义（必须与训练时完全一致）
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
        q_dot = torch.autograd.grad(q, t, torch.ones_like(q), create_graph=True)[0]
        q_ddot = torch.autograd.grad(q_dot, t, torch.ones_like(q_dot), create_graph=True)[0]
        return q, q_dot, q_ddot


# ============================================================
# 精确解
# ============================================================
def compute_exact_solution(t):
    """精确解（分段函数）"""
    theta = np.zeros_like(t)

    # 冲击前：θ = θ0 cos(ω0 t)
    mask_before = t < t0
    theta[mask_before] = theta0 * np.cos(omega0 * t[mask_before])

    # 冲击后：振幅减小，相位调整
    mask_after = t >= t0
    dt = t[mask_after] - t0
    theta_max_after = e * theta0
    theta[mask_after] = -theta_max_after * np.sin(omega0 * dt)

    return theta


# ============================================================
# 主程序
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    try:
        net_left = PendulumNet()
        net_right = PendulumNet()
        net_left.load_state_dict(torch.load("pendulum_left.pth", map_location=device))
        net_right.load_state_dict(torch.load("pendulum_right.pth", map_location=device))
        print("模型加载成功")
    except FileNotFoundError:
        print("错误：找不到模型文件 pendulum_left.pth 或 pendulum_right.pth")
        print("请先运行 train_pendulum_complete.py 训练模型")
        return

    net_left.to(device)
    net_right.to(device)

    # 生成测试点
    t_left = torch.linspace(0, t0, 1000, device=device).reshape(-1, 1)
    t_right = torch.linspace(t0, t_final, 1000, device=device).reshape(-1, 1)

    # 预测
    theta_left, _, _ = net_left.derivatives(t_left)
    theta_right, _, _ = net_right.derivatives(t_right)

    # 合并结果
    t_np = np.concatenate([t_left.detach().cpu().numpy().flatten(), t_right.detach().cpu().numpy().flatten()])
    theta_pinn = np.concatenate([theta_left.detach().cpu().numpy().flatten(), theta_right.detach().cpu().numpy().flatten()])

    # 精确解
    t_exact = np.linspace(0, t_final, 2000)
    theta_exact = compute_exact_solution(t_exact)

    # 插值计算误差
    theta_exact_at_pinn = np.interp(t_np, t_exact, theta_exact)
    error = np.abs(theta_pinn - theta_exact_at_pinn)
    l2_error = np.sqrt(np.mean(error**2))
    max_error = np.max(error)

    print(f"\n误差分析:")
    print(f"  L2误差: {l2_error:.4e} rad ({l2_error*180/np.pi:.2f}°)")
    print(f"  最大误差: {max_error:.4e} rad ({max_error*180/np.pi:.2f}°)")

    # 检查模型是否真的训练过（通过检查t0时刻的值）
    t0_tensor = torch.tensor([[t0]], device=device)
    with torch.no_grad():
        theta_left_t0, _, _ = net_left.derivatives(t0_tensor)
        theta_right_t0, _, _ = net_right.derivatives(t0_tensor)

    print(f"\n在冲击时刻 t0 = {t0:.4f} s:")
    print(f"  左网络预测 θ = {theta_left_t0.item():.4f} rad")
    print(f"  右网络预测 θ = {theta_right_t0.item():.4f} rad")
    print(f"  精确解 θ = 0 rad")

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 角度对比
    axes[0, 0].plot(t_exact, theta_exact, "k-", linewidth=2, label="Exact Solution")
    axes[0, 0].plot(t_np, theta_pinn, "r--", linewidth=1.5, label="PINN Prediction")
    axes[0, 0].axvline(x=t0, color="b", linestyle=":", linewidth=1.5, label=f"Impact at t={t0:.3f}s")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Angle (rad)")
    axes[0, 0].legend(loc="upper right")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title("Pendulum Motion with Impact")

    # 误差
    axes[0, 1].semilogy(t_np, error, "g-", linewidth=1.5)
    axes[0, 1].axvline(x=t0, color="b", linestyle=":")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Absolute Error (rad)")
    axes[0, 1].set_title(f"Error (L₂ = {l2_error:.2e} rad, Max = {max_error:.2e} rad)")
    axes[0, 1].grid(True, alpha=0.3)

    # 冲击前后对比
    axes[1, 0].plot(t_np[t_np < t0], theta_pinn[t_np < t0], "b-", label="PINN (before)")
    axes[1, 0].plot(t_np[t_np >= t0], theta_pinn[t_np >= t0], "r-", label="PINN (after)")
    axes[1, 0].axvline(x=t0, color="k", linestyle=":")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Angle (rad)")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_title("PINN Prediction (Before and After Impact)")

    # 相位图
    theta_dot_pinn = np.gradient(theta_pinn, t_np)
    axes[1, 1].plot(theta_pinn, theta_dot_pinn, "b-", linewidth=1)
    axes[1, 1].set_xlabel("Angle (rad)")
    axes[1, 1].set_ylabel("Angular Velocity (rad/s)")
    axes[1, 1].grid(True)
    axes[1, 1].set_title("Phase Portrait")

    plt.tight_layout()
    plt.savefig("pendulum_test_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\n图片已保存: pendulum_test_results.png")

    # 检查是否需要重新训练
    if l2_error > 0.01:
        print("\n⚠️ 警告: 误差较大，模型可能需要重新训练")
        print("建议:")
        print("  1. 增加训练轮数到 20000")
        print("  2. 增加配点数到 1000")
        print("  3. 调整损失权重")
    else:
        print("\n✅ 模型表现良好！")


if __name__ == "__main__":
    main()
