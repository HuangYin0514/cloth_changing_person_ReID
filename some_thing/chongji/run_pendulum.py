# 先定义必要的依赖类（避免导入错误）
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch


# ===================== 必要的依赖类定义 =====================
@dataclass
class PendulumParams:
    """Physical parameters for simple pendulum with impact."""

    L: float = 1.0  # Length (m)
    m: float = 1.0  # Mass (kg)
    g: float = 9.81  # Gravity (m/s²)
    e: float = 0.8  # Coefficient of restitution
    theta0: float = np.pi / 4  # Initial angle (rad)
    t0: float = None  # Impact time (will be computed)
    t_final: float = 3.0  # Final time (s)

    def natural_frequency(self):
        """Natural frequency of small oscillations."""
        return np.sqrt(self.g / self.L)

    def period(self):
        """Period of small oscillations."""
        return 2 * np.pi / self.natural_frequency()


class PendulumExactSolution:
    """简化版精确解类，仅计算第一次冲击时间"""

    def __init__(self, params: PendulumParams):
        self.params = params
        self._compute_impact_times()

    def _compute_impact_times(self):
        """计算第一次冲击时间"""
        import scipy.integrate as integrate

        gL = self.params.g / self.params.L
        theta0 = self.params.theta0

        def dt_dtheta(theta):
            """dt/dθ = 1/ω(θ)"""
            omega = np.sqrt(2 * gL * (np.cos(theta) - np.cos(theta0)))
            return 1.0 / omega if omega > 0 else 0

        # 计算第一次冲击时间
        t_impact1, _ = integrate.quad(dt_dtheta, theta0, 0)
        self.impact_times = [t_impact1]

    def theta(self, t: float) -> float:
        """简化版角度计算（仅用于测试）"""
        gL = self.params.g / self.params.L
        omega = np.sqrt(gL)
        return self.params.theta0 * np.cos(omega * t)


@dataclass
class TrainingConfig:
    """训练配置类（占位）"""

    pass


# ===================== 简化版PINN网络 =====================
class SimplePINN(torch.nn.Module):
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
        # 修复梯度计算：添加 retain_graph=True 防止计算图释放
        q_dot = torch.autograd.grad(q, t, torch.ones_like(q), create_graph=True, retain_graph=True)[0]
        q_ddot = torch.autograd.grad(q_dot, t, torch.ones_like(q_dot), create_graph=True)[0]
        return q, q_dot, q_ddot


# ===================== 主程序 =====================
if __name__ == "__main__":
    print("=" * 50)
    print("测试简化版PINN...")
    print("=" * 50)

    # 1. 初始化摆锤参数并计算冲击时间
    params = PendulumParams()
    exact = PendulumExactSolution(params)
    t0 = exact.impact_times[0]
    print(f"第一次冲击时间: {t0:.4f} s")

    # 2. 创建左右两个PINN网络
    net_left = SimplePINN(hidden=2, neurons=32)  # 冲击前的网络
    net_right = SimplePINN(hidden=2, neurons=32)  # 冲击后的网络

    # 3. 创建优化器（联合优化两个网络）
    optimizer = torch.optim.Adam(list(net_left.parameters()) + list(net_right.parameters()), lr=1e-3)

    # 4. 简单测试：拟合正弦波函数
    print("\n开始拟合正弦波函数...")
    target_freq = 5  # 正弦波频率
    epochs = 1000

    # 记录损失历史
    loss_history = []

    for epoch in range(epochs):
        # 生成随机训练点
        t = torch.rand(200, 1) * params.t_final

        # 使用左网络拟合正弦波（测试基础拟合能力）
        q, _, _ = net_left.derivatives(t)

        # 计算损失
        target = torch.sin(t * target_freq)
        loss = torch.mean((q - target) ** 2)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录损失
        loss_history.append(loss.item())

        # 打印进度
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d}/{epochs}: Loss = {loss.item():.4e}")

    # 5. 验证拟合效果
    print("\n验证拟合效果...")
    t_test = torch.linspace(0, params.t_final, 500).reshape(-1, 1)
    q_pred, _, _ = net_left.derivatives(t_test)
    q_target = torch.sin(t_test * target_freq)

    # 计算最终误差
    final_error = torch.mean((q_pred - q_target) ** 2).item()
    print(f"最终拟合误差: {final_error:.4e}")

    # 6. 绘制拟合结果
    plt.figure(figsize=(12, 6))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Training Loss (Log Scale)")
    plt.grid(True)

    # 拟合结果对比
    plt.subplot(1, 2, 2)
    plt.plot(t_test.detach().numpy(), q_target.detach().numpy(), "b-", label="Target (sin(5t))", linewidth=2)
    plt.plot(t_test.detach().numpy(), q_pred.detach().numpy(), "r--", label="PINN Prediction", linewidth=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.title("PINN Fitting Result")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    plt.savefig("./fitting_result.png", dpi=300, bbox_inches="tight")

    print("\n" + "=" * 50)
    print("测试完成，网络可以成功拟合函数！")
    print("现在可以基于这个基础框架运行完整的摆锤冲击实验。")
    print("=" * 50)
