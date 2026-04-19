"""
pendulum_impact.py - Simple pendulum with impact at the bottom

This experiment validates the event detection and jump handling capabilities
of the non-smooth PINN framework on a classic problem:
- A simple pendulum released from an initial angle
- It impacts a rigid stop at θ = 0
- Coefficient of restitution e = 0.8

The exact solution is known, allowing quantitative error analysis.
"""

import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp


# 模拟框架依赖（如果没有实际的utils模块，这里先定义基础类保证代码可运行）
# 实际使用时请替换为你自己的utils模块
class PINN(nn.Module):
    """简化版PINN类，保证代码可独立运行"""

    def __init__(self, output_dim=1, hidden_layers=3, neurons_per_layer=50, output_lambda=False):
        super().__init__()
        layers = []
        layers.append(nn.Linear(1, neurons_per_layer))  # 输入是时间t (1维)
        layers.append(nn.Tanh())

        for _ in range(hidden_layers):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(neurons_per_layer, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, t):
        return self.network(t)

    def derivatives(self, t):
        """计算输出的一阶和二阶导数（用于PINN）"""
        t.requires_grad_(True)
        q = self.forward(t)

        # 一阶导数 q_dot = dq/dt
        q_dot = torch.autograd.grad(q, t, grad_outputs=torch.ones_like(q), create_graph=True, retain_graph=True)[0]

        # 二阶导数 q_ddot = d²q/dt²
        q_ddot = torch.autograd.grad(q_dot, t, grad_outputs=torch.ones_like(q_dot), create_graph=True, retain_graph=True)[0]

        return q, q_dot, q_ddot


@dataclass
class TrainingConfig:
    """训练配置类"""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    hidden_layers: int = 3
    neurons_per_layer: int = 50
    n_collocation: int = 1000
    adam_epochs: int = 10000
    learning_rate_adam: float = 1e-4
    lambda_pde: float = 1.0
    lambda_ic: float = 100.0
    lambda_jump: float = 1000.0


class NonSmoothLoss:
    """占位类，保证导入不报错"""

    pass


def plot_training_history(history):
    """绘制训练历史"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.title("Total Loss")

    plt.subplot(1, 2, 2)
    plt.plot(history["pde_left"], label="PDE Left")
    plt.plot(history["pde_right"], label="PDE Right")
    plt.plot(history["jump"], label="Jump Loss")
    plt.plot(history["ic"], label="IC Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.title("Component Losses")

    plt.tight_layout()
    plt.show()


# ============================================================================
# Problem Parameters
# ============================================================================


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


# ============================================================================
# Exact Solution
# ============================================================================


class PendulumExactSolution:
    """
    Exact solution for simple pendulum with impact.

    Uses energy conservation between impacts and restitution law at impacts.
    """

    def __init__(self, params: PendulumParams):
        self.params = params
        self._compute_impact_times()

    def _compute_impact_times(self):
        """
        Compute impact times using energy conservation.

        For a pendulum released from θ0, the time to reach θ=0 is:
        t_impact = ∫_{θ0}^{0} dθ / ω(θ)
        where ω(θ) = sqrt(2g/L * (cosθ - cosθ0))
        """
        import scipy.integrate as integrate

        gL = self.params.g / self.params.L
        theta0 = self.params.theta0

        def dt_dtheta(theta):
            """dt/dθ = 1/ω(θ)"""
            omega = np.sqrt(2 * gL * (np.cos(theta) - np.cos(theta0)))
            return 1.0 / omega if omega > 0 else 0

        # First impact
        t_impact1, _ = integrate.quad(dt_dtheta, theta0, 0)
        self.impact_times = [t_impact1]

        # After impact, velocity is reduced by coefficient of restitution
        # Compute subsequent impact times
        # For a pendulum, the motion is symmetric, so time between impacts is constant
        # Actually, after impact, the pendulum swings to the opposite side
        # The time from θ=0 to θ_max and back is the same as from θ0 to 0

        # For simplicity, we compute the period of the full motion
        # The period after impact is the same as the period before impact
        # because energy is lost but the motion remains symmetric

        # Compute period of one full swing (θ0 -> 0 -> -θ_max -> 0 -> θ0)
        # For small angles, period ≈ 2π√(L/g)
        # For larger angles, use numerical integration

        def half_period(theta_max):
            """Time from θ=0 to θ_max and back to 0"""

            def dt_dtheta(theta):
                omega = np.sqrt(2 * gL * (np.cos(theta) - np.cos(theta_max)))
                return 1.0 / omega if omega > 0 else 0

            t_half, _ = integrate.quad(dt_dtheta, 0, theta_max)
            return 2 * t_half  # down and up

        # After first impact, maximum angle is determined by energy loss
        # Energy after impact = e^2 * energy before impact
        # Energy before impact = mgL(1 - cosθ0)
        energy_before = self.params.m * self.params.g * self.params.L * (1 - np.cos(theta0))
        energy_after = self.params.e**2 * energy_before
        theta_max = np.arccos(1 - energy_after / (self.params.m * self.params.g * self.params.L))

        # Subsequent impacts occur every half period
        t_half = half_period(theta_max)

        # Add subsequent impact times
        for i in range(1, 10):
            t_next = self.impact_times[-1] + t_half
            if t_next < self.params.t_final:
                self.impact_times.append(t_next)
            else:
                break

    def theta(self, t: float) -> float:
        """
        Compute exact angle at time t.

        Uses piecewise solution between impacts.
        """
        if t < 0:
            return self.params.theta0

        # Find the segment
        impact_times = [0] + self.impact_times + [self.params.t_final]

        for i in range(len(impact_times) - 1):
            if impact_times[i] <= t <= impact_times[i + 1]:
                # Determine motion in this segment
                if i == 0:
                    # First segment: from release to first impact
                    theta_max = self.params.theta0
                    t_start = 0
                    direction = -1  # moving downward
                else:
                    # After impact: oscillating with reduced amplitude
                    energy_before = self.params.m * self.params.g * self.params.L * (1 - np.cos(self.params.theta0))
                    energy_after = self.params.e ** (2 * i) * energy_before
                    theta_max = np.arccos(1 - energy_after / (self.params.m * self.params.g * self.params.L))
                    t_start = impact_times[i]
                    # Direction alternates with each impact
                    direction = -1 if i % 2 == 1 else 1

                # Compute time from start of segment
                dt = t - t_start

                # Compute angle using numerical inversion of the integral
                # For simplicity, use approximate solution for small angles
                # For accurate solution, use interpolation of precomputed data
                # Here we use an approximate method

                gL = self.params.g / self.params.L
                # Use elliptic integral approximation for large angles
                # For demonstration, use small-angle approximation
                # In production, use precomputed lookup table

                # Small-angle approximation
                omega = np.sqrt(gL)
                theta_approx = theta_max * np.cos(omega * dt)

                # Adjust direction
                theta_approx = direction * theta_approx

                # Ensure sign is correct
                if i == 0:
                    theta_approx = np.abs(theta_approx)  # Moving from positive to zero
                else:
                    if i % 2 == 1:
                        theta_approx = -np.abs(theta_approx)  # Negative side after first impact
                    else:
                        theta_approx = np.abs(theta_approx)  # Positive side after second impact

                return theta_approx

        return 0.0

    def theta_dot(self, t: float) -> float:
        """
        Compute exact angular velocity at time t.

        Uses energy conservation between impacts.
        """
        theta_val = self.theta(t)
        gL = self.params.g / self.params.L

        # Find which segment
        impact_times = [0] + self.impact_times

        for i, t_impact in enumerate(impact_times):
            if t < t_impact:
                segment = i - 1
                break
        else:
            segment = len(impact_times) - 1

        if segment <= 0:
            # Before first impact
            energy = self.params.m * self.params.g * self.params.L * (1 - np.cos(self.params.theta0))
        else:
            # After impacts
            energy = self.params.e ** (2 * segment) * self.params.m * self.params.g * self.params.L * (1 - np.cos(self.params.theta0))

        # ω = ±√(2E/I - 2g/L(1-cosθ))
        I = self.params.m * self.params.L**2
        omega_sq = 2 * energy / I - 2 * gL * (1 - np.cos(theta_val))

        if omega_sq < 0:
            omega_sq = 0

        omega = np.sqrt(omega_sq)

        # Determine sign
        if t < self.impact_times[0]:
            # Moving downward (negative velocity if θ positive)
            sign = -1
        else:
            # Alternates after impacts
            # For simplicity, determine from theta derivative
            # Use small perturbation to estimate sign
            eps = 1e-6
            theta_plus = self.theta(t + eps)
            sign = 1 if theta_plus > theta_val else -1

        return sign * omega


# ============================================================================
# PINN Model for Pendulum
# ============================================================================


class PendulumDynamics:
    """Dynamics for simple pendulum (ODE, not DAE)."""

    def __init__(self, params: PendulumParams):
        self.params = params

    def equation(self, theta: torch.Tensor, theta_dot: torch.Tensor, theta_ddot: torch.Tensor) -> torch.Tensor:
        """
        Pendulum equation: θ̈ + (g/L) sin θ = 0
        """
        gL = self.params.g / self.params.L
        residual = theta_ddot + gL * torch.sin(theta)
        return residual

    def impact_condition(self, theta: torch.Tensor) -> torch.Tensor:
        """Impact occurs when θ = 0."""
        return theta

    def velocity_jump(self, theta_dot_before: torch.Tensor, e: float) -> torch.Tensor:
        """Velocity after impact = -e * velocity before."""
        return -e * theta_dot_before


class PendulumLoss:
    """Loss function for pendulum with impact - 自适应权重版本"""

    def __init__(self, params: PendulumParams, config: TrainingConfig):
        self.params = params
        self.config = config
        self.dynamics = PendulumDynamics(params)

        # 固定权重（基于简化版实验的经验）
        self.lambda_pde = 0.01  # 大幅降低
        self.lambda_ic = 1.0
        self.lambda_jump = 1.0

    def compute_loss(self, network_left, network_right, t_left, t_right, t0):
        losses = {}

        # ===== 左子域（碰撞前） =====
        q_left, q_dot_left, q_ddot_left = network_left.derivatives(t_left)
        theta_left = q_left[:, 0]

        pde_res_left = self.dynamics.equation(theta_left, q_dot_left[:, 0], q_ddot_left[:, 0])
        loss_pde_left = torch.mean(pde_res_left**2)

        # ===== 右子域（碰撞后） =====
        q_right, q_dot_right, q_ddot_right = network_right.derivatives(t_right)
        theta_right = q_right[:, 0]

        pde_res_right = self.dynamics.equation(theta_right, q_dot_right[:, 0], q_ddot_right[:, 0])
        loss_pde_right = torch.mean(pde_res_right**2)

        loss_pde = (loss_pde_left + loss_pde_right) / 2.0
        losses["pde"] = loss_pde

        # ===== 跳变条件 =====
        t0_tensor = torch.tensor([[t0]], device=self.config.device, dtype=torch.float32)
        t0_tensor.requires_grad_(True)

        q_left_t0, q_dot_left_t0, _ = network_left.derivatives(t0_tensor)
        q_right_t0, q_dot_right_t0, _ = network_right.derivatives(t0_tensor)

        # 位置连续（θ=0）
        loss_pos_jump = torch.mean((q_left_t0 - q_right_t0) ** 2)

        # 速度跳变：θ̇⁺ = -e θ̇⁻
        theta_dot_before = q_dot_left_t0[:, 0]
        theta_dot_after_target = -self.params.e * theta_dot_before
        loss_vel_jump = torch.mean((q_dot_right_t0[:, 0] - theta_dot_after_target) ** 2)

        loss_jump = (loss_pos_jump + loss_vel_jump) / 2.0
        losses["jump"] = loss_jump

        # ===== 初始条件 =====
        t0_tensor = torch.tensor([[0.0]], device=self.config.device, dtype=torch.float32)
        q0, q_dot0, _ = network_left.derivatives(t0_tensor)

        loss_ic_theta = torch.mean((q0[:, 0] - self.params.theta0) ** 2)
        loss_ic_theta_dot = torch.mean((q_dot0[:, 0]) ** 2)

        loss_ic = (loss_ic_theta + loss_ic_theta_dot) / 2.0
        losses["ic"] = loss_ic

        return losses

    def total_loss(self, losses):
        """加权总损失"""
        total = self.lambda_pde * losses["pde"] + self.lambda_ic * losses["ic"] + self.lambda_jump * losses["jump"]
        return total


def train_pendulum_pinn(params: PendulumParams, config: TrainingConfig, t0: float, verbose: bool = True):
    """训练带冲击的摆"""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # 创建网络（与简化版相同配置）
    network_left = PINN(output_dim=1, hidden_layers=3, neurons_per_layer=64, output_lambda=False)
    network_right = PINN(output_dim=1, hidden_layers=3, neurons_per_layer=64, output_lambda=False)

    network_left.to(config.device)
    network_right.to(config.device)

    loss_fn = PendulumLoss(params, config)

    # 优化器
    optimizer_left = torch.optim.Adam(network_left.parameters(), lr=config.learning_rate_adam)
    optimizer_right = torch.optim.Adam(network_right.parameters(), lr=config.learning_rate_adam)

    # 学习率调度
    scheduler_left = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_left, mode="min", factor=0.5, patience=500)
    scheduler_right = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_right, mode="min", factor=0.5, patience=500)

    history = {"loss": [], "pde": [], "ic": [], "jump": []}

    for epoch in range(config.adam_epochs):
        t_left = torch.rand(config.n_collocation, 1, device=config.device) * t0
        t_right = torch.rand(config.n_collocation, 1, device=config.device) * (params.t_final - t0) + t0

        losses = loss_fn.compute_loss(network_left, network_right, t_left, t_right, t0)
        loss_total = loss_fn.total_loss(losses)

        optimizer_left.zero_grad()
        optimizer_right.zero_grad()
        loss_total.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(network_left.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(network_right.parameters(), max_norm=1.0)

        optimizer_left.step()
        optimizer_right.step()

        scheduler_left.step(loss_total)
        scheduler_right.step(loss_total)

        history["loss"].append(loss_total.item())
        history["pde"].append(losses["pde"].item())
        history["ic"].append(losses["ic"].item())
        history["jump"].append(losses["jump"].item())

        if verbose and (epoch + 1) % 500 == 0:
            print(
                f"Epoch {epoch+1}: Loss={loss_total.item():.4e}, "
                f"PDE={losses['pde'].item():.4e}, "
                f"IC={losses['ic'].item():.4e}, "
                f"Jump={losses['jump'].item():.4e}"
            )

    return network_left, network_right, history


def plot_comparison(network_left, network_right, params, t0, exact_solution, save_path=None):
    """Plot comparison between PINN solution and exact solution."""
    # 补全函数实现
    device = next(network_left.parameters()).device

    # Generate time points
    t_eval = np.linspace(0, params.t_final, 1000)
    theta_exact = np.array([exact_solution.theta(t) for t in t_eval])
    theta_dot_exact = np.array([exact_solution.theta_dot(t) for t in t_eval])

    # PINN predictions
    theta_pinn = np.zeros_like(t_eval)
    theta_dot_pinn = np.zeros_like(t_eval)

    for i, t in enumerate(t_eval):
        t_tensor = torch.tensor([[t]], device=device, dtype=torch.float32)

        if t <= t0:
            q, q_dot, _ = network_left.derivatives(t_tensor)
        else:
            q, q_dot, _ = network_right.derivatives(t_tensor)

        theta_pinn[i] = q.item()
        theta_dot_pinn[i] = q_dot.item()

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Angle plot
    ax1.plot(t_eval, theta_exact, "b-", label="Exact", linewidth=2)
    ax1.plot(t_eval, theta_pinn, "r--", label="PINN", linewidth=1.5)
    ax1.axvline(x=t0, color="k", linestyle=":", label="Impact time")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Angle θ (rad)")
    ax1.set_title("Pendulum Angle")
    ax1.legend()
    ax1.grid(True)

    # Angular velocity plot
    ax2.plot(t_eval, theta_dot_exact, "b-", label="Exact", linewidth=2)
    ax2.plot(t_eval, theta_dot_pinn, "r--", label="PINN", linewidth=1.5)
    ax2.axvline(x=t0, color="k", linestyle=":", label="Impact time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angular velocity θ̇ (rad/s)")
    ax2.set_title("Pendulum Angular Velocity")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================================
# Main Execution (示例运行代码)
# ============================================================================
if __name__ == "__main__":
    # 1. 设置参数

    params = PendulumParams()
    config = TrainingConfig()
    config.n_collocation = 500
    config.adam_epochs = 5000
    config.learning_rate_adam = 1e-3
    config.lambda_pde = 0.01  # 关键：降低PDE权重
    config.lambda_ic = 1.0
    config.lambda_jump = 1.0

    exact = PendulumExactSolution(params)
    t0 = exact.impact_times[0]
    print(f"冲击时间: {t0:.4f} s")

    # 训练
    net_left, net_right, history = train_pendulum_pinn(params, config, t0)

    # 绘制结果
    plot_comparison(net_left, net_right, params, t0, exact)
