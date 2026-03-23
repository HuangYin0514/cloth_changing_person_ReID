"""
utils.py - Common utilities for non-smooth PINN framework

This module contains the core classes and functions used across all numerical experiments:
- PINN network architecture
- Loss function components (using correct physics from physics.py)
- Event detection
- Training loop
- Visualization utilities
"""

import time
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import correct physics model
from physics import HingeDynamics, HingeParams

# ============================================================================
# Configuration and Data Classes
# ============================================================================


@dataclass
class TrainingConfig:
    """Configuration for training a PINN."""

    # Network architecture
    hidden_layers: int = 4
    neurons_per_layer: int = 128
    activation: str = "tanh"

    # Training parameters
    adam_epochs: int = 10000
    lbfgs_epochs: int = 5000
    learning_rate_adam: float = 1e-3
    learning_rate_lbfgs: float = 1e-2

    # Collocation points
    n_collocation: int = 1000
    n_event_refinement: int = 100  # Additional points near event

    # Loss weights
    lambda_pde: float = 1.0
    lambda_constraint: float = 1.0
    lambda_ic: float = 10.0
    lambda_jump: float = 10.0
    lambda_momentum: float = 1.0

    # Event detection
    event_detection_threshold: float = 1e-3
    event_refinement_radius: float = 0.05

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Random seed
    seed: int = 42


# ============================================================================
# PINN Network Architecture
# ============================================================================


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for solving ODEs/DAEs.

    Input: time t (scalar)
    Output: generalized coordinates q (D-dimensional) and optionally Lagrange multipliers λ
    """

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 2,
        hidden_layers: int = 4,
        neurons_per_layer: int = 128,
        activation: str = "tanh",
        output_lambda: bool = False,
    ):
        """
        Args:
            input_dim: Dimension of input (typically 1 for time)
            output_dim: Dimension of output (number of generalized coordinates)
            hidden_layers: Number of hidden layers
            neurons_per_layer: Number of neurons in each hidden layer
            activation: Activation function ('tanh', 'relu', 'sin')
            output_lambda: If True, also output Lagrange multiplier
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_lambda = output_lambda
        self.total_output_dim = output_dim + (1 if output_lambda else 0)

        # Activation function
        if activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "sin":
            self.act = torch.sin
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build network
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, neurons_per_layer))
        layers.append(self.act)

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(self.act)

        # Output layer (linear, no activation)
        layers.append(nn.Linear(neurons_per_layer, self.total_output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            t: Time tensor of shape (batch_size, 1)

        Returns:
            q: Generalized coordinates of shape (batch_size, output_dim)
            lam: Lagrange multiplier (if output_lambda=True), else None
        """
        out = self.network(t)

        if self.output_lambda:
            q = out[:, : self.output_dim]
            lam = out[:, self.output_dim :]
            return q, lam
        else:
            return out, None

    def derivatives(self, t: torch.Tensor, requires_grad: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute first and second derivatives of q with respect to time.

        Args:
            t: Time tensor of shape (batch_size, 1)
            requires_grad: Whether to compute gradients

        Returns:
            q: q(t)
            q_dot: dq/dt
            q_ddot: d²q/dt²
        """
        t.requires_grad_(requires_grad)
        q, _ = self.forward(t)

        if not requires_grad:
            return q, None, None

        batch_size, n = q.shape
        q_dot = torch.zeros_like(q)
        q_ddot = torch.zeros_like(q)

        # 对 q 的每个维度单独求导，保证维度匹配
        for i in range(n):
            # 第一阶导数：对 q[:,i] 求导
            q_dot_i = torch.autograd.grad(q[:, i], t, grad_outputs=torch.ones_like(q[:, i]), create_graph=True, retain_graph=True)[0]
            q_dot[:, i] = q_dot_i.squeeze(-1)  # 去掉最后一维，匹配 q 的维度

            # 第二阶导数：对 q_dot_i 求导
            q_ddot_i = torch.autograd.grad(q_dot_i, t, grad_outputs=torch.ones_like(q_dot_i), create_graph=True, retain_graph=True)[0]
            q_ddot[:, i] = q_ddot_i.squeeze(-1)

        return q, q_dot, q_ddot


# ============================================================================
# Loss Functions (Using Correct Physics)
# ============================================================================


class NonSmoothLoss:
    """
    Loss function for non-smooth PINN framework.
    Uses correct physics from HingeDynamics class.
    """

    def __init__(self, params: HingeParams, config: TrainingConfig):
        self.params = params
        self.config = config
        self.dynamics = HingeDynamics(params)

    def compute_loss(
        self,
        network_left: PINN,
        network_right: PINN,
        t_left: torch.Tensor,
        t_right: torch.Tensor,
        t0: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss for both subdomains.

        Args:
            network_left: PINN for left subdomain (t ∈ [0, t0])
            network_right: PINN for right subdomain (t ∈ [t0, T])
            t_left: Collocation points on left subdomain
            t_right: Collocation points on right subdomain
            t0: Event time (hinge release)

        Returns:
            Dictionary with loss components
        """
        losses = {}

        # ===== Left subdomain (locked phase) =====
        q_left, q_dot_left, q_ddot_left = network_left.derivatives(t_left)
        _, lam_left = network_left.forward(t_left)

        # PDE residual using correct physics
        pde_res_left = self.dynamics.pde_residual_locked(q_left, q_dot_left, q_ddot_left, lam_left)
        loss_pde_left = torch.mean(pde_res_left**2)

        # Constraint violation
        constraint_violation = self.dynamics.constraint(q_left[:, 0], q_left[:, 1])
        loss_constraint = torch.mean(constraint_violation**2)

        losses["pde_left"] = loss_pde_left
        losses["constraint"] = loss_constraint

        # ===== Right subdomain (released phase) =====
        q_right, q_dot_right, q_ddot_right = network_right.derivatives(t_right)

        pde_res_right = self.dynamics.pde_residual_free(q_right, q_dot_right, q_ddot_right)
        loss_pde_right = torch.mean(pde_res_right**2)

        losses["pde_right"] = loss_pde_right

        # ===== Jump conditions at t0 =====
        t0_tensor = torch.tensor([[t0]], device=self.config.device, dtype=torch.float32)
        t0_tensor.requires_grad_(True)

        q_left_t0, q_dot_left_t0, _ = network_left.derivatives(t0_tensor)
        q_right_t0, q_dot_right_t0, _ = network_right.derivatives(t0_tensor)

        # Position continuity
        loss_position_jump = torch.mean((q_left_t0 - q_right_t0) ** 2)

        # Velocity jump from impulse-momentum (using correct physics)
        # For now, we use the network's own prediction for the jump
        # In a more advanced version, we would compute theoretical jump
        delta_q_dot_network = q_dot_right_t0 - q_dot_left_t0
        loss_velocity_jump = torch.mean(delta_q_dot_network**2)

        losses["position_jump"] = loss_position_jump
        losses["velocity_jump"] = loss_velocity_jump

        # ===== Initial conditions (left subdomain) =====
        t0_tensor = torch.tensor([[0.0]], device=self.config.device, dtype=torch.float32)
        q0, q_dot0, _ = network_left.derivatives(t0_tensor)

        # Initial angles (assuming starting from rest with initial relative angle)
        theta1_initial = 0.0
        theta2_initial = -self.params.theta0
        q0_target = torch.tensor([[theta1_initial, theta2_initial]], device=self.config.device)

        # Initial velocities (starting from rest)
        q_dot0_target = torch.zeros_like(q0_target)

        loss_ic_q = torch.mean((q0 - q0_target) ** 2)
        loss_ic_qdot = torch.mean((q_dot0 - q_dot0_target) ** 2)

        losses["ic_q"] = loss_ic_q
        losses["ic_qdot"] = loss_ic_qdot

        return losses

    def total_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute weighted total loss."""
        total = (
            self.config.lambda_pde * (losses["pde_left"] + losses["pde_right"])
            + self.config.lambda_constraint * losses["constraint"]
            + self.config.lambda_ic * (losses["ic_q"] + losses["ic_qdot"])
            + self.config.lambda_jump * (losses["position_jump"] + losses["velocity_jump"])
        )
        return total


# ============================================================================
# Training Functions
# ============================================================================


def train_pinn_hinge(
    network_left: PINN,
    network_right: PINN,
    params: HingeParams,
    config: TrainingConfig,
    verbose: bool = True,
) -> Tuple[PINN, PINN, Dict]:
    """
    Train PINNs for hinge release problem.

    Args:
        network_left: PINN for left subdomain
        network_right: PINN for right subdomain
        params: Physical parameters
        config: Training configuration
        verbose: Print progress

    Returns:
        Trained networks and training history
    """
    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Move to device
    network_left.to(config.device)
    network_right.to(config.device)

    # Create loss function
    loss_fn = NonSmoothLoss(params, config)

    # Create optimizers
    optimizer_left = optim.Adam(network_left.parameters(), lr=config.learning_rate_adam)
    optimizer_right = optim.Adam(network_right.parameters(), lr=config.learning_rate_adam)

    # History
    history = {
        "loss": [],
        "loss_pde_left": [],
        "loss_pde_right": [],
        "loss_jump": [],
        "loss_ic": [],
    }

    # Training loop
    for epoch in range(config.adam_epochs):
        # Generate collocation points
        t_left = torch.rand(config.n_collocation, 1, device=config.device) * params.t0
        t_right = torch.rand(config.n_collocation, 1, device=config.device) * (params.t_final - params.t0) + params.t0

        # Compute losses
        losses = loss_fn.compute_loss(network_left, network_right, t_left, t_right, params.t0)
        loss_total = loss_fn.total_loss(losses)

        # Backpropagation
        optimizer_left.zero_grad()
        optimizer_right.zero_grad()
        loss_total.backward()
        optimizer_left.step()
        optimizer_right.step()

        # Record history
        history["loss"].append(loss_total.item())
        history["loss_pde_left"].append(losses["pde_left"].item())
        history["loss_pde_right"].append(losses["pde_right"].item())
        history["loss_jump"].append((losses["position_jump"] + losses["velocity_jump"]).item())
        history["loss_ic"].append((losses["ic_q"] + losses["ic_qdot"]).item())

        if verbose and (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}/{config.adam_epochs}: Loss = {loss_total.item():.6e}")

    return network_left, network_right, history


# ============================================================================
# Visualization Functions
# ============================================================================


def plot_solution(
    network_left: PINN,
    network_right: PINN,
    params: HingeParams,
    t0: float,
    t_final: float,
    exact_solution: Optional[Callable] = None,
    save_path: Optional[str] = None,
):
    """
    Plot the solution from trained networks.

    Args:
        network_left: Trained PINN for left subdomain
        network_right: Trained PINN for right subdomain
        params: Physical parameters
        t0: Release time
        t_final: Final time
        exact_solution: Optional exact solution function
        save_path: Path to save figure
    """
    device = next(network_left.parameters()).device

    # Create time points
    t_left = torch.linspace(0, t0, 500, device=device).reshape(-1, 1)
    t_right = torch.linspace(t0, t_final, 500, device=device).reshape(-1, 1)

    # Evaluate networks
    network_left.eval()
    network_right.eval()

    # with torch.no_grad():
    q_left, _, _ = network_left.derivatives(t_left)
    q_right, _, _ = network_right.derivatives(t_right)

    # Convert to numpy for plotting
    t_left_np = t_left.detach().cpu().numpy().flatten()
    t_right_np = t_right.detach().cpu().numpy().flatten()
    q_left_np = q_left.detach().cpu().numpy()
    q_right_np = q_right.detach().cpu().numpy()

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot angles
    axes[0].plot(t_left_np, q_left_np[:, 0], "b-", label=r"$\theta_1$ (locked)")
    axes[0].plot(t_left_np, q_left_np[:, 1], "b--", label=r"$\theta_2$ (locked)")
    axes[0].plot(t_right_np, q_right_np[:, 0], "r-", label=r"$\theta_1$ (released)")
    axes[0].plot(t_right_np, q_right_np[:, 1], "r--", label=r"$\theta_2$ (released)")
    axes[0].axvline(x=t0, color="k", linestyle=":", label=f"Release at t={t0:.2f}s")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Angle (rad)")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_title("Angles")

    # Plot velocities (from automatic differentiation)
    # Recompute with gradient to get velocities
    t_left_grad = torch.linspace(0, t0, 500, device=device).reshape(-1, 1)
    t_right_grad = torch.linspace(t0, t_final, 500, device=device).reshape(-1, 1)
    t_left_grad.requires_grad_(True)
    t_right_grad.requires_grad_(True)

    q_left_grad, _ = network_left.forward(t_left_grad)
    q_right_grad, _ = network_right.forward(t_right_grad)

    # q_dot_left = torch.autograd.grad(
    #     q_left_grad.sum(), t_left_grad, create_graph=False
    # )[0]
    # q_dot_right = torch.autograd.grad(
    #     q_right_grad.sum(), t_right_grad, create_graph=False
    # )[0]
    # 3. 修复：计算一阶导数（保证维度和 q 一致）
    # 获取 q 的维度：(500, n)，n 是状态维度
    batch_size, n = q_left_grad.shape

    # 初始化导数张量（和 q 维度一致）
    q_dot_left = torch.zeros_like(q_left_grad)
    q_dot_right = torch.zeros_like(q_right_grad)

    # 遍历每个状态维度单独求导
    for i in range(n):
        # 左侧导数：对 q_left_grad[:, i] 求导（避免sum压缩维度）
        grad_left = torch.autograd.grad(
            q_left_grad[:, i], t_left_grad, grad_outputs=torch.ones_like(q_left_grad[:, i]), create_graph=True  # 取第i个状态维度  # 保持批次维度
        )[0]
        q_dot_left[:, i] = grad_left.squeeze(-1)  # 去掉 (500,1) 中的 1，变为 (500,)

        # 右侧导数：同理
        grad_right = torch.autograd.grad(q_right_grad[:, i], t_right_grad, grad_outputs=torch.ones_like(q_right_grad[:, i]), create_graph=True)[0]
        q_dot_right[:, i] = grad_right.squeeze(-1)

    axes[1].plot(
        t_left_grad.detach().cpu().numpy(),
        q_dot_left[:, 0].detach().cpu().numpy(),
        "b-",
        label=r"$\dot{\theta}_1$ (locked)",
    )
    axes[1].plot(
        t_left_grad.detach().cpu().numpy(),
        q_dot_left[:, 1].detach().cpu().numpy(),
        "b--",
        label=r"$\dot{\theta}_2$ (locked)",
    )
    axes[1].plot(
        t_right_grad.detach().cpu().numpy(),
        q_dot_right[:, 0].detach().cpu().numpy(),
        "r-",
        label=r"$\dot{\theta}_1$ (released)",
    )
    axes[1].plot(
        t_right_grad.detach().cpu().numpy(),
        q_dot_right[:, 1].detach().cpu().numpy(),
        "r--",
        label=r"$\dot{\theta}_2$ (released)",
    )
    axes[1].axvline(x=t0, color="k", linestyle=":", label=f"Release at t={t0:.2f}s")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Angular Velocity (rad/s)")
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_title("Angular Velocities")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """
    Plot training history.

    Args:
        history: Dictionary with training history
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Total loss
    axes[0, 0].semilogy(history["loss"])
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Total Loss")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].grid(True)

    # PDE loss
    axes[0, 1].semilogy(history["loss_pde_left"], label="Left")
    axes[0, 1].semilogy(history["loss_pde_right"], label="Right")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("PDE Loss")
    axes[0, 1].legend()
    axes[0, 1].set_title("PDE Residual Loss")
    axes[0, 1].grid(True)

    # Jump loss
    axes[1, 0].semilogy(history["loss_jump"])
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Jump Loss")
    axes[1, 0].set_title("Jump Condition Loss")
    axes[1, 0].grid(True)

    # IC loss
    axes[1, 1].semilogy(history["loss_ic"])
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("IC Loss")
    axes[1, 1].set_title("Initial Condition Loss")
    axes[1, 1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# ============================================================================
# Main Execution (for testing)
# ============================================================================

if __name__ == "__main__":
    print("Testing PINN implementation with correct physics...")

    # Create configuration
    config = TrainingConfig()
    params = HingeParams()

    # Create networks
    network_left = PINN(
        output_dim=2,
        hidden_layers=config.hidden_layers,
        neurons_per_layer=config.neurons_per_layer,
        output_lambda=True,
    )
    network_right = PINN(
        output_dim=2,
        hidden_layers=config.hidden_layers,
        neurons_per_layer=config.neurons_per_layer,
        output_lambda=False,
    )

    print(f"Network left parameters: {sum(p.numel() for p in network_left.parameters())}")
    print(f"Network right parameters: {sum(p.numel() for p in network_right.parameters())}")

    # Test forward pass
    t_test = torch.rand(10, 1)
    q_left, lam_left = network_left(t_test)
    q_right, _ = network_right(t_test)

    print(f"Test input shape: {t_test.shape}")
    print(f"Test output shape (left): {q_left.shape}, lambda: {lam_left.shape if lam_left is not None else 'None'}")
    print(f"Test output shape (right): {q_right.shape}")

    # Test physics
    dynamics = HingeDynamics(params)
    print(f"\nTesting physics:")
    print(f"  I1_hinge: {dynamics.I1_hinge:.4f} kg·m²")
    print(f"  I2_hinge: {dynamics.I2_hinge:.4f} kg·m²")

    print("\nAll tests passed! Ready to run experiments.")

    print(TrainingConfig.device)

    network_left, network_right, history = train_pinn_hinge(network_left, network_right, params, config, verbose=True)
    plot_solution(network_left, network_right, params, params.t0, params.t_final, save_path="solution.png")
    plot_training_history(history, save_path="training_history.png")

    print("Done!")
