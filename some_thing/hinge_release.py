"""
hinge_release.py - Hinge Release Experiment (Fixed)
"""

import time
from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ============================================================================
# Physical Parameters
# ============================================================================


@dataclass
class HingeReleaseParams:
    """Physical parameters for hinge release problem."""

    l1: float = 0.5
    l2: float = 0.5
    m1: float = 2.0
    m2: float = 2.0
    I1: float = 0.5
    I2: float = 0.5
    theta0: float = np.pi / 6
    t0: float = 0.5
    t_final: float = 1.5
    g: float = 9.81
    tau1: float = 0.0
    tau2: float = 0.0

    def inertia_about_hinge(self):
        I1_hinge = self.I1 + self.m1 * self.l1**2
        I2_hinge = self.I2 + self.m2 * self.l2**2
        return I1_hinge, I2_hinge


class HingeReleaseDynamics:
    """Correct dynamics for hinge release problem."""

    def __init__(self, params: HingeReleaseParams):
        self.params = params
        self.I1, self.I2 = params.inertia_about_hinge()

    def mass_matrix(self) -> torch.Tensor:
        M = torch.zeros(2, 2)
        M[0, 0] = self.I1
        M[1, 1] = self.I2
        return M

    def gravity_torques(self, theta1: torch.Tensor, theta2: torch.Tensor) -> torch.Tensor:
        tau1 = -self.params.m1 * self.params.g * self.params.l1 * torch.cos(theta1)
        tau2 = -self.params.m2 * self.params.g * self.params.l2 * torch.cos(theta2)
        return torch.stack([tau1, tau2], dim=0)

    def constraint(self, theta1: torch.Tensor, theta2: torch.Tensor) -> torch.Tensor:
        return theta1 - theta2 - self.params.theta0

    def constraint_jacobian(self) -> torch.Tensor:
        return torch.tensor([[1.0, -1.0]])

    def pde_residual_locked(self, q: torch.Tensor, q_dot: torch.Tensor, q_ddot: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """
        PDE residual for locked phase.
        Residual = M * q_ddot + G - F_ext - J^T * λ
        """
        # 处理形状：支持 (batch,2) 和 (2,) 两种输入
        if q_ddot.dim() == 1:
            theta1_ddot = q_ddot[0]
            theta2_ddot = q_ddot[1]
            batch_size = 1
        else:
            theta1_ddot = q_ddot[:, 0]
            theta2_ddot = q_ddot[:, 1]
            batch_size = q_ddot.shape[0]

        M = self.mass_matrix().to(q.device)
        G = self.gravity_torques(q[:, 0], q[:, 1])
        J = self.constraint_jacobian().to(q.device)

        F_ext = torch.tensor([[self.params.tau1], [self.params.tau2]], device=q.device, dtype=q.dtype)

        # M * q_ddot
        Mq_ddot = torch.stack([M[0, 0] * theta1_ddot, M[1, 1] * theta2_ddot], dim=0)

        # 处理 lambda 形状
        if lam.dim() == 1:
            lam = lam.unsqueeze(1)

        # Residual
        if batch_size == 1:
            residual = Mq_ddot + G - F_ext - J.T @ lam
            return residual.T
        else:
            residual = Mq_ddot + G - F_ext - J.T @ lam.T
            return residual.T

    def pde_residual_free(self, q: torch.Tensor, q_dot: torch.Tensor, q_ddot: torch.Tensor) -> torch.Tensor:
        """
        PDE residual for free phase.
        Residual = M * q_ddot + G - F_ext
        """
        if q_ddot.dim() == 1:
            theta1_ddot = q_ddot[0]
            theta2_ddot = q_ddot[1]
            batch_size = 1
        else:
            theta1_ddot = q_ddot[:, 0]
            theta2_ddot = q_ddot[:, 1]
            batch_size = q_ddot.shape[0]

        M = self.mass_matrix().to(q.device)
        G = self.gravity_torques(q[:, 0], q[:, 1])

        F_ext = torch.tensor([[self.params.tau1], [self.params.tau2]], device=q.device, dtype=q.dtype)

        Mq_ddot = torch.stack([M[0, 0] * theta1_ddot, M[1, 1] * theta2_ddot], dim=0)

        residual = Mq_ddot + G - F_ext

        if batch_size == 1:
            return residual.T
        else:
            return residual.T


# ============================================================================
# Neural Network
# ============================================================================


class HingeReleaseNet(nn.Module):
    def __init__(self, hidden_layers: int = 3, neurons: int = 64, output_lambda: bool = False):
        super().__init__()
        self.output_lambda = output_lambda
        output_dim = 2 + (1 if output_lambda else 0)

        layers = [nn.Linear(1, neurons), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons, neurons))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(neurons, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
        out = self.net(t)
        q = out[:, :2]
        lam = out[:, 2:] if self.output_lambda else None
        return q, lam

    def derivatives(self, t: torch.Tensor) -> Tuple:
        t.requires_grad_(True)
        q, lam = self.forward(t)

        q_dot = torch.autograd.grad(q, t, torch.ones_like(q), create_graph=True, retain_graph=True)[0]
        q_ddot = torch.autograd.grad(q_dot, t, torch.ones_like(q_dot), create_graph=True)[0]

        lam_dot = None
        if lam is not None:
            lam_dot = torch.autograd.grad(lam, t, torch.ones_like(lam), create_graph=True)[0]

        return q, q_dot, q_ddot, lam_dot


# ============================================================================
# Loss Function
# ============================================================================


class HingeReleaseLoss:
    def __init__(self, params: HingeReleaseParams, dynamics: HingeReleaseDynamics, device: torch.device):
        self.params = params
        self.dynamics = dynamics
        self.device = device
        self.lambda_pde = 0.01
        self.lambda_ic = 1.0
        self.lambda_jump = 1.0
        self.lambda_constraint = 1.0

    def compute_loss(self, net_left, net_right, t_left, t_right, t0):
        losses = {}

        # Left subdomain (locked)
        q_left, q_dot_left, q_ddot_left, lam_left = net_left.derivatives(t_left)
        pde_left = self.dynamics.pde_residual_locked(q_left, q_dot_left, q_ddot_left, lam_left)
        loss_pde_left = torch.mean(pde_left**2)

        # Constraint violation
        constraint_val = self.dynamics.constraint(q_left[:, 0], q_left[:, 1])
        loss_constraint = torch.mean(constraint_val**2)

        # Right subdomain (free)
        q_right, q_dot_right, q_ddot_right, _ = net_right.derivatives(t_right)
        pde_right = self.dynamics.pde_residual_free(q_right, q_dot_right, q_ddot_right)
        loss_pde_right = torch.mean(pde_right**2)

        loss_pde = (loss_pde_left + loss_pde_right) / 2.0
        losses["pde"] = loss_pde
        losses["constraint"] = loss_constraint

        # Jump conditions
        t0_tensor = torch.tensor([[t0]], device=self.device, dtype=torch.float32)
        q_left_t0, q_dot_left_t0, _, _ = net_left.derivatives(t0_tensor)
        q_right_t0, q_dot_right_t0, _, _ = net_right.derivatives(t0_tensor)

        loss_pos_jump = torch.mean((q_left_t0 - q_right_t0) ** 2)

        # Velocity jump (simplified: both bodies maintain angular momentum)
        theta1_dot_before = q_dot_left_t0[:, 0]
        theta2_dot_before = q_dot_left_t0[:, 1]
        theta_dot_after = (theta1_dot_before + theta2_dot_before) / 2.0
        theta_dot_target = torch.stack([theta_dot_after, theta_dot_after], dim=1)
        loss_vel_jump = torch.mean((q_dot_right_t0 - theta_dot_target) ** 2)

        losses["jump"] = loss_pos_jump + loss_vel_jump

        # Initial conditions
        t_ic = torch.zeros(1, 1, device=self.device)
        q_ic, q_dot_ic, _, _ = net_left.derivatives(t_ic)

        theta1_ic = 0.0
        theta2_ic = -self.params.theta0
        loss_ic_theta = torch.mean((q_ic[:, 0] - theta1_ic) ** 2 + (q_ic[:, 1] - theta2_ic) ** 2)
        loss_ic_vel = torch.mean(q_dot_ic**2)
        losses["ic"] = (loss_ic_theta + loss_ic_vel) / 2.0

        return losses

    def total_loss(self, losses):
        return self.lambda_pde * losses["pde"] + self.lambda_ic * losses["ic"] + self.lambda_jump * losses["jump"] + self.lambda_constraint * losses["constraint"]


# ============================================================================
# Training
# ============================================================================


def train_hinge_release(params: HingeReleaseParams, epochs: int = 10000, n_coll: int = 500, lr: float = 1e-3, verbose: bool = True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    net_left = HingeReleaseNet(hidden_layers=3, neurons=64, output_lambda=True).to(device)
    net_right = HingeReleaseNet(hidden_layers=3, neurons=64, output_lambda=False).to(device)

    dynamics = HingeReleaseDynamics(params)
    loss_fn = HingeReleaseLoss(params, dynamics, device)

    optimizer = torch.optim.Adam(list(net_left.parameters()) + list(net_right.parameters()), lr=lr)

    t0 = params.t0
    history = {"loss": [], "pde": [], "ic": [], "jump": [], "constraint": []}

    print(f"\n{'='*60}")
    print(f"Hinge Release Training")
    print(f"Release time: t0 = {t0:.3f} s")
    print(f"Training epochs: {epochs}")
    print(f"Collocation points: {n_coll}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(epochs):
        t_left = torch.rand(n_coll, 1, device=device) * t0
        t_right = torch.rand(n_coll, 1, device=device) * (params.t_final - t0) + t0

        losses = loss_fn.compute_loss(net_left, net_right, t_left, t_right, t0)
        loss_total = loss_fn.total_loss(losses)

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        history["loss"].append(loss_total.item())
        history["pde"].append(losses["pde"].item())
        history["ic"].append(losses["ic"].item())
        history["jump"].append(losses["jump"].item())
        history["constraint"].append(losses["constraint"].item())

        if verbose and (epoch + 1) % 1000 == 0:
            print(
                f"Epoch {epoch+1:5d}: Loss={loss_total.item():.4e}, "
                f"PDE={losses['pde'].item():.4e}, "
                f"IC={losses['ic'].item():.4e}, "
                f"Jump={losses['jump'].item():.4e}"
            )

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f} seconds")

    return net_left, net_right, history


# ============================================================================
# Plotting
# ============================================================================


def plot_results(net_left, net_right, params, history, save_path="hinge_release_results.png"):
    device = next(net_left.parameters()).device
    net_left.eval()
    net_right.eval()

    t0 = params.t0
    t_final = params.t_final

    t_left = torch.linspace(0, t0, 500, device=device).reshape(-1, 1)
    t_right = torch.linspace(t0, t_final, 500, device=device).reshape(-1, 1)

    with torch.no_grad():
        q_left, _, _, _ = net_left.derivatives(t_left)
        q_right, _, _, _ = net_right.derivatives(t_right)

    t_np = np.concatenate([t_left.cpu().numpy().flatten(), t_right.cpu().numpy().flatten()])
    theta1 = np.concatenate([q_left[:, 0].cpu().numpy(), q_right[:, 0].cpu().numpy()])
    theta2 = np.concatenate([q_left[:, 1].cpu().numpy(), q_right[:, 1].cpu().numpy()])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Angles
    axes[0, 0].plot(t_np, theta1, "b-", label=r"$\theta_1$")
    axes[0, 0].plot(t_np, theta2, "r-", label=r"$\theta_2$")
    axes[0, 0].axvline(x=t0, color="k", linestyle=":", label=f"Release at t={t0:.2f}s")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Angle (rad)")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_title("Angles vs Time")

    # Relative angle
    rel_angle = theta1 - theta2
    axes[0, 1].plot(t_np, rel_angle, "g-")
    axes[0, 1].axhline(y=params.theta0, color="b", linestyle="--", label=f"Locked: {params.theta0:.3f}")
    axes[0, 1].axvline(x=t0, color="k", linestyle=":")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Relative Angle (rad)")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].set_title("Relative Angle (θ₁ - θ₂)")

    # Loss history
    axes[1, 0].semilogy(history["loss"], label="Total")
    axes[1, 0].semilogy(history["pde"], label="PDE")
    axes[1, 0].semilogy(history["ic"], label="IC")
    axes[1, 0].semilogy(history["jump"], label="Jump")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_title("Training Loss")

    # Constraint violation
    axes[1, 1].semilogy(history["constraint"])
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Constraint Violation")
    axes[1, 1].grid(True)
    axes[1, 1].set_title("Constraint Violation")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Figure saved to: {save_path}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Hinge Release Experiment")
    print("=" * 60)

    params = HingeReleaseParams()
    net_left, net_right, history = train_hinge_release(params, epochs=5000, n_coll=300, lr=1e-3)

    torch.save(net_left.state_dict(), "hinge_release_left.pth")
    torch.save(net_right.state_dict(), "hinge_release_right.pth")
    print("Models saved")

    plot_results(net_left, net_right, params, history)
    print("Done!")
