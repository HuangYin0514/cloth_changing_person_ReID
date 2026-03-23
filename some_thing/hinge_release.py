"""
hinge_release.py - Hinge Release Experiment

This is the core experiment for the paper, validating the non-smooth PINN
framework on the hinge release problem - a fundamental component of space
deployable structures.

The problem:
- Two rigid bodies connected by a hinge with a locking mechanism
- Locked phase (t < t0): bodies rotate together with constraint θ1 - θ2 = θ0
- Release at t = t0: constraint is removed
- Released phase (t > t0): free rotation of both bodies
- Velocity jump occurs at release due to impulse

Key features:
- Non-smooth dynamics (velocity jump)
- Variable topology (constraint added/removed)
- Jump conditions enforce physical consistency
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
    # Geometry (units: m, kg, s)
    l1: float = 0.5      # Distance from hinge to COM of body 1
    l2: float = 0.5      # Distance from hinge to COM of body 2
    m1: float = 2.0      # Mass of body 1
    m2: float = 2.0      # Mass of body 2
    I1: float = 0.5      # Moment of inertia of body 1 about COM
    I2: float = 0.5      # Moment of inertia of body 2 about COM
    
    # Initial conditions
    theta0: float = np.pi / 6   # Initial relative angle (rad)
    
    # Time parameters
    t0: float = 0.5      # Release time (s)
    t_final: float = 1.5 # Final time (s)
    
    # Gravity (m/s²)
    g: float = 9.81
    
    # External torques (optional)
    tau1: float = 0.0
    tau2: float = 0.0
    
    def inertia_about_hinge(self):
        """Moment of inertia about hinge point (parallel axis theorem)"""
        I1_hinge = self.I1 + self.m1 * self.l1**2
        I2_hinge = self.I2 + self.m2 * self.l2**2
        return I1_hinge, I2_hinge


class HingeReleaseDynamics:
    """Correct dynamics for hinge release problem."""
    
    def __init__(self, params: HingeReleaseParams):
        self.params = params
        self.I1, self.I2 = params.inertia_about_hinge()
    
    def mass_matrix(self) -> torch.Tensor:
        """Mass matrix (diagonal for independent rotation about hinge)."""
        M = torch.zeros(2, 2)
        M[0, 0] = self.I1
        M[1, 1] = self.I2
        return M
    
    def gravity_torques(self, theta1: torch.Tensor, theta2: torch.Tensor) -> torch.Tensor:
        """
        Gravitational torques about hinge point.
        Torque = m * g * l * cos(θ)
        """
        tau1 = -self.params.m1 * self.params.g * self.params.l1 * torch.cos(theta1)
        tau2 = -self.params.m2 * self.params.g * self.params.l2 * torch.cos(theta2)
        return torch.stack([tau1, tau2], dim=0)
    
    def constraint(self, theta1: torch.Tensor, theta2: torch.Tensor) -> torch.Tensor:
        """Holonomic constraint for locked phase."""
        return theta1 - theta2 - self.params.theta0
    
    def constraint_jacobian(self) -> torch.Tensor:
        """Jacobian of constraint: J = [∂Φ/∂θ1, ∂Φ/∂θ2] = [1, -1]."""
        return torch.tensor([[1.0, -1.0]])
    
    def pde_residual_locked(self, q: torch.Tensor, q_dot: torch.Tensor,
                            q_ddot: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """
        PDE residual for locked phase.
        Residual = M * q_ddot + G - F_ext - J^T * λ
        """
        theta1, theta2 = q[:, 0], q[:, 1]
        theta1_ddot, theta2_ddot = q_ddot[:, 0], q_ddot[:, 1]
        
        M = self.mass_matrix().to(q.device)
        G = self.gravity_torques(theta1, theta2)
        J = self.constraint_jacobian().to(q.device)
        
        # External torques
        F_ext = torch.tensor([[self.params.tau1], [self.params.tau2]], 
                            device=q.device, dtype=q.dtype)
        
        # M * q_ddot
        Mq_ddot = torch.stack([
            M[0, 0] * theta1_ddot,
            M[1, 1] * theta2_ddot
        ], dim=0)
        
        # Residual
        residual = Mq_ddot + G - F_ext - J.T @ lam.T
        
        return residual.T
    
    def pde_residual_free(self, q: torch.Tensor, q_dot: torch.Tensor,
                          q_ddot: torch.Tensor) -> torch.Tensor:
        """
        PDE residual for free phase (released).
        Residual = M * q_ddot + G - F_ext
        """
        theta1, theta2 = q[:, 0], q[:, 1]
        theta1_ddot, theta2_ddot = q_ddot[:, 0], q_ddot[:, 1]
        
        M = self.mass_matrix().to(q.device)
        G = self.gravity_torques(theta1, theta2)
        
        F_ext = torch.tensor([[self.params.tau1], [self.params.tau2]],
                            device=q.device, dtype=q.dtype)
        
        Mq_ddot = torch.stack([
            M[0, 0] * theta1_ddot,
            M[1, 1] * theta2_ddot
        ], dim=0)
        
        residual = Mq_ddot + G - F_ext
        return residual.T


# ============================================================================
# Neural Network
# ============================================================================

class HingeReleaseNet(nn.Module):
    """PINN for hinge release problem."""
    
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
        self._init_weights()
    
    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
        out = self.net(t)
        q = out[:, :2]
        lam = out[:, 2:] if self.output_lambda else None
        return q, lam
    
    def derivatives(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Compute q, q_dot, q_ddot, and optionally lambda."""
        t.requires_grad_(True)
        q, lam = self.forward(t)
        
        q_dot = torch.autograd.grad(q, t, torch.ones_like(q), 
                                    create_graph=True, retain_graph=True)[0]
        q_ddot = torch.autograd.grad(q_dot, t, torch.ones_like(q_dot),
                                      create_graph=True)[0]
        
        lam_dot = None
        if lam is not None:
            lam_dot = torch.autograd.grad(lam, t, torch.ones_like(lam),
                                          create_graph=True)[0]
        
        return q, q_dot, q_ddot, lam_dot


# ============================================================================
# Loss Function
# ============================================================================

class HingeReleaseLoss:
    """Loss function with adaptive weighting."""
    
    def __init__(self, params: HingeReleaseParams, dynamics: HingeReleaseDynamics,
                 device: torch.device, lambda_pde: float = 0.01,
                 lambda_ic: float = 1.0, lambda_jump: float = 1.0,
                 lambda_constraint: float = 1.0):
        self.params = params
        self.dynamics = dynamics
        self.device = device
        self.lambda_pde = lambda_pde
        self.lambda_ic = lambda_ic
        self.lambda_jump = lambda_jump
        self.lambda_constraint = lambda_constraint
    
    def compute_loss(self, net_left: HingeReleaseNet, net_right: HingeReleaseNet,
                     t_left: torch.Tensor, t_right: torch.Tensor,
                     t0: float) -> Dict[str, torch.Tensor]:
        """Compute all loss components."""
        losses = {}
        
        # ===== Left subdomain (locked phase, t < t0) =====
        q_left, q_dot_left, q_ddot_left, lam_left = net_left.derivatives(t_left)
        
        # PDE residual
        pde_left = self.dynamics.pde_residual_locked(q_left, q_dot_left, q_ddot_left, lam_left)
        loss_pde_left = torch.mean(pde_left**2)
        
        # Constraint violation
        constraint_val = self.dynamics.constraint(q_left[:, 0], q_left[:, 1])
        loss_constraint = torch.mean(constraint_val**2)
        
        # ===== Right subdomain (free phase, t > t0) =====
        q_right, q_dot_right, q_ddot_right, _ = net_right.derivatives(t_right)
        
        pde_right = self.dynamics.pde_residual_free(q_right, q_dot_right, q_ddot_right)
        loss_pde_right = torch.mean(pde_right**2)
        
        loss_pde = (loss_pde_left + loss_pde_right) / 2.0
        losses['pde'] = loss_pde
        losses['constraint'] = loss_constraint
        
        # ===== Jump conditions at t0 =====
        t0_tensor = torch.tensor([[t0]], device=self.device, dtype=torch.float32)
        t0_tensor.requires_grad_(True)
        
        q_left_t0, q_dot_left_t0, _, _ = net_left.derivatives(t0_tensor)
        q_right_t0, q_dot_right_t0, _, _ = net_right.derivatives(t0_tensor)
        
        # Position continuity
        loss_pos_jump = torch.mean((q_left_t0 - q_right_t0)**2)
        
        # Velocity jump: from impulse-momentum
        # For hinge release, the constraint impulse causes the velocities to jump
        # Simplified: after release, both bodies maintain their angular momentum
        theta1_dot_before = q_dot_left_t0[:, 0]
        theta2_dot_before = q_dot_left_t0[:, 1]
        
        # Assuming the impulse makes both velocities equal (hinge becomes free)
        theta_dot_after = (theta1_dot_before + theta2_dot_before) / 2.0
        theta_dot_after_target = torch.stack([theta_dot_after, theta_dot_after], dim=1)
        
        loss_vel_jump = torch.mean((q_dot_right_t0 - theta_dot_after_target)**2)
        
        losses['position_jump'] = loss_pos_jump
        losses['velocity_jump'] = loss_vel_jump
        
        # ===== Initial conditions (t = 0) =====
        t_ic = torch.zeros(1, 1, device=self.device)
        q_ic, q_dot_ic, _, _ = net_left.derivatives(t_ic)
        
        # Initial angles
        theta1_ic = 0.0
        theta2_ic = -self.params.theta0
        loss_ic_theta = torch.mean((q_ic[:, 0] - theta1_ic)**2 + (q_ic[:, 1] - theta2_ic)**2)
        
        # Initial velocities (starting from rest)
        loss_ic_vel = torch.mean(q_dot_ic**2)
        
        losses['ic'] = (loss_ic_theta + loss_ic_vel) / 2.0
        
        return losses
    
    def total_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute weighted total loss."""
        total = (self.lambda_pde * losses['pde'] +
                 self.lambda_ic * losses['ic'] +
                 self.lambda_jump * (losses['position_jump'] + losses['velocity_jump']) +
                 self.lambda_constraint * losses['constraint'])
        return total


# ============================================================================
# Training Function
# ============================================================================

def train_hinge_release(params: HingeReleaseParams, epochs: int = 10000,
                        n_coll: int = 500, lr: float = 1e-3,
                        verbose: bool = True) -> Tuple[HingeReleaseNet, HingeReleaseNet, Dict]:
    """Train PINN for hinge release problem."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize networks
    net_left = HingeReleaseNet(hidden_layers=3, neurons=64, output_lambda=True).to(device)
    net_right = HingeReleaseNet(hidden_layers=3, neurons=64, output_lambda=False).to(device)
    
    # Dynamics
    dynamics = HingeReleaseDynamics(params)
    
    # Loss function
    loss_fn = HingeReleaseLoss(params, dynamics, device,
                               lambda_pde=0.01, lambda_ic=1.0,
                               lambda_jump=1.0, lambda_constraint=1.0)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        list(net_left.parameters()) + list(net_right.parameters()),
        lr=lr
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1000
    )
    
    t0 = params.t0
    history = {'loss': [], 'pde': [], 'ic': [], 'jump': [], 'constraint': []}
    
    print(f"\n{'='*60}")
    print(f"Hinge Release Training")
    print(f"Release time: t0 = {t0:.3f} s")
    print(f"Training epochs: {epochs}")
    print(f"Collocation points per domain: {n_coll}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Generate collocation points
        t_left = torch.rand(n_coll, 1, device=device) * t0
        t_right = torch.rand(n_coll, 1, device=device) * (params.t_final - t0) + t0
        
        # Compute loss
        losses = loss_fn.compute_loss(net_left, net_right, t_left, t_right, t0)
        loss_total = loss_fn.total_loss(losses)
        
        # Backward pass
        optimizer.zero_grad()
        loss_total.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(net_left.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(net_right.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step(loss_total)
        
        # Record history
        history['loss'].append(loss_total.item())
        history['pde'].append(losses['pde'].item())
        history['ic'].append(losses['ic'].item())
        history['jump'].append((losses['position_jump'] + losses['velocity_jump']).item())
        history['constraint'].append(losses['constraint'].item())
        
        # Print progress
        if verbose and (epoch + 1) % 1000 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:5d}: Loss={loss_total.item():.4e}, "
                  f"PDE={losses['pde'].item():.4e}, "
                  f"IC={losses['ic'].item():.4e}, "
                  f"Jump={losses['position_jump'].item()+losses['velocity_jump'].item():.4e}, "
                  f"LR={current_lr:.2e}")
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f} seconds")
    
    return net_left, net_right, history


# ============================================================================
# Visualization
# ============================================================================

def plot_results(net_left: HingeReleaseNet, net_right: HingeReleaseNet,
                 params: HingeReleaseParams, history: Dict,
                 save_path: str = 'hinge_release_results.png'):
    """Plot training results."""
    
    device = next(net_left.parameters()).device
    net_left.eval()
    net_right.eval()
    
    t0 = params.t0
    t_final = params.t_final
    
    # Create time points
    t_left = torch.linspace(0, t0, 500, device=device).reshape(-1, 1)
    t_right = torch.linspace(t0, t_final, 500, device=device).reshape(-1, 1)
    
    # Evaluate networks
    with torch.no_grad():
        q_left, _, _, _ = net_left.derivatives(t_left)
        q_right, _, _, _ = net_right.derivatives(t_right)
    
    t_np = np.concatenate([t_left.cpu().numpy().flatten(), t_right.cpu().numpy().flatten()])
    theta1_np = np.concatenate([q_left[:, 0].cpu().numpy(), q_right[:, 0].cpu().numpy()])
    theta2_np = np.concatenate([q_left[:, 1].cpu().numpy(), q_right[:, 1].cpu().numpy()])
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot 1: Angles
    axes[0, 0].plot(t_np, theta1_np, 'b-', label=r'$\theta_1$', linewidth=2)
    axes[0, 0].plot(t_np, theta2_np, 'r-', label=r'$\theta_2$', linewidth=2)
    axes[0, 0].axvline(x=t0, color='k', linestyle=':', linewidth=1.5, label=f'Release at t={t0:.2f}s')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Angle (rad)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title('Angles vs. Time')
    
    # Subplot 2: Relative angle
    rel_angle = theta1_np - theta2_np
    axes[0, 1].plot(t_np, rel_angle, 'g-', linewidth=2)
    axes[0, 1].axhline(y=params.theta0, color='b', linestyle='--', 
                       label=f'Locked: {params.theta0:.3f} rad')
    axes[0, 1].axvline(x=t0, color='k', linestyle=':', linewidth=1.5)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Relative Angle (rad)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title('Relative Angle (θ₁ - θ₂)')
    
    # Subplot 3: Training loss (log scale)
    axes[1, 0].semilogy(history['loss'], label='Total Loss')
    axes[1, 0].semilogy(history['pde'], label='PDE Residual')
    axes[1, 0].semilogy(history['ic'], label='Initial Conditions')
    axes[1, 0].semilogy(history['jump'], label='Jump Conditions')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title('Training Loss History')
    
    # Subplot 4: Constraint violation
    axes[1, 1].semilogy(history['constraint'], 'm-', linewidth=1.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Constraint Violation')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_title('Constraint Violation during Locked Phase')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nFigure saved to: {save_path}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Hinge Release Experiment")
    print("Non-Smooth PINN Framework Validation")
    print("=" * 60)
    
    # Parameters
    params = HingeReleaseParams()
    
    # Train
    net_left, net_right, history = train_hinge_release(
        params, epochs=10000, n_coll=500, lr=1e-3, verbose=True
    )
    
    # Save models
    torch.save(net_left.state_dict(), 'hinge_release_left.pth')
    torch.save(net_right.state_dict(), 'hinge_release_right.pth')
    print("Models saved to: hinge_release_left.pth, hinge_release_right.pth")
    
    # Plot results
    plot_results(net_left, net_right, params, history)
    
    print("\n" + "=" * 60)
    print("Experiment completed successfully!")
    print("=" * 60)