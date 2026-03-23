"""
physics.py - Correct physical models for hinge release problem

This module contains the correct physics for:
- Mass matrix
- Gravity torques
- Constraint handling
- Velocity jump calculation
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class HingeParams:
    """Physical parameters for hinge release problem."""

    # Geometry
    l1: float = 0.5  # Distance from hinge to COM of body 1 (m)
    l2: float = 0.5  # Distance from hinge to COM of body 2 (m)

    # Mass and inertia
    m1: float = 2.0  # Mass of body 1 (kg)
    m2: float = 2.0  # Mass of body 2 (kg)
    I1: float = 0.5  # Moment of inertia of body 1 about COM (kg·m²)
    I2: float = 0.5  # Moment of inertia of body 2 about COM (kg·m²)

    # Initial conditions
    theta0: float = np.pi / 6  # Initial relative angle (rad)

    # Gravity
    g: float = 9.81  # Acceleration due to gravity (m/s²)

    # Time
    t0: float = 0.5  # Release time (s)
    t_final: float = 1.5  # Final time (s)

    # External torques (optional)
    tau_ext1: float = 0.0
    tau_ext2: float = 0.0


    def inertia_about_hinge(self):
        """
        Moment of inertia of each body about the hinge point.
        Using parallel axis theorem: I_hinge = I_com + m * l²
        """
        I1_hinge = self.I1 + self.m1 * self.l1**2
        I2_hinge = self.I2 + self.m2 * self.l2**2
        return I1_hinge, I2_hinge


class HingeDynamics:
    """
    Correct dynamics for hinge release problem.
    """

    def __init__(self, params: HingeParams):
        self.params = params
        self.I1_hinge, self.I2_hinge = params.inertia_about_hinge()

    def mass_matrix(self, theta1: torch.Tensor, theta2: torch.Tensor) -> torch.Tensor:
        """
        Compute the mass matrix M.

        For independent rotation about hinge, M is diagonal:
        M = diag([I1_hinge, I2_hinge])

        Args:
        theta1, theta2: Angles (not used for mass matrix, but kept for consistency)

        Returns:
        M: 2x2 mass matrix
        """
        M = torch.zeros(2, 2, dtype=theta1.dtype, device=theta1.device)
        M[0, 0] = self.I1_hinge
        M[1, 1] = self.I2_hinge
        # M[0,1] = M[1,0] = 0 because rotations are independent
        return M


    def coriolis(
        self,
        theta1: torch.Tensor,
        theta2: torch.Tensor,
        theta1_dot: torch.Tensor,
        theta2_dot: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Coriolis and centrifugal forces.

        For independent rotations about fixed hinges, Coriolis terms are zero.
        (Each body rotates about a fixed point, no velocity-dependent forces)

        Returns:
        C: 2x1 vector of Coriolis forces
        """
        return torch.zeros(2, 1, dtype=theta1.dtype, device=theta1.device)


    def gravity_torques(self, theta1: torch.Tensor, theta2: torch.Tensor) -> torch.Tensor:
        """
        Compute gravitational torques about the hinge point.

        For a body rotating about a hinge:
        Torque = m * g * l * cos(θ)
        where θ=0 is horizontal position.

        Args:
        theta1, theta2: Angles (rad)

        Returns:
        G: 2x1 vector of gravitational torques
        """
        G1 = -self.params.m1 * self.params.g * self.params.l1 * torch.cos(theta1)
        G2 = -self.params.m2 * self.params.g * self.params.l2 * torch.cos(theta2)

        G = torch.stack([G1, G2], dim=0)
        return G


    def constraint(self, theta1: torch.Tensor, theta2: torch.Tensor) -> torch.Tensor:
        """
        Holonomic constraint for locked phase: θ1 - θ2 = θ0.

        Returns:
        Φ: Constraint violation
        """
        return theta1 - theta2 - self.params.theta0


    def constraint_jacobian(self) -> torch.Tensor:
        """
        Jacobian of the constraint: J = [∂Φ/∂θ1, ∂Φ/∂θ2] = [1, -1]

        Returns:
        J: 1x2 Jacobian matrix
        """
        J = torch.tensor(
            [[1.0, -1.0]],
            device=(
                self.params.get_device() if hasattr(self.params, "get_device") else "cpu"
            ),
        )
        return J


    def acceleration_locked(
        self,
        theta1: torch.Tensor,
        theta2: torch.Tensor,
        theta1_dot: torch.Tensor,
        theta2_dot: torch.Tensor,
        lambda_val: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute accelerations for locked phase.

        Equations:
        M * q_ddot + G = F_ext + J^T * λ
        J * q_ddot = -J_dot * q_dot  (derived from constraint)

        For this problem, J_dot = 0 because J is constant.

        Args:
        theta1, theta2: Angles
        theta1_dot, theta2_dot: Angular velocities
        lambda_val: Lagrange multiplier (constraint force)

        Returns:
        q_ddot: 2x1 acceleration vector
        """
        # Mass matrix
        M = self.mass_matrix(theta1, theta2)

        # Gravity torques
        G = self.gravity_torques(theta1, theta2)

        # External torques
        F_ext = torch.tensor(
            [[self.params.tau_ext1], [self.params.tau_ext2]],
            dtype=theta1.dtype,
            device=theta1.device,
        )

        # Constraint Jacobian
        J = self.constraint_jacobian().to(theta1.device)

        # Right-hand side: F_ext - G + J^T * λ
        rhs = F_ext - G + J.T @ lambda_val

        # Solve for accelerations
        # For locked phase, accelerations are determined by the constraint
        # We can solve: M * q_ddot = rhs, with constraint J * q_ddot = 0

        # Since J = [1, -1], constraint means θ1_ddot = θ2_ddot
        # Let a = θ1_ddot = θ2_ddot
        # Then (M11 + M22) * a = rhs1 + rhs2
        M11 = M[0, 0].item()
        M22 = M[1, 1].item()
        rhs1 = rhs[0, 0].item()
        rhs2 = rhs[1, 0].item()

        a = (rhs1 + rhs2) / (M11 + M22)

        q_ddot = torch.tensor([[a], [a]], dtype=theta1.dtype, device=theta1.device)

        return q_ddot


    def acceleration_free(
        self,
        theta1: torch.Tensor,
        theta2: torch.Tensor,
        theta1_dot: torch.Tensor,
        theta2_dot: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute accelerations for free phase (released).

        Equation: M * q_ddot = F_ext - G

        Args:
        theta1, theta2: Angles
        theta1_dot, theta2_dot: Angular velocities

        Returns:
        q_ddot: 2x1 acceleration vector
        """
        # Mass matrix
        M = self.mass_matrix(theta1, theta2)

        # Gravity torques
        G = self.gravity_torques(theta1, theta2)

        # External torques
        F_ext = torch.tensor(
            [[self.params.tau_ext1], [self.params.tau_ext2]],
            dtype=theta1.dtype,
            device=theta1.device,
        )

        # Solve M * q_ddot = F_ext - G
        rhs = F_ext - G

        # Since M is diagonal, we can solve directly
        q_ddot1 = rhs[0, 0] / M[0, 0]
        q_ddot2 = rhs[1, 0] / M[1, 1]

        q_ddot = torch.stack([q_ddot1, q_ddot2], dim=0)

        return q_ddot


    def velocity_jump(
        self,
        theta1: torch.Tensor,
        theta2: torch.Tensor,
        theta1_dot_before: torch.Tensor,
        theta2_dot_before: torch.Tensor,
        lambda_impulse: torch.Tensor,
    ):
        """
        Compute velocity jump at release due to impulse.

        At the instant of release, the constraint force λ produces an impulse I = ∫ λ dt.
        The impulse-momentum equation: M * (q_dot_after - q_dot_before) = J^T * I

        Args:
        theta1, theta2: Angles at release
        theta1_dot_before, theta2_dot_before: Velocities just before release
        lambda_impulse: Impulse magnitude (scalar)

        Returns:
        theta1_dot_after, theta2_dot_after: Velocities just after release
        """
        # Mass matrix
        M = self.mass_matrix(theta1, theta2)

        # Constraint Jacobian
        J = self.constraint_jacobian().to(theta1.device)

        # Before velocities as vector
        q_dot_before = torch.stack([theta1_dot_before, theta2_dot_before], dim=0)

        # Impulse vector: J^T * λ_impulse
        impulse = J.T * lambda_impulse

        # Solve for velocity after: M * Δq_dot = impulse
        # Δq_dot = M^{-1} * impulse
        delta_q_dot = torch.linalg.solve(M, impulse)

        # After velocities
        q_dot_after = q_dot_before + delta_q_dot

        return q_dot_after[0, 0], q_dot_after[1, 0]


    def pde_residual_locked(
        self, q: torch.Tensor, q_dot: torch.Tensor, q_ddot: torch.Tensor, lam: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute PDE residual for locked phase.

        Residual = M * q_ddot + G - F_ext - J^T * λ

        Args:
        q: (batch, 2) generalized coordinates
        q_dot: (batch, 2) velocities
        q_ddot: (batch, 2) accelerations
        lam: (batch, 1) Lagrange multiplier

        Returns:
        Residual (batch, 2)
        """
        batch_size = q.shape[0]
        device = q.device

        residuals = []

        for i in range(batch_size):
            theta1 = q[i, 0].unsqueeze(0).unsqueeze(0)
            theta2 = q[i, 1].unsqueeze(0).unsqueeze(0)
            theta1_dot = q_dot[i, 0].unsqueeze(0).unsqueeze(0)
            theta2_dot = q_dot[i, 1].unsqueeze(0).unsqueeze(0)
            q_ddot_i = q_ddot[i].unsqueeze(1)
            lam_i = lam[i].unsqueeze(0).unsqueeze(0)

            M = self.mass_matrix(theta1, theta2)
            G = self.gravity_torques(theta1, theta2)
            J = self.constraint_jacobian().to(device)

            F_ext = torch.tensor(
                [[self.params.tau_ext1], [self.params.tau_ext2]],
                dtype=theta1.dtype,
                device=device,
            )

            residual = M @ q_ddot_i + G - F_ext - J.T @ lam_i
            residuals.append(residual.squeeze())

        return torch.stack(residuals, dim=0)


    def pde_residual_free(
        self, q: torch.Tensor, q_dot: torch.Tensor, q_ddot: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute PDE residual for free phase.

        Residual = M * q_ddot + G - F_ext

        Args:
        q: (batch, 2) generalized coordinates
        q_dot: (batch, 2) velocities
        q_ddot: (batch, 2) accelerations

        Returns:
        Residual (batch, 2)
        """
        batch_size = q.shape[0]
        device = q.device

        residuals = []

        for i in range(batch_size):
            theta1 = q[i, 0].unsqueeze(0).unsqueeze(0)
            theta2 = q[i, 1].unsqueeze(0).unsqueeze(0)
            q_ddot_i = q_ddot[i].unsqueeze(1)

            M = self.mass_matrix(theta1, theta2)
            G = self.gravity_torques(theta1, theta2)

            F_ext = torch.tensor(
                [[self.params.tau_ext1], [self.params.tau_ext2]],
                dtype=theta1.dtype,
                device=device,
            )

            residual = M @ q_ddot_i + G - F_ext
            residuals.append(residual.squeeze())

        return torch.stack(residuals, dim=0)
