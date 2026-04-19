"""
MDANN: Multi-scale Differential-Algebraic Neural Network
PyTorch Implementation for Flexible Manipulator Dynamics
"""

import pickle
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ==================== 1. 多尺度网络模块 ====================


class SlowManifoldNetwork(nn.Module):
    """慢流形网络：学习刚体动力学和低阶模态"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        """
        参数:
            input_dim: 输入维度（慢变量维度）
            hidden_dim: 隐藏层维度
            num_layers: 隐藏层数量
        """
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, input_dim))  # 输出加速度

        self.net = nn.Sequential(*layers)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        输入: y = [θ, θ̇, w₁, w₂, ..., w_m]
        输出: y_dot = [θ̈, ẅ₁, ẅ₂, ..., ẅ_m]
        """
        return self.net(y)


class FastManifoldNetwork(nn.Module):
    """快流形网络：学习高频柔性振动"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        """
        参数:
            input_dim: 输入维度（快变量维度 + 尺度参数）
            hidden_dim: 隐藏层维度
            num_layers: 隐藏层数量
        """
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, input_dim - 1))  # 输出快加速度（不含尺度参数）

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        """
        输入: z = [w_k, ẇ_k, ...] (快变量), epsilon (尺度参数)
        输出: z_dot = [ẅ_k, ...] (快加速度)
        """
        # 将epsilon拼接到输入
        batch_size = z.shape[0]
        eps_expanded = epsilon.expand(batch_size, 1)
        z_input = torch.cat([z, eps_expanded], dim=1)

        return self.net(z_input)


# ==================== 2. 可微投影层 ====================


class DifferentiableProjectionLayer(nn.Module):
    """
    可微投影层：将数值解投影到约束流形
    实现三级约束投影（位移、速度、加速度）
    """

    def __init__(self, n_dof: int, constraint_func: callable = None):
        """
        参数:
            n_dof: 自由度数量
            constraint_func: 约束函数 g(q) = 0
        """
        super().__init__()
        self.n_dof = n_dof
        self.constraint_func = constraint_func

        # 可学习的约束乘子（可选）
        self.lambda_multiplier = nn.Parameter(torch.ones(1) * 0.01)

    def forward(self, x: torch.Tensor, constraint_type: str = "position") -> torch.Tensor:
        """
        投影操作

        参数:
            x: 状态向量 [q, q̇, q̈]
            constraint_type: 'position', 'velocity', 'acceleration'
        """
        if constraint_type == "position":
            # 位置级约束投影
            return self._project_position(x)
        elif constraint_type == "velocity":
            # 速度级约束投影
            return self._project_velocity(x)
        elif constraint_type == "acceleration":
            # 加速度级约束投影
            return self._project_acceleration(x)
        else:
            return x

    def _project_position(self, x: torch.Tensor) -> torch.Tensor:
        """投影到位置约束流形"""
        if self.constraint_func is None:
            return x

        # 简化版本：使用梯度下降投影
        # 实际应用中使用牛顿迭代或QR分解
        q = x[..., : self.n_dof]

        # 计算约束违反
        g = self.constraint_func(q)

        # 投影修正
        correction = -self.lambda_multiplier * g.unsqueeze(-1) * torch.ones_like(q)
        q_projected = q + correction

        x_projected = x.clone()
        x_projected[..., : self.n_dof] = q_projected

        return x_projected

    def _project_velocity(self, x: torch.Tensor) -> torch.Tensor:
        """投影到速度约束"""
        # 类似实现，此处简化
        return x

    def _project_acceleration(self, x: torch.Tensor) -> torch.Tensor:
        """投影到加速度约束"""
        # 类似实现，此处简化
        return x


# ==================== 3. Chebyshev谱微分层 ====================


class ChebyshevSpectralLayer(nn.Module):
    """
    Chebyshev谱微分层：实现谱精度的时间离散化
    """

    def __init__(self, n_nodes: int, dt: float):
        """
        参数:
            n_nodes: Chebyshev节点数量
            dt: 时间步长
        """
        super().__init__()
        self.n_nodes = n_nodes
        self.dt = dt

        # 预计算Chebyshev节点和微分矩阵
        self.register_buffer("tau", self._compute_chebyshev_nodes())
        self.register_buffer("D", self._compute_diff_matrix())

    def _compute_chebyshev_nodes(self) -> torch.Tensor:
        """计算Chebyshev节点"""
        k = torch.arange(1, self.n_nodes + 1)
        tau = torch.cos((2 * k - 1) / (2 * self.n_nodes) * np.pi)
        return tau

    def _compute_diff_matrix(self) -> torch.Tensor:
        """计算Chebyshev谱微分矩阵"""
        n = self.n_nodes
        D = torch.zeros(n, n)

        tau = self.tau

        for i in range(n):
            for j in range(n):
                if i != j:
                    # 计算系数
                    c_i = 2.0 if i == 0 or i == n - 1 else 1.0
                    c_j = 2.0 if j == 0 or j == n - 1 else 1.0
                    D[i, j] = (c_i / c_j) * ((-1) ** (i + j)) / (tau[i] - tau[j])
                else:
                    # 对角元
                    D[i, i] = -torch.sum(D[i, :])

        # 缩放到时间区间 [0, dt]
        D = D * (2 / self.dt)

        return D

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        计算谱微分

        参数:
            u: 在Chebyshev节点上的函数值 [batch, n_nodes, dim]
        返回:
            du/dt: 时间导数 [batch, n_nodes, dim]
        """
        # u: [batch, n_nodes, dim]
        # D: [n_nodes, n_nodes]
        du_dt = torch.einsum("ij,bjd->bid", self.D, u)
        return du_dt


# ==================== 4. MDANN完整模型 ====================


class MDANN(nn.Module):
    """
    多尺度微分-代数神经网络

    架构：
    - 慢网络：学习刚体动力学和低阶模态
    - 快网络：学习高频柔性振动
    - 可微投影层：保持约束
    - Chebyshev谱层：实现谱精度时间离散
    """

    def __init__(
        self,
        n_rigid: int = 1,  # 刚体自由度
        n_flex_slow: int = 2,  # 慢柔性模态数
        n_flex_fast: int = 0,  # 快柔性模态数
        hidden_dim_slow: int = 256,  # 慢网络隐藏层维度
        hidden_dim_fast: int = 128,  # 快网络隐藏层维度
        epsilon: float = 0.01,  # 时间尺度比
        dt: float = 0.01,  # 时间步长
        n_cheb_nodes: int = 10,  # Chebyshev节点数
        use_projection: bool = True,
    ):  # 是否使用投影
        super().__init__()

        self.n_rigid = n_rigid
        self.n_flex_slow = n_flex_slow
        self.n_flex_fast = n_flex_fast
        self.epsilon = nn.Parameter(torch.tensor(epsilon), requires_grad=True)
        self.dt = dt
        self.use_projection = use_projection

        # 输入输出维度
        self.slow_input_dim = n_rigid * 2 + n_flex_slow * 2  # [q, q̇, w_slow, ẇ_slow]
        self.slow_output_dim = n_rigid + n_flex_slow  # [q̈, ẅ_slow]

        self.fast_input_dim = n_flex_fast * 2 + 1  # [w_fast, ẇ_fast, epsilon]
        self.fast_output_dim = n_flex_fast  # [ẅ_fast]

        # 慢网络
        self.slow_net = SlowManifoldNetwork(input_dim=self.slow_input_dim, hidden_dim=hidden_dim_slow, num_layers=3)

        # 快网络（如果存在）
        if n_flex_fast > 0:
            self.fast_net = FastManifoldNetwork(input_dim=self.fast_input_dim, hidden_dim=hidden_dim_fast, num_layers=2)

        # Chebyshev谱层
        self.cheb_layer = ChebyshevSpectralLayer(n_cheb_nodes, dt)

        # 投影层
        if use_projection:
            self.proj_layer = DifferentiableProjectionLayer(n_dof=n_rigid + n_flex_slow + n_flex_fast)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, y0: torch.Tensor, z0: torch.Tensor, n_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：从初始状态预测轨迹

        参数:
            y0: 慢变量初始状态 [batch, slow_dim]
            z0: 快变量初始状态 [batch, fast_dim]
            n_steps: 预测步数
        返回:
            y_traj: 慢变量轨迹 [batch, n_steps, slow_dim]
            z_traj: 快变量轨迹 [batch, n_steps, fast_dim]
        """
        batch_size = y0.shape[0]

        # 初始化轨迹
        y_traj = [y0]
        z_traj = [z0]

        y_current = y0
        z_current = z0

        for step in range(n_steps):
            # 1. 计算慢加速度
            y_dot = self.slow_net(y_current)  # [batch, slow_output_dim]

            # 2. 计算快加速度
            if self.n_flex_fast > 0:
                z_dot = self.fast_net(z_current, self.epsilon)
            else:
                z_dot = torch.zeros_like(z_current)

            # 3. 欧拉积分更新
            y_next = y_current + self.dt * y_dot
            z_next = z_current + self.dt / self.epsilon * z_dot

            # 4. 投影（保持约束）
            if self.use_projection:
                y_next = self.proj_layer(y_next, "position")

            y_traj.append(y_next)
            z_traj.append(z_next)

            y_current = y_next
            z_current = z_next

        # 堆叠轨迹
        y_traj = torch.stack(y_traj, dim=1)  # [batch, n_steps+1, slow_dim]
        z_traj = torch.stack(z_traj, dim=1)  # [batch, n_steps+1, fast_dim]

        return y_traj, z_traj

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, z_pred: torch.Tensor, z_true: torch.Tensor, lambda_constr: float = 0.01) -> torch.Tensor:
        """
        计算损失函数

        损失组成:
        1. 轨迹预测误差 (MSE)
        2. 约束违反惩罚
        3. 能量漂移惩罚（可选）
        """
        # 1. 轨迹误差
        loss_y = nn.functional.mse_loss(y_pred, y_true)
        loss_z = nn.functional.mse_loss(z_pred, z_true)

        # 2. 约束违反（如果使用投影层）
        loss_constraint = torch.tensor(0.0, device=y_pred.device)
        if self.use_projection:
            # 简化：使用输出范数作为约束惩罚
            loss_constraint = lambda_constr * torch.norm(y_pred, dim=-1).mean()

        total_loss = loss_y + loss_z + loss_constraint

        return total_loss


# ==================== 5. 数据集类 ====================


class FlexibleManipulatorDataset(Dataset):
    """柔性机械臂数据集"""

    def __init__(self, data_path: str, slow_indices: List[int] = None, fast_indices: List[int] = None):
        """
        参数:
            data_path: 数据文件路径 (.pkl)
            slow_indices: 慢变量索引（刚体 + 低阶模态）
            fast_indices: 快变量索引（高阶模态）
        """
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

        self.trajectories = self.data["trajectories"]

        # 默认慢变量：刚体 + 前2阶模态
        if slow_indices is None:
            self.slow_indices = [0, 1, 2, 3]  # [θ, θ̇, w₁, ẇ₁]
        else:
            self.slow_indices = slow_indices

        # 默认快变量：后2阶模态
        if fast_indices is None:
            self.fast_indices = [4, 5, 6, 7]  # [w₂, ẇ₂, w₃, ẇ₃]（如果存在）
        else:
            self.fast_indices = fast_indices

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]

        # 提取状态
        # 假设traj包含: theta, theta_dot, w1, w1_dot, w2, w2_dot, ...
        theta = traj["theta"]
        theta_dot = traj["theta_dot"]
        w = traj["w"]
        w_dot = traj["w_dot"]

        # 构建慢变量
        y = np.column_stack([theta, theta_dot, w[:, :2], w_dot[:, :2]])

        # 构建快变量
        if w.shape[1] > 2:
            z = np.column_stack([w[:, 2:], w_dot[:, 2:]])
        else:
            z = np.zeros((len(theta), 1))

        # 转换为张量
        y = torch.tensor(y, dtype=torch.float32)
        z = torch.tensor(z, dtype=torch.float32)

        # 输入: 初始状态
        y0 = y[0]
        z0 = z[0]

        # 目标: 完整轨迹
        y_target = y[1:]
        z_target = z[1:]

        return y0, z0, y_target, z_target


# ==================== 6. 训练器 ====================


class MDANNTrainer:
    """MDANN训练器"""

    def __init__(self, model: MDANN, learning_rate: float = 1e-3, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=10)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0

        for y0, z0, y_target, z_target in dataloader:
            y0 = y0.to(self.device)
            z0 = z0.to(self.device)
            y_target = y_target.to(self.device)
            z_target = z_target.to(self.device)

            self.optimizer.zero_grad()

            # 前向传播
            n_steps = y_target.shape[1]
            y_pred, z_pred = self.model(y0, z0, n_steps)

            # 计算损失
            loss = self.model.compute_loss(y_pred[:, 1:], y_target, z_pred[:, 1:], z_target)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, n_epochs: int = 100, early_stop_patience: int = 20):
        """完整训练流程"""
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            self.scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{n_epochs}: " f"Train Loss = {train_loss:.6f}, " f"Val Loss = {val_loss:.6f}")

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), "best_mdann_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        return train_losses, val_losses

    def validate(self, dataloader: DataLoader) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for y0, z0, y_target, z_target in dataloader:
                y0 = y0.to(self.device)
                z0 = z0.to(self.device)
                y_target = y_target.to(self.device)
                z_target = z_target.to(self.device)

                n_steps = y_target.shape[1]
                y_pred, z_pred = self.model(y0, z0, n_steps)

                loss = self.model.compute_loss(y_pred[:, 1:], y_target, z_pred[:, 1:], z_target)
                total_loss += loss.item()

        return total_loss / len(dataloader)


# ==================== 7. 评估与可视化 ====================


class MDANNEvaluator:
    """MDANN评估器"""

    def __init__(self, model: MDANN, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device

    def predict(self, y0: np.ndarray, z0: np.ndarray, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """预测轨迹"""
        self.model.eval()

        y0_tensor = torch.tensor(y0, dtype=torch.float32).unsqueeze(0).to(self.device)
        z0_tensor = torch.tensor(z0, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            y_pred, z_pred = self.model(y0_tensor, z0_tensor, n_steps)

        return y_pred.squeeze(0).cpu().numpy(), z_pred.squeeze(0).cpu().numpy()

    def compute_metrics(self, y_pred: np.ndarray, y_true: np.ndarray, z_pred: np.ndarray, z_true: np.ndarray) -> Dict[str, float]:
        """计算评估指标"""
        # RMSE
        rmse_y = np.sqrt(np.mean((y_pred - y_true) ** 2))
        rmse_z = np.sqrt(np.mean((z_pred - z_true) ** 2))

        # 能量误差
        # 假设能量 = 0.5 * (θ̇² + Σẇ² + Σw²)
        energy_pred = 0.5 * (y_pred[:, 1] ** 2 + np.sum(z_pred[:, ::2] ** 2, axis=1) + np.sum(z_pred[:, 1::2] ** 2, axis=1))
        energy_true = 0.5 * (y_true[:, 1] ** 2 + np.sum(z_true[:, ::2] ** 2, axis=1) + np.sum(z_true[:, 1::2] ** 2, axis=1))

        energy_error = np.mean(np.abs(energy_pred - energy_true) / (np.abs(energy_true) + 1e-8))

        return {"RMSE_y": rmse_y, "RMSE_z": rmse_z, "Energy_Error": energy_error}

    def plot_prediction(self, y_pred: np.ndarray, y_true: np.ndarray, z_pred: np.ndarray, z_true: np.ndarray, save_path: str = None):
        """绘制预测对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        t = np.arange(len(y_pred)) * 0.01  # 假设dt=0.01

        # 刚体角度
        axes[0, 0].plot(t, y_true[:, 0], "b-", label="True", linewidth=2)
        axes[0, 0].plot(t, y_pred[:, 0], "r--", label="Predicted", linewidth=2)
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("θ (rad)")
        axes[0, 0].set_title("Joint Angle")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 刚体角速度
        axes[0, 1].plot(t, y_true[:, 1], "b-", label="True", linewidth=2)
        axes[0, 1].plot(t, y_pred[:, 1], "r--", label="Predicted", linewidth=2)
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].set_ylabel("θ̇ (rad/s)")
        axes[0, 1].set_title("Joint Angular Velocity")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 柔性模态
        if z_true.shape[1] >= 2:
            axes[1, 0].plot(t, z_true[:, 0], "b-", label="True", linewidth=2)
            axes[1, 0].plot(t, z_pred[:, 0], "r--", label="Predicted", linewidth=2)
            axes[1, 0].set_xlabel("Time (s)")
            axes[1, 0].set_ylabel("w₁")
            axes[1, 0].set_title("First Flexible Mode")
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        # 能量
        energy_pred = 0.5 * (y_pred[:, 1] ** 2 + np.sum(z_pred[:, ::2] ** 2, axis=1) + np.sum(z_pred[:, 1::2] ** 2, axis=1))
        energy_true = 0.5 * (y_true[:, 1] ** 2 + np.sum(z_true[:, ::2] ** 2, axis=1) + np.sum(z_true[:, 1::2] ** 2, axis=1))

        axes[1, 1].plot(t, energy_true, "b-", label="True", linewidth=2)
        axes[1, 1].plot(t, energy_pred, "r--", label="Predicted", linewidth=2)
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("Energy")
        axes[1, 1].set_title("Total Energy")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()


# ==================== 8. 主程序 ====================


def main():
    """主程序：训练和评估MDANN"""

    print("=" * 60)
    print("MDANN: Multi-scale Differential-Algebraic Neural Network")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1] 加载数据集...")
    dataset = FlexibleManipulatorDataset("flexible_manipulator_dataset.pkl")

    # 划分训练/验证/测试
    n_train = int(0.7 * len(dataset))
    n_val = int(0.15 * len(dataset))
    n_test = len(dataset) - n_train - n_val

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"    训练集: {n_train} 轨迹")
    print(f"    验证集: {n_val} 轨迹")
    print(f"    测试集: {n_test} 轨迹")

    # 2. 创建模型
    print("\n[2] 创建MDANN模型...")
    model = MDANN(
        n_rigid=1,
        n_flex_slow=2,
        n_flex_fast=0,  # 如果模态数>2，可设置快模态
        hidden_dim_slow=256,
        hidden_dim_fast=128,
        epsilon=0.01,
        dt=0.01,
        n_cheb_nodes=10,
        use_projection=True,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"    模型参数量: {n_params:,}")

    # 3. 训练
    print("\n[3] 开始训练...")
    trainer = MDANNTrainer(model, learning_rate=1e-3)
    train_losses, val_losses = trainer.train(train_loader, val_loader, n_epochs=3, early_stop_patience=20)

    # 4. 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)
    plt.savefig("mdann_training_history.png", dpi=150)
    plt.show()

    # 5. 评估
    print("\n[4] 评估模型...")
    evaluator = MDANNEvaluator(model)

    # 取一个测试样本
    test_sample = test_dataset[0]
    y0, z0, y_target, z_target = test_sample

    y_pred, z_pred = evaluator.predict(y0.numpy(), z0.numpy(), len(y_target) - 1)

    # 计算指标
    metrics = evaluator.compute_metrics(y_pred, y_target.numpy(), z_pred, z_target.numpy())
    print(f"\n    测试结果:")
    print(f"    RMSE_y: {metrics['RMSE_y']:.6f}")
    print(f"    RMSE_z: {metrics['RMSE_z']:.6f}")
    print(f"    Energy Error: {metrics['Energy_Error']:.6f}")

    # 6. 可视化
    print("\n[5] 可视化预测结果...")
    evaluator.plot_prediction(y_pred, y_target.numpy(), z_pred, z_target.numpy(), save_path="mdann_prediction.png")

    print("\n" + "=" * 60)
    print("MDANN训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
