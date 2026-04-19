"""
MDANN训练代码 - 适配柔性机械臂数据
基于之前生成的数据格式
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ========================== 1. 数据集类 ==========================


class FlexArmDataset(Dataset):
    """柔性机械臂数据集"""

    def __init__(self, data_path, seq_len=50, normalize=True):
        """
        参数:
            data_path: 数据文件路径 (.pkl)
            seq_len: 序列长度（时间步数）
            normalize: 是否归一化
        """
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

        self.trajectories = self.data["trajectories"]
        self.seq_len = seq_len
        self.normalize = normalize

        # 计算归一化参数（如果需要）
        if normalize:
            self._compute_normalization_params()

    def _compute_normalization_params(self):
        """计算所有轨迹的归一化参数"""
        all_theta = []
        all_theta_dot = []
        all_q = []
        all_q_dot = []
        all_theta_ddot = []
        all_q_ddot = []

        for traj in self.trajectories:
            all_theta.append(traj["theta"])
            all_theta_dot.append(traj["theta_dot"])
            all_q.append(traj["q"])
            all_q_dot.append(traj["q_dot"])
            all_theta_ddot.append(traj["theta_ddot"])
            all_q_ddot.append(traj["q_ddot"])

        all_theta = np.concatenate(all_theta)
        all_theta_dot = np.concatenate(all_theta_dot)
        all_q = np.concatenate(all_q)
        all_q_dot = np.concatenate(all_q_dot)
        all_theta_ddot = np.concatenate(all_theta_ddot)
        all_q_ddot = np.concatenate(all_q_ddot)

        self.theta_mean = np.mean(all_theta)
        self.theta_std = np.std(all_theta) + 1e-8
        self.theta_dot_mean = np.mean(all_theta_dot)
        self.theta_dot_std = np.std(all_theta_dot) + 1e-8
        self.q_mean = np.mean(all_q, axis=0)
        self.q_std = np.std(all_q, axis=0) + 1e-8
        self.q_dot_mean = np.mean(all_q_dot, axis=0)
        self.q_dot_std = np.std(all_q_dot, axis=0) + 1e-8
        self.theta_ddot_mean = np.mean(all_theta_ddot)
        self.theta_ddot_std = np.std(all_theta_ddot) + 1e-8
        self.q_ddot_mean = np.mean(all_q_ddot, axis=0)
        self.q_ddot_std = np.std(all_q_ddot, axis=0) + 1e-8

    def __len__(self):
        # 每条轨迹可以切分成多个序列
        total = 0
        for traj in self.trajectories:
            total += max(0, len(traj["t"]) - self.seq_len)
        return total

    def __getitem__(self, idx):
        # 找到对应的轨迹和起始位置
        for traj in self.trajectories:
            n_points = len(traj["t"])
            if idx < n_points - self.seq_len:
                start = idx
                # 提取序列
                theta = traj["theta"][start : start + self.seq_len]
                theta_dot = traj["theta_dot"][start : start + self.seq_len]
                q = traj["q"][start : start + self.seq_len]
                q_dot = traj["q_dot"][start : start + self.seq_len]
                theta_ddot = traj["theta_ddot"][start : start + self.seq_len]
                q_ddot = traj["q_ddot"][start : start + self.seq_len]
                break
            else:
                idx -= n_points - self.seq_len

        # 构建输入：慢变量 [θ, θ̇, q₁, q₂]
        X = np.column_stack([theta, theta_dot, q])

        # 构建输出：慢变量加速度 [θ̈, q̈₁, q̈₂]
        Y = np.column_stack([theta_ddot, q_ddot])

        # 归一化
        if self.normalize:
            X[:, 0] = (X[:, 0] - self.theta_mean) / self.theta_std
            X[:, 1] = (X[:, 1] - self.theta_dot_mean) / self.theta_dot_std
            X[:, 2:] = (X[:, 2:] - self.q_mean) / self.q_std
            Y[:, 0] = (Y[:, 0] - self.theta_ddot_mean) / self.theta_ddot_std
            Y[:, 1:] = (Y[:, 1:] - self.q_ddot_mean) / self.q_ddot_std

        return torch.FloatTensor(X), torch.FloatTensor(Y)


# ========================== 2. MDANN模型 ==========================


class MDANN(nn.Module):
    """多尺度微分-代数神经网络"""

    def __init__(self, input_dim, hidden_dim=128, num_layers=3):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, input_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        输入: x = [θ, θ̇, q₁, q₂]
        输出: y = [θ̈, q̈₁, q̈₂]
        """
        return self.net(x)


# ========================== 3. 训练函数 ==========================


def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3, device="cpu"):
    """训练模型"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0.0
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)

            optimizer.zero_grad()
            Y_pred = model(X)
            loss = criterion(Y_pred, Y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.to(device), Y.to(device)
                Y_pred = model(X)
                loss = criterion(Y_pred, Y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    return train_losses, val_losses


# ========================== 4. 预测函数 ==========================


def predict_trajectory(model, dataset, traj_idx=0, start_idx=0, n_steps=200, device="cpu"):
    """预测轨迹（自回归）"""
    model.eval()

    traj = dataset.trajectories[traj_idx]
    n_points = len(traj["t"])

    if start_idx + n_steps > n_points:
        n_steps = n_points - start_idx - 1

    # 获取初始状态
    theta0 = traj["theta"][start_idx]
    theta_dot0 = traj["theta_dot"][start_idx]
    q0 = traj["q"][start_idx]

    # 归一化
    if dataset.normalize:
        theta0 = (theta0 - dataset.theta_mean) / dataset.theta_std
        theta_dot0 = (theta_dot0 - dataset.theta_dot_mean) / dataset.theta_dot_std
        q0 = (q0 - dataset.q_mean) / dataset.q_std

    # 当前状态
    dt = traj["t"][1] - traj["t"][0]
    state = np.array([theta0, theta_dot0] + list(q0))

    # 预测轨迹
    pred_states = [state.copy()]
    pred_times = [0]

    for step in range(n_steps):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        acc = model(state_tensor).cpu().detach().numpy()[0]

        # 欧拉积分
        state[0] += state[1] * dt
        state[1] += acc[0] * dt
        state[2:] += acc[1:] * dt

        pred_states.append(state.copy())
        pred_times.append((step + 1) * dt)

    # 反归一化
    if dataset.normalize:
        for i, s in enumerate(pred_states):
            s[0] = s[0] * dataset.theta_std + dataset.theta_mean
            s[1] = s[1] * dataset.theta_dot_std + dataset.theta_dot_mean
            s[2:] = s[2:] * dataset.q_std + dataset.q_mean

    pred_states = np.array(pred_states)

    return pred_times, pred_states


# ========================== 5. 可视化 ==========================


def plot_prediction(traj, pred_times, pred_states, start_idx=0, save_path=None):
    """绘制预测结果对比"""
    n_points = len(traj["t"])
    t_true = traj["t"][start_idx : start_idx + len(pred_times)]

    # 截取真实轨迹对应部分
    theta_true = traj["theta"][start_idx : start_idx + len(pred_times)]
    q_true = traj["q"][start_idx : start_idx + len(pred_times)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 关节角度
    axes[0, 0].plot(t_true, np.rad2deg(theta_true), "b-", label="True", linewidth=2)
    axes[0, 0].plot(pred_times, np.rad2deg(pred_states[:, 0]), "r--", label="Predicted", linewidth=2)
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Angle (deg)")
    axes[0, 0].set_title("Joint Angle")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 关节角速度
    axes[0, 1].plot(t_true, traj["theta_dot"][start_idx : start_idx + len(pred_times)], "b-", label="True", linewidth=2)
    axes[0, 1].plot(pred_times, pred_states[:, 1], "r--", label="Predicted", linewidth=2)
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Angular Velocity (rad/s)")
    axes[0, 1].set_title("Joint Angular Velocity")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 柔性模态
    for i in range(traj["q"].shape[1]):
        axes[1, 0].plot(t_true, q_true[:, i], "b-", label=f"q{i+1} true", linewidth=2)
        axes[1, 0].plot(pred_times, pred_states[:, 2 + i], "r--", label=f"q{i+1} pred", linewidth=2)
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Modal Displacement")
    axes[1, 0].set_title("Flexible Modes")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 误差
    theta_error = np.rad2deg(theta_true - pred_states[:, 0])
    axes[1, 1].plot(pred_times, theta_error, "r-", linewidth=2)
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Angle Error (deg)")
    axes[1, 1].set_title("Prediction Error")
    axes[1, 1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"图已保存: {save_path}")

    plt.show()


# ========================== 6. 主程序 ==========================


def main():
    print("=" * 60)
    print("MDANN训练 - 柔性机械臂")
    print("=" * 60)

    # 检查数据文件
    data_dir = "data"
    if not os.path.exists(f"{data_dir}/flex_train.pkl"):
        print("错误: 未找到训练数据，请先运行数据生成代码")
        print("运行: python generate_training_data.py")
        return

    # 加载数据集
    print("\n[1/4] 加载数据集...")
    train_dataset = FlexArmDataset(f"{data_dir}/flex_train.pkl", seq_len=1, normalize=True)
    val_dataset = FlexArmDataset(f"{data_dir}/flex_val.pkl", seq_len=1, normalize=True)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    print(f"  训练集样本数: {len(train_dataset)}")
    print(f"  验证集样本数: {len(val_dataset)}")

    # 创建模型
    print("\n[2/4] 创建MDANN模型...")
    input_dim = 2 + train_dataset.data["params"]["system"]["n_modes"]
    model = MDANN(input_dim, hidden_dim=128, num_layers=3)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  输入维度: {input_dim}")
    print(f"  参数量: {n_params:,}")

    # 训练
    print("\n[3/4] 训练模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=200, lr=1e-3, device=device)

    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{data_dir}/training_history.png", dpi=150)
    plt.show()

    # 预测测试
    print("\n[4/4] 测试预测...")
    test_dataset = FlexArmDataset(f"{data_dir}/flex_test.pkl", seq_len=1, normalize=True)

    # 取一条测试轨迹进行预测
    pred_times, pred_states = predict_trajectory(model, test_dataset, traj_idx=0, start_idx=0, n_steps=300, device=device)

    # 可视化
    plot_prediction(test_dataset.trajectories[0], pred_times, pred_states, start_idx=0, save_path=f"{data_dir}/mdann_prediction.png")

    # 保存模型
    torch.save(model.state_dict(), f"{data_dir}/mdann_model.pth")
    print(f"\n模型已保存: {data_dir}/mdann_model.pth")

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
