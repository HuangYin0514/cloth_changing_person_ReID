import matplotlib.pyplot as plt
import numpy as np
import torch

# 物理参数
g = 9.81
L = 1.0
theta0 = np.pi / 4
e = 0.8
t_final = 3.0


# 简化版网络
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


def pde_residual(theta, theta_dot, theta_ddot):
    """θ̈ + (g/L) sinθ = 0"""
    gL = g / L
    return theta_ddot + gL * torch.sin(theta)


def train_pendulum(t0, epochs=5000, n_coll=500):
    """训练带冲击的摆"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建网络
    net_left = SimplePendulumNet(hidden=3, neurons=64).to(device)
    net_right = SimplePendulumNet(hidden=3, neurons=64).to(device)

    # 优化器
    optimizer = torch.optim.Adam(list(net_left.parameters()) + list(net_right.parameters()), lr=1e-3)

    # 损失权重
    lambda_pde = 0.01
    lambda_ic = 1.0
    lambda_jump = 1.0

    print(f"冲击时间: {t0:.4f} s")
    print(f"训练轮数: {epochs}")
    print("-" * 60)

    for epoch in range(epochs):
        # 生成配点 - 明确指定 float32 类型
        t_left = torch.rand(n_coll, 1, device=device, dtype=torch.float32) * t0
        t_right = torch.rand(n_coll, 1, device=device, dtype=torch.float32) * (t_final - t0) + t0

        # ===== 左子域 =====
        theta_left, theta_dot_left, theta_ddot_left = net_left.derivatives(t_left)
        pde_left = pde_residual(theta_left, theta_dot_left, theta_ddot_left)
        loss_pde_left = torch.mean(pde_left**2)

        # ===== 右子域 =====
        theta_right, theta_dot_right, theta_ddot_right = net_right.derivatives(t_right)
        pde_right = pde_residual(theta_right, theta_dot_right, theta_ddot_right)
        loss_pde_right = torch.mean(pde_right**2)
        loss_pde = (loss_pde_left + loss_pde_right) / 2

        # ===== 初始条件 =====
        t0_tensor = torch.zeros(1, 1, device=device, dtype=torch.float32)
        theta0_pred, theta_dot0_pred, _ = net_left.derivatives(t0_tensor)
        # 将 numpy 的 theta0 转换为 float32 tensor
        theta0_tensor = torch.tensor(theta0, device=device, dtype=torch.float32)
        loss_ic_theta = torch.mean((theta0_pred - theta0_tensor) ** 2)
        loss_ic_vel = torch.mean((theta_dot0_pred) ** 2)
        loss_ic = loss_ic_theta + loss_ic_vel

        # ===== 冲击条件 =====
        # 关键修复：明确指定 dtype=torch.float32
        t0_tensor = torch.tensor([[t0]], device=device, dtype=torch.float32)
        t0_tensor.requires_grad_(True)
        theta_left_t0, theta_dot_left_t0, _ = net_left.derivatives(t0_tensor)
        theta_right_t0, theta_dot_right_t0, _ = net_right.derivatives(t0_tensor)

        loss_pos_jump = torch.mean((theta_left_t0 - theta_right_t0) ** 2)
        loss_vel_jump = torch.mean((theta_dot_right_t0 + e * theta_dot_left_t0) ** 2)
        loss_jump = loss_pos_jump + loss_vel_jump

        # ===== 总损失 =====
        loss_total = lambda_pde * loss_pde + lambda_ic * loss_ic + lambda_jump * loss_jump

        # 反向传播
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        # 打印进度
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}: Loss={loss_total.item():.4e}, " f"PDE={loss_pde.item():.4e}, " f"IC={loss_ic.item():.4e}, " f"Jump={loss_jump.item():.4e}")

    return net_left, net_right


if __name__ == "__main__":
    # 计算冲击时间
    # 简单摆从 θ0 到 0 的时间（能量守恒）
    # t_impact = ∫_{θ0}^{0} dθ / sqrt(2g/L (cosθ - cosθ0))
    # 用小角度近似快速计算
    omega0 = np.sqrt(g / L)
    t0_approx = np.arccos(0) / omega0  # 近似值

    print("=" * 60)
    print("训练带冲击的简单摆")
    print("=" * 60)

    net_left, net_right = train_pendulum(t0_approx, epochs=5000, n_coll=500)

    print("\n训练完成！")
    print("\n下一步：保存模型并绘图")

    # 保存模型
    torch.save(net_left.state_dict(), "pendulum_left.pth")
    torch.save(net_right.state_dict(), "pendulum_right.pth")
    print("模型已保存到 pendulum_left.pth 和 pendulum_right.pth")
