"""
test_pendulum_physics.py - 先测试无冲击的简单摆
只训练左子域，验证PDE损失是否能下降
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# 物理参数
g = 9.81
L = 1.0
theta0 = np.pi / 4  # 初始角度
t_final = 0.5  # 冲击前的时间


# 简单网络
class PendulumNet(nn.Module):
    def __init__(self, hidden=3, neurons=64):
        super().__init__()
        layers = [nn.Linear(1, neurons), nn.Tanh()]
        for _ in range(hidden - 1):
            layers.append(nn.Linear(neurons, neurons))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(neurons, 1))
        self.net = nn.Sequential(*layers)

        # 使用xavier初始化
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        return self.net(t)

    def derivatives(self, t):
        t.requires_grad_(True)
        q = self.net(t)
        q_dot = torch.autograd.grad(q, t, torch.ones_like(q), create_graph=True, retain_graph=True)[0]
        q_ddot = torch.autograd.grad(q_dot, t, torch.ones_like(q_dot), create_graph=True)[0]
        return q, q_dot, q_ddot


def pde_residual(theta, theta_dot, theta_ddot):
    """PDE残差: θ̈ + (g/L) sinθ = 0"""
    gL = g / L
    return theta_ddot + gL * torch.sin(theta)


def train_simple_pendulum(epochs=5000, verbose=True):
    """训练无冲击的简单摆"""

    # 创建网络
    net = PendulumNet(hidden=3, neurons=64)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=500)

    # 损失权重
    lambda_pde = 1.0
    lambda_ic = 10.0

    history = {"loss": [], "pde": [], "ic": []}

    for epoch in range(epochs):
        # 生成配点
        n_coll = 500
        t_coll = torch.rand(n_coll, 1) * t_final

        # 初始条件点
        t_ic = torch.zeros(1, 1)

        # 前向计算
        theta_coll, theta_dot_coll, theta_ddot_coll = net.derivatives(t_coll)
        theta_ic, theta_dot_ic, _ = net.derivatives(t_ic)

        # PDE损失
        pde = pde_residual(theta_coll, theta_dot_coll, theta_ddot_coll)
        loss_pde = torch.mean(pde**2)

        # 初始条件损失
        loss_ic_theta = torch.mean((theta_ic - theta0) ** 2)
        loss_ic_theta_dot = torch.mean((theta_dot_ic - 0.0) ** 2)
        loss_ic = loss_ic_theta + loss_ic_theta_dot

        # 总损失
        loss_total = lambda_pde * loss_pde + lambda_ic * loss_ic

        # 反向传播
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step(loss_total)

        # 记录
        history["loss"].append(loss_total.item())
        history["pde"].append(loss_pde.item())
        history["ic"].append(loss_ic.item())

        if verbose and (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}: Loss={loss_total.item():.4e}, " f"PDE={loss_pde.item():.4e}, IC={loss_ic.item():.4e}, " f"LR={optimizer.param_groups[0]['lr']:.2e}")

    return net, history


def plot_results(net, t_final):
    """绘制结果"""
    net.eval()
    t_test = torch.linspace(0, t_final, 500).reshape(-1, 1)
    theta, theta_dot, _ = net.derivatives(t_test)

    # 参考解（小角度近似）
    omega0 = np.sqrt(g / L)
    theta_ref = theta0 * np.cos(omega0 * t_test.numpy().flatten())

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    axes[0].plot(t_test.numpy(), theta.detach().numpy(), "b-", label="PINN")
    axes[0].plot(t_test.numpy(), theta_ref, "r--", label="Reference (small angle)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Angle (rad)")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_title("Pendulum Motion")

    axes[1].plot(t_test.numpy(), theta_dot.detach().numpy(), "b-", label="PINN")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Angular Velocity (rad/s)")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("训练无冲击的简单摆...")
    net, history = train_simple_pendulum(epochs=3000)

    # 绘制损失曲线
    plt.figure(figsize=(10, 4))
    plt.semilogy(history["loss"], label="Total Loss")
    plt.semilogy(history["pde"], label="PDE Loss")
    plt.semilogy(history["ic"], label="IC Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.title("Training Loss")
    plt.show()

    # 绘制结果
    plot_results(net, t_final=0.5)

    print(f"\n最终损失: {history['loss'][-1]:.4e}")
    print(f"最终PDE损失: {history['pde'][-1]:.4e}")
