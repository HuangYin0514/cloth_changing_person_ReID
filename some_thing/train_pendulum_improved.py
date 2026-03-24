"""
冲击摆PINN：全局时间 t + 单个网络 + 无分段
严格满足：全局t，单网络，无轨迹学习
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import torch

# ============================================================
# 物理参数
# ============================================================
g = 9.81
L = 1.0
theta0 = np.pi / 4
e = 0.8
t_final = 2.0
omega0 = np.sqrt(g / L)
t0 = np.pi / (2 * omega0)

print("=" * 60)
print("冲击摆训练：全局时间 t + 单个网络")
print("=" * 60)
print(f"冲击时间: t0 = {t0:.4f} s")


# ============================================================
# 单个网络 + 全局 t（核心）
# ============================================================
class GlobalTimeNet(torch.nn.Module):
    def __init__(self, neurons=160):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, neurons),
            torch.nn.Tanh(),
            torch.nn.Linear(neurons, neurons),
            torch.nn.SiLU(),
            torch.nn.Linear(neurons, neurons),
            torch.nn.SiLU(),
            torch.nn.Linear(neurons, neurons),
            torch.nn.SiLU(),
            torch.nn.Linear(neurons, 1),
        )

    def forward(self, t):
        # 全程只使用 全局时间 t，无任何分段/坐标变换
        return self.net(t)

    def get_derivs(self, t):
        # 自动微分，全局 t
        t.requires_grad_(True)
        q = self.forward(t.float())
        qt = torch.autograd.grad(q.sum(), t, create_graph=True)[0]
        qtt = torch.autograd.grad(qt.sum(), t, create_graph=True)[0]
        return q, qt, qtt


# ============================================================
# 精确解（仅用于测试）
# ============================================================
def exact_solution(t):
    res = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < t0:
            res[i] = theta0 * np.cos(omega0 * ti)
        else:
            res[i] = -e * theta0 * np.sin(omega0 * (ti - t0))
    return res


# ============================================================
# 训练：全局 t，单个网络
# ============================================================
def train(epochs=22000, lr=1.2e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GlobalTimeNet(neurons=160).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1500, factor=0.5)

    history = {"loss": [], "pde": [], "ic": [], "jump": []}
    start_time = time.time()

    for epoch in range(epochs):
        # ===================== 全局 t 采样 =====================
        t = torch.rand(1200, 1, device=device) * t_final
        t_impact = torch.rand(450, 1, device=device) * 0.04 + (t0 - 0.02)  # 冲击点加密
        t_coll = torch.cat([t, t_impact]).clamp(0, t_final)

        # 前向：全局 t 输入
        theta, th_d, th_dd = model.get_derivs(t_coll)

        # ===================== 物理损失 =====================
        # PDE损失
        pde = th_dd + (g / L) * torch.sin(theta)
        loss_pde = torch.mean(pde**2)

        # 初始条件损失 t=0
        t_ic = torch.zeros(1, 1, device=device)
        th_ic, thd_ic, _ = model.get_derivs(t_ic)
        loss_ic = (th_ic - theta0) ** 2 + (thd_ic) ** 2
        loss_ic = loss_ic.mean() * 80

        # 冲击损失（全局 t 直接计算）
        eps = 8e-5
        tb = torch.tensor([[t0 - eps]], device=device)
        ta = torch.tensor([[t0 + eps]], device=device)
        th_b, thd_b, _ = model.get_derivs(tb)
        th_a, thd_a, _ = model.get_derivs(ta)

        loss_jump_pos = (th_a**2).mean() * 120
        loss_jump_vel = ((thd_a + e * thd_b) ** 2).mean() * 120
        loss_jump = loss_jump_pos + loss_jump_vel

        # 总损失
        loss = loss_pde + loss_ic + loss_jump

        # ===================== 优化 =====================
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        scheduler.step(loss)

        # 日志
        history["loss"].append(loss.item())
        history["pde"].append(loss_pde.item())
        history["ic"].append(loss_ic.item())
        history["jump"].append(loss_jump.item())

        if (epoch + 1) % 2000 == 0:
            print(f"Epoch {epoch+1:5d} | Total={loss.item():.2e} | PDE={loss_pde.item():.2e} | Jump={loss_jump.item():.2e}")

    print(f"\n训练完成，用时：{time.time()-start_time:.1f}s")
    torch.save(model.state_dict(), "global_t_pendulum.pth")
    return model, history, device


# ============================================================
# 绘图测试
# ============================================================
def plot(model, history, device):
    t = torch.linspace(0, t_final, 1000, device=device).reshape(-1, 1)
    pred = model(t).deatch().cpu().numpy().flatten()

    t_np = t.deatch().cpu().numpy().flatten()
    exact = exact_solution(t_np)
    error = np.abs(pred - exact)
    l2_err = np.sqrt(np.mean(error**2))

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(t_np, exact, "k-", lw=2.5, label="Exact")
    plt.plot(t_np, pred, "r--", lw=2, label="PINN")
    plt.axvline(t0, c="b", ls=":", lw=2, label=f"t0={t0:.3f}s")
    plt.title("Global t - Pendulum Impact")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.semilogy(t_np, error, "g-", lw=1.5)
    plt.axvline(t0, c="b", ls=":")
    plt.title(f"Error | L2 = {l2_err:.2e}")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.semilogy(history["loss"], label="Total")
    plt.semilogy(history["pde"], label="PDE")
    plt.semilogy(history["ic"], label="IC")
    plt.semilogy(history["jump"], label="Jump")
    plt.title("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    t0_pt = torch.tensor([[t0]], device=device)
    with torch.no_grad():
        val = model(t0_pt).item()
    plt.axhline(val, c="r", lw=2, label=f"θ(t0) = {val:.4f}")
    plt.axhline(0, c="k", ls="--")
    plt.title("Impact Position")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("global_t_result.png", dpi=300)
    plt.show()

    print("\n✅ 测试结果：")
    print(f"θ(t0) = {val:.6f} rad")
    print(f"L2 误差 = {l2_err:.4f} rad")


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    model, hist, device = train(epochs=22000)
    plot(model, hist, device)
