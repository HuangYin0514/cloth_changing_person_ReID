# ═══════════════════════════════════════════════════════════════════════
#  柔性机械臂动力学模型训练代码（Google Colab 专用）
#  MDANN  vs  HNN  vs  LNN  三模型对比训练
#
#  使用前提：已运行 flexible_arm_datagen.py，data/ 目录下有 pkl 文件
#
#  三种模型简介：
#    MDANN  直接回归加速度，输入(q,dq)→输出 ddq，黑盒神经网络
#    LNN    拉格朗日神经网络，学习拉格朗日量 L(q,dq)，从 L 推导方程
#    HNN    哈密顿神经网络，学习哈密顿量 H(q,p)，从 H 推导方程
#
#  状态约定（与仿真代码一致）：
#    theta=-pi/2 为竖直向下平衡点
#    输入：[theta, dtheta, q1, q2, dq1, dq2]  (6维)
#    输出：[ddtheta, ddq1, ddq2]              (3维)
# ═══════════════════════════════════════════════════════════════════════

# ── 安装依赖（首次运行取消注释）────────────────────────────────────────
# !pip install torch numpy scipy matplotlib tqdm -q

import numpy as np
import pickle
import os
import time
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

import matplotlib
matplotlib.use('Agg')    # Colab 中改为: %matplotlib inline
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Noto Sans CJK JP', 'WenQuanYi Zen Hei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.facecolor': '#0d1117', 'axes.facecolor':  '#161b22',
    'axes.edgecolor':   '#30363d', 'axes.labelcolor': '#c9d1d9',
    'xtick.color':      '#8b949e', 'ytick.color':     '#8b949e',
    'text.color':       '#c9d1d9', 'grid.color':      '#21262d',
    'grid.linewidth':    0.8,      'legend.facecolor': '#161b22',
    'legend.edgecolor': '#30363d', 'font.size': 11,
})
C = ['#58a6ff', '#ff7b72', '#56d364', '#f0883e', '#bc8cff', '#39d353', '#ffa657']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('models',  exist_ok=True)
os.makedirs('results', exist_ok=True)
print(f"设备: {DEVICE}")
print(f"PyTorch: {torch.__version__}")


# ══════════════════════════════════════════════════════════════════════
# ① 数据加载与归一化
# ══════════════════════════════════════════════════════════════════════

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def build_arrays(dataset):
    """
    将数据集中所有轨迹拼接为大数组。
    返回：
      X  (N, 6)  输入状态  [theta, dtheta, q1, q2, dq1, dq2]
      Y  (N, 3)  输出加速度 [ddtheta, ddq1, ddq2]
      X_hnn (N, 6)  HNN输入  [theta, q1, q2, p_theta, p_q1, p_q2]
      Y_hnn (N, 6)  HNN输出  [dtheta, dq1, dq2, dp_theta, dp_q1, dp_q2]
    """
    trajs = dataset['trajectories']
    X_list, Y_list, Xh_list, Yh_list = [], [], [], []

    for tr in trajs:
        T = len(tr['t'])
        # ── MDANN / LNN 格式 ──
        x = np.stack([
            tr['theta'], tr['dtheta'],
            tr['q'][0],  tr['q'][1],
            tr['dq'][0], tr['dq'][1],
        ], axis=1)   # (T, 6)
        y = np.stack([
            tr['ddtheta'], tr['ddq'][0], tr['ddq'][1],
        ], axis=1)   # (T, 3)
        X_list.append(x);  Y_list.append(y)

        # ── HNN 格式：广义坐标 + 广义动量 ──
        xh = np.stack([
            tr['theta'],   tr['q'][0],   tr['q'][1],
            tr['p_theta'], tr['p_q'][0], tr['p_q'][1],
        ], axis=1)   # (T, 6)
        # 目标：哈密顿方程的时间导数
        # dq/dt 直接来自数据；dp/dt ≈ -∂H/∂q 用 ddtheta/ddq 近似
        yh = np.stack([
            tr['dtheta'],  tr['dq'][0],   tr['dq'][1],
            tr['ddtheta'], tr['ddq'][0],  tr['ddq'][1],
        ], axis=1)   # (T, 6)
        Xh_list.append(xh); Yh_list.append(yh)

    X  = np.concatenate(X_list,  axis=0).astype(np.float32)
    Y  = np.concatenate(Y_list,  axis=0).astype(np.float32)
    Xh = np.concatenate(Xh_list, axis=0).astype(np.float32)
    Yh = np.concatenate(Yh_list, axis=0).astype(np.float32)
    return X, Y, Xh, Yh


class Normalizer:
    """零均值单位方差归一化器，支持 numpy 和 torch tensor。"""
    def __init__(self, data: np.ndarray):
        self.mean = data.mean(axis=0).astype(np.float32)
        self.std  = (data.std(axis=0) + 1e-8).astype(np.float32)

    def transform(self, x):
        if isinstance(x, torch.Tensor):
            m = torch.tensor(self.mean, device=x.device)
            s = torch.tensor(self.std,  device=x.device)
            return (x - m) / s
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        if isinstance(x, torch.Tensor):
            m = torch.tensor(self.mean, device=x.device)
            s = torch.tensor(self.std,  device=x.device)
            return x * s + m
        return x * self.std + self.mean


# ── 加载数据 ──
print("\n加载数据集...")
train_ds = load_pkl('data/flexible_arm_train.pkl')
val_ds   = load_pkl('data/flexible_arm_val.pkl')
test_ds  = load_pkl('data/flexible_arm_test.pkl')

X_tr, Y_tr, Xh_tr, Yh_tr = build_arrays(train_ds)
X_va, Y_va, Xh_va, Yh_va = build_arrays(val_ds)
X_te, Y_te, Xh_te, Yh_te = build_arrays(test_ds)

# ── 归一化 ──
norm_X  = Normalizer(X_tr)
norm_Y  = Normalizer(Y_tr)
norm_Xh = Normalizer(Xh_tr)
norm_Yh = Normalizer(Yh_tr)

def norm_ds(X, Y, nx, ny):
    return (
        torch.tensor(nx.transform(X)),
        torch.tensor(ny.transform(Y)),
    )

Xtn, Ytn   = norm_ds(X_tr, Y_tr, norm_X, norm_Y)
Xvn, Yvn   = norm_ds(X_va, Y_va, norm_X, norm_Y)
Xten, Yten = norm_ds(X_te, Y_te, norm_X, norm_Y)

Xhtn, Yhtn   = norm_ds(Xh_tr, Yh_tr, norm_Xh, norm_Yh)
Xhvn, Yhvn   = norm_ds(Xh_va, Yh_va, norm_Xh, norm_Yh)
Xhten, Yhten = norm_ds(Xh_te, Yh_te, norm_Xh, norm_Yh)

print(f"训练集: X={X_tr.shape}  Y={Y_tr.shape}")
print(f"验证集: X={X_va.shape}  验证={X_te.shape}")


# ══════════════════════════════════════════════════════════════════════
# ② 模型定义
# ══════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────
#  MDANN：多层感知机直接回归加速度
#  输入：[theta, dtheta, q1, q2, dq1, dq2]  → (N, 6)
#  输出：[ddtheta, ddq1, ddq2]              → (N, 3)
# ─────────────────────────────────────────────────────
class MDANN(nn.Module):
    def __init__(self, in_dim=6, out_dim=3,
                 hidden_dims=(128, 128, 128), dropout=0.0):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────
#  LNN：拉格朗日神经网络
#
#  思路：神经网络学习标量拉格朗日量 L(q, dq)
#  运动方程由欧拉-拉格朗日方程推导：
#    d/dt(∂L/∂dq) - ∂L/∂q = 0
#  展开得：
#    M(q,dq) * ddq = F(q,dq)
#  其中 M = ∂²L/∂dq²（质量矩阵），F 由梯度计算
#
#  实现：用 torch.autograd 计算 L 对 q, dq 的梯度
#
#  输入：[theta, dtheta, q1, q2, dq1, dq2]  → (N, 6)
#  输出：[ddtheta, ddq1, ddq2]              → (N, 3)
# ─────────────────────────────────────────────────────
class LNN(nn.Module):
    def __init__(self, dof=3, hidden_dims=(128, 128, 128)):
        """
        dof：广义坐标维数（theta + q1 + q2 = 3）
        输入维度 = 2 * dof（坐标 + 速度）
        """
        super().__init__()
        self.dof = dof
        in_dim = 2 * dof
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, 1))   # 输出标量 L
        self.net = nn.Sequential(*layers)

    def lagrangian(self, x):
        """计算拉格朗日量 L(q, dq)，返回标量 (N, 1)"""
        return self.net(x)

    def forward(self, x):
        """
        输入 x: (N, 2*dof)，前 dof 列为 q，后 dof 列为 dq
        输出：加速度 ddq (N, dof)，由欧拉-拉格朗日方程推导
        """
        n   = x.shape[0]
        dof = self.dof
        x   = x.requires_grad_(True)

        # 拉格朗日量
        L   = self.lagrangian(x).sum()   # 标量

        # ∂L/∂x（对所有输入的梯度）
        dL  = torch.autograd.grad(L, x, create_graph=True)[0]  # (N, 2*dof)
        dLdq  = dL[:, :dof]    # ∂L/∂q
        dLddq = dL[:, dof:]    # ∂L/∂dq = 广义动量

        # 质量矩阵 M = ∂²L/∂dq²  形状 (N, dof, dof)
        M = torch.zeros(n, dof, dof, device=x.device)
        for i in range(dof):
            grad_i = torch.autograd.grad(
                dLddq[:, i].sum(), x, create_graph=True)[0][:, dof:]
            M[:, i, :] = grad_i

        # ∂²L/∂dq∂q * dq  (速度耦合项)
        C = torch.zeros(n, dof, device=x.device)
        dq = x[:, dof:]
        for i in range(dof):
            grad_i = torch.autograd.grad(
                dLddq[:, i].sum(), x, create_graph=True)[0][:, :dof]
            C[:, i] = (grad_i * dq).sum(dim=1)

        # 欧拉-拉格朗日：M * ddq = dL/dq - (∂²L/∂dq∂q)*dq
        rhs = dLdq - C   # (N, dof)

        # 求解 M * ddq = rhs，加正则化保证可逆
        M_reg = M + 1e-4 * torch.eye(dof, device=x.device).unsqueeze(0)
        ddq   = torch.linalg.solve(M_reg, rhs.unsqueeze(-1)).squeeze(-1)
        return ddq


# ─────────────────────────────────────────────────────
#  HNN：哈密顿神经网络
#
#  思路：神经网络学习标量哈密顿量 H(q, p)
#  哈密顿方程：
#    dq/dt =  ∂H/∂p
#    dp/dt = -∂H/∂q
#
#  实现：用 torch.autograd 计算 H 对 q、p 的梯度
#
#  输入：[theta, q1, q2, p_theta, p_q1, p_q2]  → (N, 6)
#  输出：[dtheta, dq1, dq2, dp_theta, dp_q1, dp_q2]  → (N, 6)
# ─────────────────────────────────────────────────────
class HNN(nn.Module):
    def __init__(self, dof=3, hidden_dims=(128, 128, 128)):
        super().__init__()
        self.dof = dof
        in_dim = 2 * dof
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, 1))   # 输出标量 H
        self.net = nn.Sequential(*layers)

    def hamiltonian(self, x):
        """计算哈密顿量 H(q, p)，返回 (N, 1)"""
        return self.net(x)

    def forward(self, x):
        """
        输入 x: (N, 2*dof)，前 dof 列为 q，后 dof 列为 p
        输出：辛梯度 (dq/dt, dp/dt)  (N, 2*dof)
        """
        x = x.requires_grad_(True)
        H = self.hamiltonian(x).sum()
        dH = torch.autograd.grad(H, x, create_graph=True)[0]  # (N, 2*dof)
        dof = self.dof
        dHdq = dH[:, :dof]    # ∂H/∂q
        dHdp = dH[:, dof:]    # ∂H/∂p

        # 哈密顿方程：dq/dt = ∂H/∂p，dp/dt = -∂H/∂q
        dqdt = dHdp
        dpdt = -dHdq
        return torch.cat([dqdt, dpdt], dim=1)


# ══════════════════════════════════════════════════════════════════════
# ③ 通用训练引擎
# ══════════════════════════════════════════════════════════════════════

def make_loader(X, Y, batch_size=64, shuffle=True):
    ds = TensorDataset(X.to(DEVICE), Y.to(DEVICE))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=False)


def train_model(
    model,
    train_X, train_Y,
    val_X,   val_Y,
    model_name    = 'model',
    epochs        = 200,
    lr            = 1e-3,
    batch_size    = 64,
    patience      = 15,         # 早停耐心
    lr_patience   = 10,         # 学习率衰减耐心
    lr_factor     = 0.5,
    weight_decay  = 1e-5,
):
    """
    通用训练函数，适用于 MDANN、LNN、HNN。
    返回 history dict：{'train_loss', 'val_loss', 'best_epoch'}
    """
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=lr_patience,
        factor=lr_factor, verbose=False)
    criterion = nn.MSELoss()

    tr_loader = make_loader(train_X, train_Y, batch_size, shuffle=True)
    va_loader = make_loader(val_X,   val_Y,   batch_size, shuffle=False)

    best_val_loss = float('inf')
    best_state    = None
    no_improve    = 0
    history       = {'train_loss': [], 'val_loss': [], 'best_epoch': 0}
    t0            = time.time()

    print(f"\n{'='*55}")
    print(f"  训练 {model_name}  参数量={sum(p.numel() for p in model.parameters()):,}")
    print(f"  epochs={epochs}  lr={lr}  batch={batch_size}  device={DEVICE}")
    print(f"{'='*55}")

    for epoch in range(1, epochs + 1):
        # ── 训练阶段 ──
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(tr_loader.dataset)

        # ── 验证阶段 ──
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                pred = model(xb)
                va_loss += criterion(pred, yb).item() * xb.size(0)
        va_loss /= len(va_loader.dataset)

        scheduler.step(va_loss)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)

        # ── 早停 + 保存最佳模型 ──
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            history['best_epoch'] = epoch
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 20 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  [{epoch:3d}/{epochs}] train={tr_loss:.6f}  "
                  f"val={va_loss:.6f}  lr={lr_now:.2e}  "
                  f"best_ep={history['best_epoch']}")

        if no_improve >= patience:
            print(f"  早停：{patience} 轮无改善，停止于第 {epoch} 轮")
            break

    # 恢复最佳参数
    model.load_state_dict(best_state)
    elapsed = time.time() - t0
    print(f"  完成！最佳验证损失={best_val_loss:.6f}  耗时={elapsed:.1f}s")

    # 保存模型
    torch.save({
        'state_dict': best_state,
        'history':    history,
        'model_name': model_name,
    }, f'models/{model_name}.pt')

    return model, history


# ══════════════════════════════════════════════════════════════════════
# ④ 实例化并训练三个模型
# ══════════════════════════════════════════════════════════════════════

EPOCHS     = 200
LR         = 1e-3
BATCH      = 64
HIDDEN     = (128, 128, 128)

# ── MDANN ──
mdann = MDANN(in_dim=6, out_dim=3, hidden_dims=HIDDEN)
mdann, hist_mdann = train_model(
    mdann, Xtn, Ytn, Xvn, Yvn,
    model_name='MDANN', epochs=EPOCHS, lr=LR, batch_size=BATCH)

# ── LNN ──
lnn = LNN(dof=3, hidden_dims=HIDDEN)
lnn, hist_lnn = train_model(
    lnn, Xtn, Ytn, Xvn, Yvn,
    model_name='LNN', epochs=EPOCHS, lr=LR, batch_size=BATCH)

# ── HNN ──
hnn = HNN(dof=3, hidden_dims=HIDDEN)
hnn, hist_hnn = train_model(
    hnn, Xhtn, Yhtn, Xhvn, Yhvn,
    model_name='HNN', epochs=EPOCHS, lr=LR, batch_size=BATCH)


# ══════════════════════════════════════════════════════════════════════
# ⑤ 评估：测试集损失 + 自回归轨迹预测
# ══════════════════════════════════════════════════════════════════════

def eval_test_loss(model, X_te, Y_te, name):
    """计算测试集 MSE 和 RMSE。"""
    model.eval()
    with torch.no_grad():
        Xt = X_te.to(DEVICE)
        Yt = Y_te.to(DEVICE)
        pred = model(Xt)
        mse  = nn.MSELoss()(pred, Yt).item()
    # 反归一化后的 RMSE（物理单位）
    pred_np = pred.cpu().numpy()
    true_np = Y_te.numpy()
    # 反归一化
    if name == 'HNN':
        pred_phys = norm_Yh.inverse_transform(pred_np)
        true_phys = norm_Yh.inverse_transform(true_np)
        # HNN 输出 6 维，取后 3 维（dp/dt ≈ 加速度）
        rmse_theta = float(np.sqrt(np.mean((pred_phys[:,3]-true_phys[:,3])**2)))
        rmse_q1    = float(np.sqrt(np.mean((pred_phys[:,4]-true_phys[:,4])**2)))
        rmse_q2    = float(np.sqrt(np.mean((pred_phys[:,5]-true_phys[:,5])**2)))
    else:
        pred_phys = norm_Y.inverse_transform(pred_np)
        true_phys = norm_Y.inverse_transform(true_np)
        rmse_theta = float(np.sqrt(np.mean((pred_phys[:,0]-true_phys[:,0])**2)))
        rmse_q1    = float(np.sqrt(np.mean((pred_phys[:,1]-true_phys[:,1])**2)))
        rmse_q2    = float(np.sqrt(np.mean((pred_phys[:,2]-true_phys[:,2])**2)))

    print(f"[{name}] 测试MSE={mse:.6f}  "
          f"RMSE_ddθ={np.degrees(rmse_theta):.4f}°/s²  "
          f"RMSE_ddq₁={rmse_q1*1e3:.4f}mm/s²  "
          f"RMSE_ddq₂={rmse_q2*1e3:.4f}mm/s²")
    return {'mse': mse, 'rmse_theta': rmse_theta,
            'rmse_q1': rmse_q1, 'rmse_q2': rmse_q2}


def rollout_mdann_lnn(model, traj, norm_x, norm_y, n_steps=None, model_name=''):
    """
    MDANN / LNN 自回归轨迹预测（欧拉积分）。
    返回预测的 theta、dtheta、q、dq 序列。
    """
    model.eval()
    t  = traj['t']
    dt = float(np.mean(np.diff(t)))
    if n_steps is None:
        n_steps = len(t) - 1

    # 初始状态
    theta   = float(traj['theta'][0])
    dtheta  = float(traj['dtheta'][0])
    q       = traj['q'][:, 0].copy().astype(np.float64)
    dq      = traj['dq'][:, 0].copy().astype(np.float64)

    pred_theta  = [theta]
    pred_dtheta = [dtheta]
    pred_q      = [q.copy()]
    pred_dq     = [dq.copy()]

    with torch.no_grad():
        for _ in range(n_steps):
            state = np.array([theta, dtheta, q[0], q[1], dq[0], dq[1]],
                             dtype=np.float32)
            state_n = norm_x.transform(state[None, :])   # (1, 6)
            inp = torch.tensor(state_n, device=DEVICE)
            out_n = model(inp).cpu().numpy()[0]           # (3,)
            acc   = norm_y.inverse_transform(out_n[None,:])[0]  # (3,)

            ddtheta = acc[0]; ddq = acc[1:]

            # 欧拉积分
            theta  += dtheta  * dt
            dtheta += ddtheta * dt
            q      += dq  * dt
            dq     += ddq * dt

            pred_theta.append(theta)
            pred_dtheta.append(dtheta)
            pred_q.append(q.copy())
            pred_dq.append(dq.copy())

    return {
        't':      t[:n_steps+1],
        'theta':  np.array(pred_theta),
        'dtheta': np.array(pred_dtheta),
        'q':      np.array(pred_q).T,
        'dq':     np.array(pred_dq).T,
    }


def rollout_hnn(model, traj, norm_xh, n_steps=None):
    """
    HNN 自回归轨迹预测（辛欧拉积分，保结构）。
    """
    model.eval()
    t  = traj['t']
    dt = float(np.mean(np.diff(t)))
    if n_steps is None:
        n_steps = len(t) - 1

    theta  = float(traj['theta'][0])
    q      = traj['q'][:, 0].copy().astype(np.float64)
    p_th   = float(traj['p_theta'][0])
    p_q    = traj['p_q'][:, 0].copy().astype(np.float64)

    pred_theta  = [theta]
    pred_q      = [q.copy()]
    pred_dtheta = [float(traj['dtheta'][0])]
    pred_dq     = [traj['dq'][:, 0].copy()]

    with torch.no_grad():
        for _ in range(n_steps):
            state = np.array([theta, q[0], q[1], p_th, p_q[0], p_q[1]],
                             dtype=np.float32)
            state_n = norm_xh.transform(state[None, :])
            inp = torch.tensor(state_n, device=DEVICE)
            out_n = model(inp).cpu().numpy()[0]   # (6,)
            # 反归一化
            out_phys = norm_xh.inverse_transform(out_n[None, :])[0]
            dqdt  = out_phys[:3]   # [dtheta, dq1, dq2]
            dpdt  = out_phys[3:]   # [dp_theta, dp_q1, dp_q2]

            # 辛欧拉积分（先更新 p，再更新 q，保辛结构）
            p_th  += dpdt[0] * dt
            p_q   += dpdt[1:] * dt
            theta += dqdt[0] * dt
            q     += dqdt[1:] * dt

            pred_theta.append(theta)
            pred_q.append(q.copy())
            pred_dtheta.append(dqdt[0])
            pred_dq.append(dqdt[1:].copy())

    return {
        't':      t[:n_steps+1],
        'theta':  np.array(pred_theta),
        'dtheta': np.array(pred_dtheta),
        'q':      np.array(pred_q).T,
        'dq':     np.array(pred_dq).T,
    }


# ── 测试集损失 ──
print("\n测试集评估：")
metrics_mdann = eval_test_loss(mdann, Xten,  Yten,  'MDANN')
metrics_lnn   = eval_test_loss(lnn,   Xten,  Yten,  'LNN')
metrics_hnn   = eval_test_loss(hnn,   Xhten, Yhten, 'HNN')

# ── 取测试轨迹做自回归预测 ──
test_traj = test_ds['trajectories'][0]
ROLLOUT   = min(200, len(test_traj['t']) - 1)

pred_mdann = rollout_mdann_lnn(mdann, test_traj, norm_X, norm_Y, ROLLOUT, 'MDANN')
pred_lnn   = rollout_mdann_lnn(lnn,   test_traj, norm_X, norm_Y, ROLLOUT, 'LNN')
pred_hnn   = rollout_hnn(hnn,  test_traj, norm_Xh, ROLLOUT)


# ══════════════════════════════════════════════════════════════════════
# ⑥ 可视化
# ══════════════════════════════════════════════════════════════════════

def savefig(name):
    plt.savefig(f'results/{name}', dpi=150, bbox_inches='tight',
                facecolor='#0d1117')
    plt.close()
    print(f"  ✓ results/{name}")


# ── 图1：训练损失曲线（三模型对比）──
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('训练损失曲线对比（归一化 MSE）',
             fontsize=13, fontweight='bold', color='#e6edf3')
for ax, (hist, name, col) in zip(axes, [
    (hist_mdann, 'MDANN', C[0]),
    (hist_lnn,   'LNN',   C[1]),
    (hist_hnn,   'HNN',   C[2]),
]):
    ep = range(1, len(hist['train_loss']) + 1)
    ax.semilogy(ep, hist['train_loss'], color=col,    lw=2,   label='训练损失')
    ax.semilogy(ep, hist['val_loss'],   color=col, lw=1.5, ls='--', alpha=0.8, label='验证损失')
    ax.axvline(hist['best_epoch'], color='#f0883e', ls=':', lw=1.2,
               label=f'最佳 ep={hist["best_epoch"]}')
    ax.set(xlabel='轮次', ylabel='MSE (log)', title=name)
    ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
savefig('fig1_loss_curves.png')


# ── 图2：关节角度预测对比 ──
true_t     = test_traj['t'][:ROLLOUT+1]
true_theta = test_traj['theta'][:ROLLOUT+1]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('自回归预测 vs 真实轨迹（关节角度）',
             fontsize=13, fontweight='bold', color='#e6edf3')

ax = axes[0, 0]
ax.plot(true_t, np.degrees(true_theta), color='white', lw=2.5, label='真实值', zorder=5)
ax.plot(pred_mdann['t'], np.degrees(pred_mdann['theta']), color=C[0], lw=2, ls='--', label='MDANN')
ax.plot(pred_lnn['t'],   np.degrees(pred_lnn['theta']),   color=C[1], lw=2, ls='-.',  label='LNN')
ax.plot(pred_hnn['t'],   np.degrees(pred_hnn['theta']),   color=C[2], lw=2, ls=':',   label='HNN')
ax.set(xlabel='时间 [s]', ylabel='角度 [°]', title='关节角度 θ(t)')
ax.legend(fontsize=9); ax.grid()

ax = axes[0, 1]
ax.plot(true_t, np.degrees(test_traj['dtheta'][:ROLLOUT+1]), color='white', lw=2.5, label='真实值')
ax.plot(pred_mdann['t'], np.degrees(pred_mdann['dtheta']), color=C[0], lw=2, ls='--', label='MDANN')
ax.plot(pred_lnn['t'],   np.degrees(pred_lnn['dtheta']),   color=C[1], lw=2, ls='-.',  label='LNN')
ax.plot(pred_hnn['t'],   np.degrees(pred_hnn['dtheta']),   color=C[2], lw=2, ls=':',   label='HNN')
ax.set(xlabel='时间 [s]', ylabel='角速度 [°/s]', title='关节角速度 dθ/dt')
ax.legend(fontsize=9); ax.grid()

ax = axes[1, 0]
ax.plot(true_t, test_traj['q'][0, :ROLLOUT+1]*1e3, color='white', lw=2.5, label='真实值')
ax.plot(pred_mdann['t'], pred_mdann['q'][0]*1e3, color=C[0], lw=2, ls='--', label='MDANN')
ax.plot(pred_lnn['t'],   pred_lnn['q'][0]*1e3,   color=C[1], lw=2, ls='-.',  label='LNN')
ax.plot(pred_hnn['t'],   pred_hnn['q'][0]*1e3,   color=C[2], lw=2, ls=':',   label='HNN')
ax.set(xlabel='时间 [s]', ylabel='模态位移 [mm]', title='柔性模态1 q₁(t)')
ax.legend(fontsize=9); ax.grid()

ax = axes[1, 1]
# 误差曲线
err_mdann = np.abs(np.degrees(pred_mdann['theta'] - true_theta))
err_lnn   = np.abs(np.degrees(pred_lnn['theta']   - true_theta))
err_hnn   = np.abs(np.degrees(pred_hnn['theta']   - true_theta))
ax.plot(true_t, err_mdann, color=C[0], lw=2, label=f'MDANN (均值={err_mdann.mean():.2f}°)')
ax.plot(true_t, err_lnn,   color=C[1], lw=2, label=f'LNN   (均值={err_lnn.mean():.2f}°)')
ax.plot(true_t, err_hnn,   color=C[2], lw=2, label=f'HNN   (均值={err_hnn.mean():.2f}°)')
ax.set(xlabel='时间 [s]', ylabel='角度误差 [°]', title='关节角度绝对误差')
ax.legend(fontsize=9); ax.grid()

plt.tight_layout()
savefig('fig2_trajectory_compare.png')


# ── 图3：柔性模态预测对比 ──
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('自回归预测 vs 真实轨迹（柔性模态）',
             fontsize=13, fontweight='bold', color='#e6edf3')

for mode_idx, row_axes in enumerate(axes):
    true_q  = test_traj['q'][mode_idx, :ROLLOUT+1] * 1e3
    true_dq = test_traj['dq'][mode_idx, :ROLLOUT+1] * 1e3

    ax = row_axes[0]
    ax.plot(true_t, true_q, color='white', lw=2.5, label='真实值')
    ax.plot(pred_mdann['t'], pred_mdann['q'][mode_idx]*1e3, color=C[0], lw=2, ls='--', label='MDANN')
    ax.plot(pred_lnn['t'],   pred_lnn['q'][mode_idx]*1e3,   color=C[1], lw=2, ls='-.',  label='LNN')
    ax.plot(pred_hnn['t'],   pred_hnn['q'][mode_idx]*1e3,   color=C[2], lw=2, ls=':',   label='HNN')
    ax.set(xlabel='时间 [s]', ylabel='位移 [mm]', title=f'模态{mode_idx+1} 位移 q{mode_idx+1}')
    ax.legend(fontsize=9); ax.grid()

    ax = row_axes[1]
    e_m = np.abs(pred_mdann['q'][mode_idx]*1e3 - true_q)
    e_l = np.abs(pred_lnn['q'][mode_idx]*1e3   - true_q)
    e_h = np.abs(pred_hnn['q'][mode_idx]*1e3   - true_q)
    ax.plot(true_t, e_m, color=C[0], lw=2, label=f'MDANN ({e_m.mean():.3f}mm)')
    ax.plot(true_t, e_l, color=C[1], lw=2, label=f'LNN   ({e_l.mean():.3f}mm)')
    ax.plot(true_t, e_h, color=C[2], lw=2, label=f'HNN   ({e_h.mean():.3f}mm)')
    ax.set(xlabel='时间 [s]', ylabel='误差 [mm]',
           title=f'模态{mode_idx+1} 位移绝对误差')
    ax.legend(fontsize=9); ax.grid()

plt.tight_layout()
savefig('fig3_flex_modes_compare.png')


# ── 图4：能量误差对比（HNN 的优势）──
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('预测轨迹能量误差对比（HNN 保辛结构优势）',
             fontsize=13, fontweight='bold', color='#e6edf3')

Kfc_np = train_ds['params']['Kfc']   # (2, 2)
J_hub  = train_ds['params']['J_hub']

def compute_energy(pred):
    KE = 0.5 * J_hub * pred['dtheta']**2
    PE = np.array([
        0.5 * float(pred['q'][:, k] @ Kfc_np @ pred['q'][:, k])
        for k in range(pred['q'].shape[1])
    ])
    return KE + PE

true_E  = compute_energy({'dtheta': test_traj['dtheta'][:ROLLOUT+1],
                           'q': test_traj['q'][:, :ROLLOUT+1]})
E_mdann = compute_energy(pred_mdann)
E_lnn   = compute_energy(pred_lnn)
E_hnn   = compute_energy(pred_hnn)

ax = axes[0]
ax.plot(true_t, true_E*1e3,  color='white', lw=2.5, label='真实')
ax.plot(true_t, E_mdann*1e3, color=C[0], lw=2, ls='--', label='MDANN')
ax.plot(true_t, E_lnn*1e3,   color=C[1], lw=2, ls='-.',  label='LNN')
ax.plot(true_t, E_hnn*1e3,   color=C[2], lw=2, ls=':',   label='HNN')
ax.set(xlabel='时间 [s]', ylabel='KE+PE_flex [mJ]', title='弹性势能+动能')
ax.legend(fontsize=9); ax.grid()

ax = axes[1]
for E_pred, name, col in [(E_mdann, 'MDANN', C[0]),
                           (E_lnn, 'LNN', C[1]),
                           (E_hnn, 'HNN', C[2])]:
    err = np.abs(E_pred - true_E) * 1e3
    ax.plot(true_t, err, color=col, lw=2, label=f'{name} (均值={err.mean():.3f}mJ)')
ax.set(xlabel='时间 [s]', ylabel='能量误差 [mJ]', title='能量绝对误差')
ax.legend(fontsize=9); ax.grid()

# 柱状图：各指标汇总
ax = axes[2]
models_name = ['MDANN', 'LNN', 'HNN']
rmse_theta_all = [
    np.degrees(np.sqrt(np.mean((pred_mdann['theta'] - true_theta)**2))),
    np.degrees(np.sqrt(np.mean((pred_lnn['theta']   - true_theta)**2))),
    np.degrees(np.sqrt(np.mean((pred_hnn['theta']   - true_theta)**2))),
]
x_pos = np.arange(3)
bars = ax.bar(x_pos, rmse_theta_all, color=[C[0], C[1], C[2]],
              width=0.5, edgecolor='#30363d')
for bar, val in zip(bars, rmse_theta_all):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.2f}°', ha='center', va='bottom', fontsize=10, color='#c9d1d9')
ax.set_xticks(x_pos); ax.set_xticklabels(models_name)
ax.set(ylabel='RMSE [°]', title='关节角度 RMSE 对比')
ax.grid(axis='y', alpha=0.4)

plt.tight_layout()
savefig('fig4_energy_compare.png')


# ── 图5：三模型性能汇总表 ──
fig, ax = plt.subplots(figsize=(10, 4))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')
ax.axis('off')

headers = ['模型', '原理', '测试MSE', 'RMSE_ddθ\n[°/s²]',
           '角度RMSE\n[°]', '模态1RMSE\n[mm]', '参数量']

def count_params(m):
    return f"{sum(p.numel() for p in m.parameters()):,}"

data_table = [
    ['MDANN', '直接回归加速度',
     f"{metrics_mdann['mse']:.5f}",
     f"{np.degrees(metrics_mdann['rmse_theta']):.4f}",
     f"{np.degrees(np.sqrt(np.mean((pred_mdann['theta']-true_theta)**2))):.3f}",
     f"{np.sqrt(np.mean((pred_mdann['q'][0]-test_traj['q'][0,:ROLLOUT+1])**2))*1e3:.4f}",
     count_params(mdann)],
    ['LNN', '欧拉-拉格朗日方程',
     f"{metrics_lnn['mse']:.5f}",
     f"{np.degrees(metrics_lnn['rmse_theta']):.4f}",
     f"{np.degrees(np.sqrt(np.mean((pred_lnn['theta']-true_theta)**2))):.3f}",
     f"{np.sqrt(np.mean((pred_lnn['q'][0]-test_traj['q'][0,:ROLLOUT+1])**2))*1e3:.4f}",
     count_params(lnn)],
    ['HNN', '哈密顿辛结构',
     f"{metrics_hnn['mse']:.5f}",
     f"{np.degrees(metrics_hnn['rmse_theta']):.4f}",
     f"{np.degrees(np.sqrt(np.mean((pred_hnn['theta']-true_theta)**2))):.3f}",
     f"{np.sqrt(np.mean((pred_hnn['q'][0]-test_traj['q'][0,:ROLLOUT+1])**2))*1e3:.4f}",
     count_params(hnn)],
]

col_colors = [['#1f3a5f']*len(headers)]
cell_colors = [['#161b22']*len(headers) for _ in data_table]

tbl = ax.table(
    cellText    = data_table,
    colLabels   = headers,
    cellLoc     = 'center',
    loc         = 'center',
    colColours  = col_colors[0],
    cellColours = cell_colors,
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.2, 2.2)

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor('#30363d')
    cell.set_text_props(color='#e6edf3')

ax.set_title('三模型性能对比汇总', color='#e6edf3',
             fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
savefig('fig5_summary_table.png')


# ══════════════════════════════════════════════════════════════════════
# ⑦ 最终输出摘要
# ══════════════════════════════════════════════════════════════════════

print(f"""
{'='*60}
  训练完成！结果汇总
{'='*60}
  MDANN  最佳epoch={hist_mdann['best_epoch']}  验证MSE={min(hist_mdann['val_loss']):.6f}
  LNN    最佳epoch={hist_lnn['best_epoch']}    验证MSE={min(hist_lnn['val_loss']):.6f}
  HNN    最佳epoch={hist_hnn['best_epoch']}    验证MSE={min(hist_hnn['val_loss']):.6f}

  模型文件：
    models/MDANN.pt
    models/LNN.pt
    models/HNN.pt

  结果图表：
    results/fig1_loss_curves.png       训练损失曲线
    results/fig2_trajectory_compare.png 关节角度预测对比
    results/fig3_flex_modes_compare.png 柔性模态预测对比
    results/fig4_energy_compare.png     能量误差对比
    results/fig5_summary_table.png      性能汇总表

  模型加载示例：
    checkpoint = torch.load('models/MDANN.pt')
    mdann.load_state_dict(checkpoint['state_dict'])
    mdann.eval()
{'='*60}
  三模型对比说明：
    MDANN  训练最快，适合工程部署，无物理约束
    LNN    保拉格朗日结构，守恒量更准确，训练较慢
    HNN    保辛结构（能量守恒），长期预测最稳定
{'='*60}
""")
