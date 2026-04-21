# ═══════════════════════════════════════════════════════════════════════
#  柔性机械臂训练数据生成代码（Google Colab 专用）
#  Flexible Robotic Arm — Training Data Generation
#
#  坐标系约定：
#    theta = 0      → 臂杆水平向右
#    theta = -pi/2  → 臂杆竖直向下（稳定平衡点）
#    theta = +pi/2  → 臂杆竖直向上（不稳定平衡点）
#
#  重力力矩：tau_g = -m*g*(L/2)*cos(theta)
#    → theta=-pi/2 时 cos=-0，力矩为零，平衡在竖直向下 ✓
#
#  状态向量：x = [theta, dtheta, q1, q2, dq1, dq2]
# ═══════════════════════════════════════════════════════════════════════

# ── 第一步：安装依赖（首次运行取消注释）──────────────────────────────
# !pip install numpy scipy matplotlib tqdm -q

import os
import pickle
import time
import warnings

import numpy as np
from scipy.integrate import solve_ivp

warnings.filterwarnings("ignore")

# ── 在 Colab 中可用 tqdm 显示进度条 ──
try:
    from tqdm.notebook import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

os.makedirs("data", exist_ok=True)
print("柔性机械臂训练数据生成系统 启动")
print("=" * 55)


# ══════════════════════════════════════════════════════════════════════
# ① 系统物理参数
# ══════════════════════════════════════════════════════════════════════

# ── 几何参数 ──
L = 0.5  # 臂长 [m]
b = 0.02  # 截面宽度 [m]
h_sec = 0.003  # 截面高度 [m]
A = b * h_sec  # 截面积 [m²]
I_sec = b * h_sec**3 / 12  # 截面惯性矩 [m⁴]

# ── 材料参数（铝合金）──
rho = 2700.0  # 密度 [kg/m³]
E = 70e9  # 弹性模量 [Pa]
EI = E * I_sec  # 抗弯刚度 [N·m²]

# ── 运动参数 ──
m_total = rho * A * L  # 臂总质量 [kg]
J_hub = 0.005  # 关节转动惯量 [kg·m²]
d_damp = 0.01  # 关节阻尼系数 [N·m·s/rad]
g = 9.81  # 重力加速度 [m/s²]

# ── 模态参数 ──
N_MODES = 2
BETA_L = [1.87510407, 4.69409113]  # 悬臂梁前两阶特征方程根

print(f"臂长={L}m  质量={m_total:.4f}kg  EI={EI:.2f}N·m²")
print(f"模态数={N_MODES}  J_hub={J_hub}  阻尼={d_damp}")


# ══════════════════════════════════════════════════════════════════════
# ② 振型函数（悬臂梁解析解）
# ══════════════════════════════════════════════════════════════════════


def make_phi(beta_L):
    """构造第 i 阶悬臂梁振型函数及其一、二阶导数"""
    beta = beta_L / L
    sigma = (np.cosh(beta_L) + np.cos(beta_L)) / (np.sinh(beta_L) + np.sin(beta_L))
    phi = lambda x, b=beta, s=sigma: (np.cosh(b * x) - np.cos(b * x) - s * (np.sinh(b * x) - np.sin(b * x)))
    dphi = lambda x, b=beta, s=sigma: b * (np.sinh(b * x) + np.sin(b * x) - s * (np.cosh(b * x) - np.cos(b * x)))
    d2phi = lambda x, b=beta, s=sigma: b**2 * (np.cosh(b * x) + np.cos(b * x) - s * (np.sinh(b * x) + np.sin(b * x)))
    return phi, dphi, d2phi


fns = [make_phi(bl) for bl in BETA_L]
mf = [f[0] for f in fns]  # 振型函数 φᵢ(x)
md1 = [f[1] for f in fns]  # 一阶导数 φᵢ'(x)
md2 = [f[2] for f in fns]  # 二阶导数 φᵢ''(x)


# ══════════════════════════════════════════════════════════════════════
# ③ 高斯积分框架 + 预计算常数矩阵
# ══════════════════════════════════════════════════════════════════════

N_GAUSS = 8
gpts_std, gwts_std = np.polynomial.legendre.leggauss(N_GAUSS)
gpts = 0.5 * L * (gpts_std + 1.0)  # 映射到 [0, L]
gwts = 0.5 * L * gwts_std

# 振型函数在积分点处的值（预计算，避免每步重复）
phi_gp = np.array([mf[i](gpts) for i in range(N_MODES)])  # (2, N_GAUSS)
d1_gp = np.array([md1[i](gpts) for i in range(N_MODES)])

# 模态质量矩阵 Mf  (对角，= ρA·∫φᵢφⱼdx)
Mfc = np.zeros((N_MODES, N_MODES))
# 模态刚度矩阵 Kf  (对角，= EI·∫φᵢ''φⱼ''dx)
Kfc = np.zeros((N_MODES, N_MODES))
for i in range(N_MODES):
    for j in range(N_MODES):
        Mfc[i, j] = np.dot(gwts, rho * A * phi_gp[i] * phi_gp[j])
        Kfc[i, j] = np.dot(gwts, EI * md2[i](gpts) * md2[j](gpts))

# 科里奥利力预计算项（∫ρA·x·φᵢ'dx，常数）
corF_pre = np.array([np.dot(gwts, rho * A * gpts * d1_gp[i]) for i in range(N_MODES)])

# 固有频率（理论值）
fn_theory = [np.sqrt(EI / (rho * A * L**4)) * bl**2 / (2 * np.pi) for bl in BETA_L]

print(f"\n模态质量矩阵 Mf（对角线）: {np.diag(Mfc)}")
print(f"模态刚度矩阵 Kf（对角线）: {np.diag(Kfc)}")
print(f"固有频率: f₁={fn_theory[0]:.2f} Hz, f₂={fn_theory[1]:.2f} Hz")


# ══════════════════════════════════════════════════════════════════════
# ④ 动力学方程（ODE 右端函数）
# ══════════════════════════════════════════════════════════════════════
#
#  状态 state = [theta, dtheta, q1, q2, dq1, dq2]
#
#  质量矩阵（分块）：
#    M = | J_tot   D^T |
#        |  D      Mf  |
#
#  力向量：
#    F_rigid = tau_ext + tau_gravity + coriolis_rigid
#    F_flex  = -Kf @ q  + coriolis_flex
#
#  关键修正：
#    tau_gravity = -m*g*(L/2)*cos(theta)   ← 平衡在竖直向下
#
# ══════════════════════════════════════════════════════════════════════


def dynamics(t, state, tau_fn):
    theta, dtheta = state[0], state[1]
    q = state[2 : 2 + N_MODES]
    dq = state[2 + N_MODES : 2 + 2 * N_MODES]

    # 变形场 w(x) = Σ φᵢ(x)·qᵢ （在积分点处）
    w = phi_gp.T @ q  # shape (N_GAUSS,)

    # 总刚体惯量（含柔性附加惯量）
    J_tot = J_hub + np.dot(gwts, rho * A * (gpts**2 + w**2))

    # 刚柔耦合向量 D = ρA∫w·φᵢdx
    D = phi_gp @ (gwts * rho * A * w)  # shape (N_MODES,)

    # 组装系统质量矩阵 (1+N_MODES, 1+N_MODES)
    n = N_MODES
    Msys = np.zeros((1 + n, 1 + n))
    Msys[0, 0] = J_tot
    Msys[0, 1:] = D
    Msys[1:, 0] = D
    Msys[1:, 1:] = Mfc

    # 科里奥利力（刚体部分）
    dJ = 2.0 * (phi_gp @ (gwts * rho * A * w))
    cor_r = 0.5 * dtheta**2 * (dJ @ dq) - d_damp * dtheta

    # 科里奥利力（柔性部分）
    cor_f = -(dtheta**2) * (D + corF_pre)

    # 重力力矩：cos(theta) 使平衡在 theta=-pi/2（竖直向下）✓
    tau_g = -m_total * g * (L / 2) * np.cos(theta)

    # 外部驱动力矩
    tau_ext = tau_fn(t)

    # 力向量
    F = np.zeros(1 + n)
    F[0] = tau_ext + tau_g + cor_r
    F[1:] = -Kfc @ q + cor_f

    # 求解加速度
    acc = np.linalg.solve(Msys, F)

    # 状态导数
    ds = np.zeros(2 + 2 * n)
    ds[0] = dtheta
    ds[1] = acc[0]
    ds[2 : 2 + n] = dq
    ds[2 + n : 2 + 2 * n] = acc[1:]
    return ds


# ══════════════════════════════════════════════════════════════════════
# ⑤ 单条轨迹生成函数
# ══════════════════════════════════════════════════════════════════════


def generate_trajectory(
    theta0,  # 初始关节角度 [rad]，0=水平，-pi/2=竖直向下
    q0=None,  # 初始柔性模态位移，None 则随机微小扰动
    dtheta0=0.0,  # 初始角速度
    tau_fn=lambda t: 0.0,  # 力矩函数
    t_end=5.0,  # 仿真时长 [s]
    n_pts=500,  # 输出时间点数
    noise_level=0.0,  # 传感器噪声标准差（0=无噪声）
):
    """
    生成一条完整仿真轨迹。

    返回字典包含：
      t         时间序列          (T,)
      theta     关节角度          (T,)
      dtheta    关节角速度        (T,)
      ddtheta   关节角加速度      (T,)
      q         柔性模态位移      (N_MODES, T)
      dq        柔性模态速度      (N_MODES, T)
      ddq       柔性模态加速度    (N_MODES, T)
      KE        动能              (T,)
      PE        弹性势能          (T,)
      E         总机械能          (T,)
      p_theta   广义动量（关节）  (T,)  用于 HNN
      p_q       广义动量（柔性）  (N_MODES, T)  用于 HNN
    """
    n = N_MODES
    if q0 is None:
        q0 = np.random.uniform(-2e-4, 2e-4, n)

    s0 = np.zeros(2 + 2 * n)
    s0[0] = theta0
    s0[1] = dtheta0
    s0[2 : 2 + n] = q0

    t_eval = np.linspace(0, t_end, n_pts)

    sol = solve_ivp(
        fun=lambda t, s: dynamics(t, s, tau_fn),
        t_span=(0, t_end),
        y0=s0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-5,
        atol=1e-7,
        max_step=0.02,
    )

    if not sol.success:
        return None  # 积分失败时返回 None，调用方跳过

    t = sol.t
    theta = sol.y[0]
    dtheta = sol.y[1]
    q = sol.y[2 : 2 + n]  # (n, T)
    dq = sol.y[2 + n : 2 + 2 * n]  # (n, T)

    # 角加速度（中心差分）
    ddtheta = np.gradient(dtheta, t)
    ddq = np.array([np.gradient(dq[i], t) for i in range(n)])

    # 能量
    KE = 0.5 * J_hub * dtheta**2
    PE = np.array([0.5 * float(q[:, k] @ Kfc @ q[:, k]) for k in range(len(t))])
    energy = KE + PE

    # 广义动量（用于哈密顿神经网络 HNN）
    # 简化：p_theta ≈ J_hub * dtheta（忽略刚柔耦合动量贡献，作为近似）
    p_theta = J_hub * dtheta
    p_q = Mfc @ dq  # (n, T) — 精确模态动量

    # 传感器噪声
    if noise_level > 0:
        theta = theta + np.random.normal(0, noise_level, theta.shape)
        dtheta = dtheta + np.random.normal(0, noise_level, dtheta.shape)
        q = q + np.random.normal(0, noise_level * 0.1, q.shape)
        dq = dq + np.random.normal(0, noise_level * 0.1, dq.shape)

    return {
        "t": t,
        "theta": theta,
        "dtheta": dtheta,
        "ddtheta": ddtheta,
        "q": q,
        "dq": dq,
        "ddq": ddq,
        "KE": KE,
        "PE": PE,
        "E": energy,
        "p_theta": p_theta,
        "p_q": p_q,
        # 元信息
        "meta": {
            "theta0": theta0,
            "dtheta0": dtheta0,
            "q0": q0.copy(),
            "t_end": t_end,
            "n_pts": n_pts,
            "noise_level": noise_level,
        },
    }


# ══════════════════════════════════════════════════════════════════════
# ⑥ 力矩函数库
# ══════════════════════════════════════════════════════════════════════


def make_torque(torque_type, **kwargs):
    """
    构造力矩函数。

    torque_type 可选：
      'free'        自由摆动，τ=0
      'constant'    恒定力矩，τ=amp
      'sinusoidal'  正弦力矩，τ=amp·sin(2π·freq·t)
      'step'        阶跃力矩，t<t_switch 时 τ=amp，之后 τ=0
      'random_zoh'  零阶保持随机力矩（每隔 interval 秒更新一次）
    """
    if torque_type == "free":
        return lambda t: 0.0

    elif torque_type == "constant":
        amp = kwargs.get("amp", 0.3)
        return lambda t, a=amp: a

    elif torque_type == "sinusoidal":
        amp = kwargs.get("amp", 0.4)
        freq = kwargs.get("freq", 1.5)
        return lambda t, a=amp, f=freq: a * np.sin(2 * np.pi * f * t)

    elif torque_type == "step":
        amp = kwargs.get("amp", 0.5)
        t_switch = kwargs.get("t_switch", 1.0)
        return lambda t, a=amp, ts=t_switch: a if t < ts else 0.0

    elif torque_type == "random_zoh":
        # 零阶保持随机力矩：提前采样好，避免每步重新采样
        amp = kwargs.get("amp", 0.5)
        interval = kwargs.get("interval", 0.5)
        t_end = kwargs.get("t_end", 5.0)
        n_switch = int(t_end / interval) + 2
        values = np.random.uniform(-amp, amp, n_switch)

        def _zoh(t, iv=interval, vs=values):
            idx = min(int(t / iv), len(vs) - 1)
            return float(vs[idx])

        return _zoh

    else:
        raise ValueError(f"未知力矩类型: {torque_type}")


# ══════════════════════════════════════════════════════════════════════
# ⑦ 批量数据集生成
# ══════════════════════════════════════════════════════════════════════


def generate_dataset(
    n_traj=500,  # 轨迹总数
    split="train",  # 'train' | 'val' | 'test'
    seed=42,
    t_end=5.0,  # 单条仿真时长 [s]
    n_pts=500,  # 每条轨迹时间点数
    noise_level=0.001,  # 传感器噪声
    save_dir="data",
):
    """
    批量生成训练/验证/测试数据集并保存为 .pkl 文件。

    初始条件采样策略：
      train  : theta0 ∈ [-pi/2 ± pi/3]，即平衡点附近 ±60°；
                多种力矩类型混合
      val    : theta0 ∈ [-pi/2 ± pi/4]；自由摆动为主
      test   : theta0 ∈ 宽范围；含训练未见过的正弦激励频率
    """
    np.random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    t0_all = time.time()

    # 平衡点 theta_eq = -pi/2（竖直向下）
    theta_eq = -np.pi / 2

    # ── 按 split 定义采样策略 ──
    if split == "train":
        theta_range = (theta_eq - np.radians(60), theta_eq + np.radians(60))
        dtheta_range = (-0.5, 0.5)
        torque_mix = [
            ("free", 0.40),  # 40% 自由摆动
            ("sinusoidal", 0.25),  # 25% 正弦激励（低频）
            ("constant", 0.15),  # 15% 恒定力矩
            ("step", 0.10),  # 10% 阶跃力矩
            ("random_zoh", 0.10),  # 10% 随机力矩
        ]
    elif split == "val":
        theta_range = (theta_eq - np.radians(45), theta_eq + np.radians(45))
        dtheta_range = (-0.3, 0.3)
        torque_mix = [
            ("free", 0.60),
            ("sinusoidal", 0.25),
            ("constant", 0.15),
        ]
    else:  # test
        theta_range = (theta_eq - np.radians(75), theta_eq + np.radians(75))
        dtheta_range = (-1.0, 1.0)
        torque_mix = [
            ("free", 0.30),
            ("sinusoidal", 0.40),  # 使用训练集未见过的频率
            ("random_zoh", 0.30),
        ]

    # 预计算力矩类型的累积概率（用于采样）
    torque_types, torque_probs = zip(*torque_mix)
    torque_probs = np.array(torque_probs)
    torque_probs /= torque_probs.sum()

    trajectories = []
    n_failed = 0

    iter_range = range(n_traj)
    if HAS_TQDM:
        iter_range = tqdm(iter_range, desc=f"生成 {split} 集")

    for i in iter_range:
        # ── 随机初始条件 ──
        theta0 = np.random.uniform(*theta_range)
        dtheta0 = np.random.uniform(*dtheta_range)
        q0 = np.random.uniform(-3e-4, 3e-4, N_MODES)

        # ── 随机力矩类型 ──
        ttype = np.random.choice(torque_types, p=torque_probs)

        if ttype == "free":
            tau_fn = make_torque("free")
        elif ttype == "constant":
            amp = np.random.uniform(-0.5, 0.5)
            tau_fn = make_torque("constant", amp=amp)
        elif ttype == "sinusoidal":
            amp = np.random.uniform(0.1, 0.6)
            # train: 0.5~3 Hz; test: 3~6 Hz（频率外推）
            if split == "test":
                freq = np.random.uniform(3.0, 6.0)
            else:
                freq = np.random.uniform(0.5, 3.0)
            tau_fn = make_torque("sinusoidal", amp=amp, freq=freq)
        elif ttype == "step":
            amp = np.random.uniform(-0.5, 0.5)
            t_switch = np.random.uniform(0.5, t_end * 0.6)
            tau_fn = make_torque("step", amp=amp, t_switch=t_switch)
        else:  # random_zoh
            amp = np.random.uniform(0.1, 0.5)
            interval = np.random.uniform(0.3, 1.0)
            tau_fn = make_torque("random_zoh", amp=amp, interval=interval, t_end=t_end)

        # ── 仿真 ──
        traj = generate_trajectory(
            theta0=theta0,
            q0=q0,
            dtheta0=dtheta0,
            tau_fn=tau_fn,
            t_end=t_end,
            n_pts=n_pts,
            noise_level=noise_level,
        )

        if traj is None:
            n_failed += 1
            continue

        traj["meta"]["torque_type"] = ttype
        trajectories.append(traj)

        if not HAS_TQDM and (i + 1) % 50 == 0:
            elapsed = time.time() - t0_all
            print(f"  [{split}] {i+1}/{n_traj}  已用时 {elapsed:.1f}s")

    print(f"\n[{split}] 成功={len(trajectories)}  失败={n_failed}")

    # ── 计算归一化统计量 ──
    all_theta = np.concatenate([tr["theta"] for tr in trajectories])
    all_dtheta = np.concatenate([tr["dtheta"] for tr in trajectories])
    all_q = np.concatenate([tr["q"] for tr in trajectories], axis=1)
    all_dq = np.concatenate([tr["dq"] for tr in trajectories], axis=1)
    all_ddtheta = np.concatenate([tr["ddtheta"] for tr in trajectories])
    all_ddq = np.concatenate([tr["ddq"] for tr in trajectories], axis=1)

    stats = {
        "theta_mean": float(all_theta.mean()),
        "theta_std": float(all_theta.std()) + 1e-8,
        "dtheta_mean": float(all_dtheta.mean()),
        "dtheta_std": float(all_dtheta.std()) + 1e-8,
        "q_mean": all_q.mean(axis=1),
        "q_std": all_q.std(axis=1) + 1e-8,
        "dq_mean": all_dq.mean(axis=1),
        "dq_std": all_dq.std(axis=1) + 1e-8,
        "ddtheta_mean": float(all_ddtheta.mean()),
        "ddtheta_std": float(all_ddtheta.std()) + 1e-8,
        "ddq_mean": all_ddq.mean(axis=1),
        "ddq_std": all_ddq.std(axis=1) + 1e-8,
    }

    dataset = {
        "trajectories": trajectories,
        "stats": stats,
        "split": split,
        "n_traj": len(trajectories),
        "params": {
            "L": L,
            "rho": rho,
            "E": E,
            "A": A,
            "I_sec": I_sec,
            "J_hub": J_hub,
            "d_damp": d_damp,
            "g": g,
            "N_MODES": N_MODES,
            "BETA_L": BETA_L,
            "Mfc": Mfc,
            "Kfc": Kfc,
            "fn_theory": fn_theory,
            "theta_eq": -np.pi / 2,
            "coord_note": "theta=0 horizontal, theta=-pi/2 vertical down (equilibrium)",
        },
    }

    path = os.path.join(save_dir, f"flexible_arm_{split}.pkl")
    with open(path, "wb") as f:
        pickle.dump(dataset, f, protocol=4)

    elapsed = time.time() - t0_all
    print(f"[{split}] 保存至 {path}  ({elapsed:.1f}s)")
    print(f"  θ 均值={stats['theta_mean']:.3f} rad  " f"σ={stats['theta_std']:.3f}")
    print(f"  q₁均值={stats['q_mean'][0]*1e3:.4f}mm  " f"σ={stats['q_std'][0]*1e3:.4f}mm")
    return dataset


# ══════════════════════════════════════════════════════════════════════
# ⑧ 数据格式转换（用于不同模型）
# ══════════════════════════════════════════════════════════════════════


def to_mdann_format(traj, stats=None, normalize=True):
    """
    MDANN / NODE / LNN（拉格朗日）格式
    ─────────────────────────────────
    输入 X : [theta, dtheta, q1, q2, dq1, dq2]   shape (T, 6)
    输出 Y : [ddtheta, ddq1, ddq2]               shape (T, 3)

    可选归一化（使用数据集统计量）。
    """
    n = N_MODES
    T = len(traj["t"])
    X = np.zeros((T, 2 + 2 * n))
    X[:, 0] = traj["theta"]
    X[:, 1] = traj["dtheta"]
    X[:, 2 : 2 + n] = traj["q"].T
    X[:, 2 + n : 2 + 2 * n] = traj["dq"].T

    Y = np.zeros((T, 1 + n))
    Y[:, 0] = traj["ddtheta"]
    Y[:, 1:] = traj["ddq"].T

    if normalize and stats is not None:
        X[:, 0] = (X[:, 0] - stats["theta_mean"]) / stats["theta_std"]
        X[:, 1] = (X[:, 1] - stats["dtheta_mean"]) / stats["dtheta_std"]
        for i in range(n):
            X[:, 2 + i] = (X[:, 2 + i] - stats["q_mean"][i]) / stats["q_std"][i]
            X[:, 2 + n + i] = (X[:, 2 + n + i] - stats["dq_mean"][i]) / stats["dq_std"][i]
        Y[:, 0] = (Y[:, 0] - stats["ddtheta_mean"]) / stats["ddtheta_std"]
        for i in range(n):
            Y[:, 1 + i] = (Y[:, 1 + i] - stats["ddq_mean"][i]) / stats["ddq_std"][i]

    return X, Y


def to_hnn_format(traj, stats=None, normalize=False):
    """
    HNN（哈密顿神经网络）格式
    ─────────────────────────
    输入 X : [theta, p_theta, q1, q2, p_q1, p_q2]   shape (T, 6)
    输出 Y : [dtheta, dp_theta, dq1, dq2, dp_q1, dp_q2]  shape (T, 6)
             即哈密顿方程的广义坐标和动量导数

    dp_theta = -∂H/∂theta（由力矩方程给出，此处用 ddtheta 近似）
    dp_q     = -∂H/∂q    = -Kf @ q（弹性力）
    """
    n = N_MODES
    T = len(traj["t"])

    X = np.zeros((T, 2 + 2 * n))
    X[:, 0] = traj["theta"]
    X[:, 1] = traj["p_theta"]
    X[:, 2 : 2 + n] = traj["q"].T
    X[:, 2 + n : 2 + 2 * n] = traj["p_q"].T

    Y = np.zeros((T, 2 + 2 * n))
    Y[:, 0] = traj["dtheta"]
    Y[:, 1] = traj["ddtheta"]  # ≈ dp_theta/dt
    Y[:, 2 : 2 + n] = traj["dq"].T
    # dp_q/dt = -Kf @ q（弹性恢复力，哈密顿方程）
    for k in range(T):
        Y[k, 2 + n : 2 + 2 * n] = -(Kfc @ traj["q"][:, k])

    return X, Y


def to_sequence_format(traj, seq_len=20, stride=5):
    """
    序列模型（LSTM / Transformer）格式
    ───────────────────────────────────
    将单条轨迹切片为 (N_windows, seq_len, 6) 的输入窗口
    和 (N_windows, 3) 的下一时刻加速度目标。

    seq_len : 输入序列长度（时间步数）
    stride  : 窗口滑动步长
    """
    n = N_MODES
    T = len(traj["t"])

    state = np.stack(
        [
            traj["theta"],
            traj["dtheta"],
            traj["q"][0],
            traj["q"][1],
            traj["dq"][0],
            traj["dq"][1],
        ],
        axis=1,
    )  # (T, 6)

    accel = np.stack(
        [
            traj["ddtheta"],
            traj["ddq"][0],
            traj["ddq"][1],
        ],
        axis=1,
    )  # (T, 3)

    windows_X, windows_Y = [], []
    for start in range(0, T - seq_len, stride):
        windows_X.append(state[start : start + seq_len])
        windows_Y.append(accel[start + seq_len - 1])

    if len(windows_X) == 0:
        return None, None
    return np.array(windows_X), np.array(windows_Y)  # (N, L, 6), (N, 3)


def build_full_dataset(dataset, fmt="mdann", normalize=True, **kwargs):
    """
    将整个数据集的所有轨迹转换为指定格式，并合并为大数组。

    fmt 可选：'mdann'  'hnn'  'sequence'
    返回 (X, Y) numpy 数组，可直接用于 torch.utils.data.TensorDataset
    """
    stats = dataset["stats"] if normalize else None
    Xs, Ys = [], []

    for traj in dataset["trajectories"]:
        if fmt == "mdann":
            X, Y = to_mdann_format(traj, stats, normalize=normalize)
        elif fmt == "hnn":
            X, Y = to_hnn_format(traj, stats, normalize=normalize)
        elif fmt == "sequence":
            seq_len = kwargs.get("seq_len", 20)
            stride = kwargs.get("stride", 5)
            X, Y = to_sequence_format(traj, seq_len=seq_len, stride=stride)
            if X is None:
                continue
        else:
            raise ValueError(f"未知格式: {fmt}")
        Xs.append(X)
        Ys.append(Y)

    X_all = np.concatenate(Xs, axis=0)
    Y_all = np.concatenate(Ys, axis=0)
    return X_all, Y_all


# ══════════════════════════════════════════════════════════════════════
# ⑨ 主程序：生成完整数据集
# ══════════════════════════════════════════════════════════════════════
#
#  根据资源调整数量：
#    Colab 免费版建议：train=300, val=80, test=80
#    Colab Pro / 本地：train=1000, val=200, test=200
#
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    TRAIN_SIZE = 300
    VAL_SIZE = 80
    TEST_SIZE = 80
    T_END = 5.0  # 每条轨迹时长 [s]
    N_PTS = 500  # 每条轨迹时间点数（dt ≈ 0.01s）
    NOISE = 0.001  # 传感器噪声水平

    print(f"\n开始生成数据集：train={TRAIN_SIZE}, val={VAL_SIZE}, test={TEST_SIZE}")
    print(f"每条轨迹：t_end={T_END}s, n_pts={N_PTS}, noise={NOISE}")
    print(f"坐标系：theta=-90° 为平衡位置（竖直向下）\n")

    train_data = generate_dataset(TRAIN_SIZE, "train", seed=42, t_end=T_END, n_pts=N_PTS, noise_level=NOISE)
    val_data = generate_dataset(VAL_SIZE, "val", seed=123, t_end=T_END, n_pts=N_PTS, noise_level=NOISE)
    test_data = generate_dataset(TEST_SIZE, "test", seed=999, t_end=T_END, n_pts=N_PTS, noise_level=NOISE)

    # ── 转换为各模型格式并打印形状 ──
    print("\n数据格式转换示例：")

    X_tr, Y_tr = build_full_dataset(train_data, fmt="mdann", normalize=True)
    print(f"  MDANN  train: X={X_tr.shape}  Y={Y_tr.shape}")

    X_hnn, Y_hnn = build_full_dataset(train_data, fmt="hnn", normalize=False)
    print(f"  HNN    train: X={X_hnn.shape}  Y={Y_hnn.shape}")

    X_seq, Y_seq = build_full_dataset(train_data, fmt="sequence", seq_len=20, stride=5)
    print(f"  LSTM   train: X={X_seq.shape}  Y={Y_seq.shape}")

    # ── 加载示例 ──
    print(
        """
══════════════════════════════════════════════════
  数据生成完成！文件列表：
    data/flexible_arm_train.pkl
    data/flexible_arm_val.pkl
    data/flexible_arm_test.pkl

  加载示例：
    import pickle
    with open('data/flexible_arm_train.pkl', 'rb') as f:
        data = pickle.load(f)

    traj = data['trajectories'][0]
    # traj 字段：
    #   t         (T,)           时间
    #   theta     (T,)           关节角度 [rad]
    #   dtheta    (T,)           角速度   [rad/s]
    #   ddtheta   (T,)           角加速度 [rad/s²]
    #   q         (2, T)         模态位移 [m]
    #   dq        (2, T)         模态速度 [m/s]
    #   ddq       (2, T)         模态加速度 [m/s²]
    #   KE/PE/E   (T,)           能量     [J]
    #   p_theta   (T,)           广义动量（关节）
    #   p_q       (2, T)         广义动量（柔性）
    #   meta      dict           仿真元信息

    # 归一化统计量：
    stats = data['stats']   # theta_mean/std, q_mean/std ...

    # 转换为训练数组：
    X, Y = build_full_dataset(data, fmt='mdann')  # → ndarray

  坐标系说明：
    theta = 0     → 臂杆水平
    theta = -pi/2 → 竖直向下（稳定平衡点）
══════════════════════════════════════════════════
"""
    )
