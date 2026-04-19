"""
柔性机械臂训练数据生成器
基于悬臂梁假设模态法的动力学仿真
输出：可用于MDANN/HNN/LNN训练的轨迹数据
"""

import os
import pickle

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import roots_legendre

# ========================== 1. 系统参数 ==========================


class FlexArmParams:
    """柔性机械臂系统参数"""

    def __init__(self):
        # 几何参数
        self.L = 1.0
        self.A = 0.0001
        self.I = (0.01**4) / 12

        # 材料参数
        self.rho = 2700
        self.E = 70e9

        # 运动参数
        self.g = 9.81
        self.J_h = 0.1
        self.b = 0.01

        # 模态参数
        self.n_modes = 2
        self._compute_mode_shapes()
        self._setup_quadrature()
        self._compute_constant_matrices()

    def _compute_mode_shapes(self):
        """计算悬臂梁模态参数"""
        betaL = np.array([1.87510407, 4.69409113])
        self.beta = betaL / self.L
        self.sigma = (np.sinh(betaL) + np.sin(betaL)) / (np.cosh(betaL) + np.cos(betaL))

    def _setup_quadrature(self):
        """设置高斯-勒让德积分"""
        n_quad = 10
        xi, w = roots_legendre(n_quad)
        self.x_quad = (self.L / 2) * (xi + 1)
        self.w_quad = (self.L / 2) * w

    def phi(self, x, i):
        """第i阶振型函数"""
        b = self.beta[i]
        s = self.sigma[i]
        return np.cosh(b * x) - np.cos(b * x) - s * (np.sinh(b * x) - np.sin(b * x))

    def d2phi(self, x, i):
        """振型函数二阶导数"""
        b = self.beta[i]
        s = self.sigma[i]
        return (b**2) * (np.cosh(b * x) + np.cos(b * x) - s * (np.sinh(b * x) + np.sin(b * x)))

    def _compute_constant_matrices(self):
        """计算常数矩阵 Mf, Kf"""
        self.Mf = np.zeros((self.n_modes, self.n_modes))
        self.Kf = np.zeros((self.n_modes, self.n_modes))

        for i in range(self.n_modes):
            for j in range(self.n_modes):
                phi_i = self.phi(self.x_quad, i)
                phi_j = self.phi(self.x_quad, j)
                self.Mf[i, j] = np.dot(self.w_quad, self.rho * self.A * phi_i * phi_j)

                d2phi_i = self.d2phi(self.x_quad, i)
                d2phi_j = self.d2phi(self.x_quad, j)
                self.Kf[i, j] = np.dot(self.w_quad, self.E * self.I * d2phi_i * d2phi_j)


# ========================== 2. 动力学系统 ==========================


class FlexArmDynamics:
    """柔性机械臂动力学系统"""

    def __init__(self, params):
        self.p = params

    def dynamics(self, t, state, torque_func):
        """
        动力学方程
        state = [θ, θ̇, w₁, w₂, ẇ₁, ẇ₂]
        """
        p = self.p

        theta = state[0]
        dtheta = state[1]
        q = state[2 : 2 + p.n_modes]
        dq = state[2 + p.n_modes :]

        x = p.x_quad
        w = p.w_quad
        phi_mat = np.array([p.phi(x, k) for k in range(p.n_modes)])

        # 计算变形场
        delta = q @ phi_mat

        # 刚体惯量（含柔性耦合）
        Mr = p.J_h + p.rho * p.A * np.dot(w, x**2)

        # 刚-柔耦合向量
        M_couple = np.array([p.rho * p.A * np.dot(w, x * p.phi(x, k)) for k in range(p.n_modes)])

        # 装配质量矩阵
        M = np.eye(1 + p.n_modes)
        M[0, 0] = Mr
        M[0, 1:] = M_couple
        M[1:, 0] = M_couple
        M[1:, 1:] = p.Mf

        # 装配力向量
        F = np.zeros(1 + p.n_modes)

        # 重力力矩
        delta_com = np.dot(w, delta) / p.L
        F[0] -= p.rho * p.A * p.L * p.g * (p.L / 2 + delta_com) * np.sin(theta)

        # 弹性力
        F[1:] -= p.Kf @ q

        # 阻尼力
        F[0] -= p.b * dtheta

        # 离心力
        for i in range(p.n_modes):
            b_val = p.beta[i]
            s = p.sigma[i]
            dphi = b_val * (np.sinh(b_val * x) - np.sin(b_val * x) - s * (np.cosh(b_val * x) - np.cos(b_val * x)))
            cent = np.dot(w, p.rho * p.A * dphi * x**2 * dtheta**2)
            F[1 + i] += 0.5 * cent

        # 外力矩
        F[0] += torque_func(t)

        # 求解加速度
        accel = np.linalg.solve(M, F)

        # 状态导数
        dstate = np.zeros(2 + 2 * p.n_modes)
        dstate[0] = dtheta
        dstate[1] = accel[0]
        dstate[2 : 2 + p.n_modes] = dq
        dstate[2 + p.n_modes :] = accel[1:]

        return dstate


# ========================== 3. 单条轨迹生成 ==========================


def generate_single_trajectory(params, init_theta_deg=0.0, init_q=None, torque_type="free", t_span=(0, 20), dt=0.05, noise_level=0.0):
    """
    生成单条轨迹

    参数:
        params: 系统参数
        init_theta_deg: 初始角度（度）
        init_q: 初始柔性模态位移
        torque_type: 力矩类型 ('free', 'const', 'sin')
        t_span: 时间区间 (start, end)
        dt: 采样间隔
        noise_level: 噪声水平

    返回:
        轨迹数据字典
    """
    dynamics = FlexArmDynamics(params)

    # 初始状态
    y0 = np.zeros(2 + 2 * params.n_modes)
    y0[0] = np.deg2rad(init_theta_deg)

    if init_q is None:
        init_q = np.array([0.001, 0.0005])
    y0[2 : 2 + params.n_modes] = init_q

    # 力矩函数
    if torque_type == "free":
        torque = lambda t: 0.0
    elif torque_type == "const":
        torque = lambda t: 2.0
    elif torque_type == "sin":
        torque = lambda t: 1.5 * np.sin(np.pi * t)
    else:
        torque = lambda t: 0.0

    # 时间点
    t_eval = np.arange(t_span[0], t_span[1], dt)

    # 求解ODE
    sol = solve_ivp(lambda t, y: dynamics.dynamics(t, y, torque), t_span, y0, t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-8)

    t = sol.t
    y = sol.y

    # 提取数据
    theta = y[0]  # 关节角度 (rad)
    theta_dot = y[1]  # 关节角速度 (rad/s)
    q = y[2 : 2 + params.n_modes].T  # 柔性模态位移
    q_dot = y[2 + params.n_modes :].T  # 柔性模态速度

    # 计算加速度（数值微分）
    dt_actual = t[1] - t[0]
    theta_ddot = np.gradient(theta_dot, dt_actual)
    q_ddot = np.gradient(q_dot, dt_actual, axis=0)

    # 计算动能
    KE_rigid = 0.5 * params.J_h * theta_dot**2

    # 计算势能
    PE_elastic = 0.5 * np.sum(q * (params.Kf @ q.T).T, axis=1)

    # 总能量
    energy = KE_rigid + PE_elastic

    # 添加噪声
    if noise_level > 0:
        theta += np.random.normal(0, noise_level, len(theta))
        theta_dot += np.random.normal(0, noise_level, len(theta_dot))
        q += np.random.normal(0, noise_level, q.shape)
        q_dot += np.random.normal(0, noise_level, q_dot.shape)
        energy += np.random.normal(0, noise_level, len(energy))

    # 返回轨迹
    return {
        "t": t,
        "theta": theta,
        "theta_dot": theta_dot,
        "theta_ddot": theta_ddot,
        "q": q,
        "q_dot": q_dot,
        "q_ddot": q_ddot,
        "energy": energy,
        "params": {"init_theta_deg": init_theta_deg, "torque_type": torque_type, "noise_level": noise_level},
    }


# ========================== 4. 批量数据生成 ==========================


def generate_dataset(params, n_trajectories=800, t_span=(0, 2), dt=0.05, torque_type="free", init_theta_range=(-45, 45), noise_level=0.0, save_path=None):
    """
    生成完整数据集

    参数:
        params: 系统参数
        n_trajectories: 轨迹数量
        t_span: 时间区间
        dt: 采样间隔
        torque_type: 力矩类型
        init_theta_range: 初始角度范围（度）
        noise_level: 噪声水平
        save_path: 保存路径
    """
    trajectories = []

    print(f"生成 {n_trajectories} 条轨迹 (力矩类型: {torque_type})...")

    for i in range(n_trajectories):
        # 随机初始条件
        init_theta = np.random.uniform(init_theta_range[0], init_theta_range[1])
        init_q = np.random.uniform(-0.01, 0.01, params.n_modes)

        # 生成轨迹
        traj = generate_single_trajectory(params, init_theta, init_q, torque_type, t_span, dt, noise_level)
        trajectories.append(traj)

        if (i + 1) % 100 == 0:
            print(f"  已生成 {i+1}/{n_trajectories} 条轨迹")

    # 计算统计信息
    all_theta = np.concatenate([t["theta"] for t in trajectories])
    all_theta_dot = np.concatenate([t["theta_dot"] for t in trajectories])
    all_q = np.concatenate([t["q"] for t in trajectories])

    stats = {
        "theta_mean": np.mean(all_theta),
        "theta_std": np.std(all_theta),
        "theta_dot_mean": np.mean(all_theta_dot),
        "theta_dot_std": np.std(all_theta_dot),
        "q_mean": np.mean(all_q),
        "q_std": np.std(all_q),
    }

    dataset = {
        "trajectories": trajectories,
        "stats": stats,
        "params": {
            "n_trajectories": n_trajectories,
            "t_span": t_span,
            "dt": dt,
            "torque_type": torque_type,
            "noise_level": noise_level,
            "system": {"L": params.L, "n_modes": params.n_modes, "J_h": params.J_h, "damping": params.b},
        },
    }

    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(dataset, f)
        print(f"数据集已保存: {save_path}")

    return dataset


# ========================== 5. 数据格式转换 ==========================


def convert_to_mdann_format(traj, stats=None):
    """
    转换为MDANN格式

    MDANN输入: 慢变量 [θ, θ̇, q₁, q₂]
    MDANN输出: 慢变量加速度 [θ̈, q̈₁, q̈₂]
    """
    m = traj["q"].shape[1]

    # 输入
    X = np.column_stack([traj["theta"], traj["theta_dot"], traj["q"]])

    # 输出
    Y = np.column_stack([traj["theta_ddot"], traj["q_ddot"]])

    # 归一化
    if stats is not None:
        X = (X - stats["X_mean"]) / stats["X_std"]
        Y = (Y - stats["Y_mean"]) / stats["Y_std"]

    return X, Y


def convert_to_hnn_format(traj, params, stats=None):
    """
    转换为HNN格式

    HNN输入: [θ, p_θ, q₁, p_q1, q₂, p_q2]
    HNN输出: [θ̇, ṗ_θ, q̇₁, ṗ_q1, q̇₂, ṗ_q2]
    """
    m = traj["q"].shape[1]

    # 计算动量
    p_theta = params.J_h * traj["theta_dot"]
    p_q = params.Mf @ traj["q_dot"].T
    p_q = p_q.T

    # 输入
    X = np.column_stack([traj["theta"], p_theta, traj["q"], p_q])

    # 输出
    dt = traj["t"][1] - traj["t"][0]
    p_theta_dot = np.gradient(p_theta, dt)
    p_q_dot = np.gradient(p_q, dt, axis=0)

    Y = np.column_stack([traj["theta_dot"], p_theta_dot, traj["q_dot"], p_q_dot])

    if stats is not None:
        X = (X - stats["X_mean"]) / stats["X_std"]
        Y = (Y - stats["Y_mean"]) / stats["Y_std"]

    return X, Y


def convert_to_lnn_format(traj, stats=None):
    """
    转换为LNN格式

    LNN输入: [θ, θ̇, q₁, q̇₁, q₂, q̇₂]
    LNN输出: [θ̈, q̈₁, q̈₂]
    """
    m = traj["q"].shape[1]

    # 输入
    X = np.column_stack([traj["theta"], traj["theta_dot"], traj["q"], traj["q_dot"]])

    # 输出
    Y = np.column_stack([traj["theta_ddot"], traj["q_ddot"]])

    if stats is not None:
        X = (X - stats["X_mean"]) / stats["X_std"]
        Y = (Y - stats["Y_mean"]) / stats["Y_std"]

    return X, Y


# ========================== 6. 可视化 ==========================


def plot_trajectory(traj, save_path=None):
    """可视化单条轨迹"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    t = traj["t"]

    # 关节角度
    axes[0, 0].plot(t, np.rad2deg(traj["theta"]), "b-", linewidth=2)
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Angle (deg)")
    axes[0, 0].set_title("Joint Angle")
    axes[0, 0].grid(True)

    # 关节角速度
    axes[0, 1].plot(t, traj["theta_dot"], "r-", linewidth=2)
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Angular Velocity (rad/s)")
    axes[0, 1].set_title("Joint Angular Velocity")
    axes[0, 1].grid(True)

    # 柔性模态
    for i in range(traj["q"].shape[1]):
        axes[1, 0].plot(t, traj["q"][:, i], label=f"q_{i+1}", linewidth=2)
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Modal Displacement")
    axes[1, 0].set_title("Flexible Modes")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 能量
    axes[1, 1].plot(t, traj["energy"], "g-", linewidth=2)
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Energy (J)")
    axes[1, 1].set_title("Total Energy")
    axes[1, 1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"图已保存: {save_path}")

    plt.show()


# ========================== 7. 主程序 ==========================


def main():
    """生成训练、验证、测试数据集"""

    print("=" * 60)
    print("柔性机械臂训练数据生成器")
    print("=" * 60)

    # 初始化系统
    params = FlexArmParams()
    print(f"\n系统参数:")
    print(f"  臂长: {params.L} m")
    print(f"  模态阶数: {params.n_modes}")
    print(f"  hub惯量: {params.J_h} kg·m²")

    # 创建数据目录
    os.makedirs("data", exist_ok=True)

    # 生成训练集（自由摆动）
    print("\n[1/3] 生成训练集...")
    train_data = generate_dataset(params, n_trajectories=3, torque_type="free", init_theta_range=(-60, 60), save_path="data/flex_train.pkl")

    # 生成验证集（自由摆动，不同初始条件）
    print("\n[2/3] 生成验证集...")
    val_data = generate_dataset(params, n_trajectories=2, torque_type="free", init_theta_range=(-60, 60), save_path="data/flex_val.pkl")

    # 生成测试集（恒定力矩）
    print("\n[3/3] 生成测试集...")
    test_data = generate_dataset(params, n_trajectories=2, torque_type="const", init_theta_range=(-30, 30), save_path="data/flex_test.pkl")

    # 可视化示例轨迹
    print("\n[4/4] 可视化示例轨迹...")
    plot_trajectory(train_data["trajectories"][0], save_path="data/sample_trajectory.png")

    # 打印统计信息
    print(f"\n数据集统计:")
    print(f"  训练集: {len(train_data['trajectories'])} 条轨迹")
    print(f"  验证集: {len(val_data['trajectories'])} 条轨迹")
    print(f"  测试集: {len(test_data['trajectories'])} 条轨迹")
    print(f"  每条轨迹时间点数: {len(train_data['trajectories'][0]['t'])}")

    print("\n" + "=" * 60)
    print("数据生成完成！")
    print("生成的文件:")
    print("  - data/flex_train.pkl")
    print("  - data/flex_val.pkl")
    print("  - data/flex_test.pkl")
    print("  - data/sample_trajectory.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
