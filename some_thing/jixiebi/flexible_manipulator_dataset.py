"""
柔性机械臂动力学建模与数据生成
适用于HNN、LNN、MDANN等深度学习方法
"""

import pickle
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from scipy.linalg import inv, solve

# ==================== 1. 系统参数类 ====================


class FlexibleManipulatorParams:
    """柔性机械臂系统参数"""

    def __init__(self):
        # 几何参数
        self.L = 0.8  # 臂长 [m]
        self.A = 2e-4  # 截面积 [m²]
        self.I_beam = 2e-10  # 截面惯性矩 [m⁴]

        # 材料参数
        self.rho = 2700  # 密度 [kg/m³]
        self.E = 70e9  # 弹性模量 [Pa]

        # hub参数
        self.J_h = 0.01  # hub转动惯量 [kg·m²]

        # 重力
        self.g = 9.81  # 重力加速度 [m/s²]

        # 模态阶数
        self.m = 2  # 柔性模态阶数

        # 数值积分参数
        self.n_quad = 10  # 高斯积分点数
        self.n_modes = self.m

        # 计算模态参数
        self._compute_mode_params()
        self._setup_quadrature()

    def _compute_mode_params(self):
        """计算模态参数 beta_i 和 sigma_i"""
        # beta_i * L 的值（前3阶）
        betaL_values = np.array([1.8751, 4.6941, 7.8548])

        self.betaL = betaL_values[: self.m]
        self.beta = self.betaL / self.L

        # sigma_i
        self.sigma = np.zeros(self.m)
        for i in range(self.m):
            bl = self.betaL[i]
            self.sigma[i] = (np.sinh(bl) - np.sin(bl)) / (np.cosh(bl) + np.cos(bl))

    def _setup_quadrature(self):
        """设置高斯-勒让德积分点和权重"""
        self.quad_x, self.quad_w = leggauss(self.n_quad)
        # 将积分点从[-1,1]映射到[0, L]
        self.x_nodes = (self.quad_x + 1) / 2 * self.L
        self.weights = self.quad_w / 2 * self.L


# ==================== 2. 模态函数类 ====================


class ModeShapeFunctions:
    """模态振型函数及其导数"""

    def __init__(self, params: FlexibleManipulatorParams):
        self.params = params
        self.beta = params.beta
        self.sigma = params.sigma
        self.L = params.L
        self.m = params.m

    def phi(self, i: int, x: np.ndarray) -> np.ndarray:
        """第i阶振型函数值"""
        b = self.beta[i]
        s = self.sigma[i]
        bx = b * x
        return np.cosh(bx) - np.cos(bx) - s * (np.sinh(bx) - np.sin(bx))

    def phi_prime(self, i: int, x: np.ndarray) -> np.ndarray:
        """一阶导数"""
        b = self.beta[i]
        s = self.sigma[i]
        bx = b * x
        return b * (np.sinh(bx) + np.sin(bx) - s * (np.cosh(bx) - np.cos(bx)))

    def phi_double_prime(self, i: int, x: np.ndarray) -> np.ndarray:
        """二阶导数"""
        b = self.beta[i]
        s = self.sigma[i]
        bx = b * x
        return b**2 * (np.cosh(bx) + np.cos(bx) - s * (np.sinh(bx) + np.sin(bx)))

    def compute_modal_integrals(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """计算模态积分矩阵"""
        params = self.params
        x = params.x_nodes
        w = params.weights
        m = params.m

        # 模态质量矩阵 M_ff
        M_ff = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                phi_i = self.phi(i, x)
                phi_j = self.phi(j, x)
                integral = np.sum(w * phi_i * phi_j)
                M_ff[i, j] = params.rho * params.A * integral

        # 刚度矩阵 K_ff
        K_ff = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                phi_ii = self.phi_double_prime(i, x)
                phi_jj = self.phi_double_prime(j, x)
                integral = np.sum(w * phi_ii * phi_jj)
                K_ff[i, j] = params.E * params.I_beam * integral

        # 刚-柔耦合矩阵系数（与y相关，需在线计算）
        # 这里返回常数部分
        return M_ff, K_ff, None, None


# ==================== 3. 动力学系统类 ====================


class FlexibleManipulatorSystem:
    """柔性机械臂动力学系统（DAE）"""

    def __init__(self, params: FlexibleManipulatorParams):
        self.params = params
        self.mode_functions = ModeShapeFunctions(params)

        # 计算常数矩阵
        self.M_ff, self.K_ff, _, _ = self.mode_functions.compute_modal_integrals()
        self.M_ff_inv = inv(self.M_ff)

        # 状态维度
        self.n_r = 1  # 刚体自由度
        self.n_f = params.m  # 柔性自由度
        self.n_state = 2 * (self.n_r + self.n_f)  # 位置+速度

    def get_state_names(self):
        """返回状态变量名称"""
        names = ["theta", "theta_dot"]
        for i in range(self.params.m):
            names.append(f"w{i+1}")
            names.append(f"w{i+1}_dot")
        return names

    def compute_rigid_flex_coupling(self, theta: float, w: np.ndarray) -> Tuple[float, np.ndarray, float]:
        """计算刚-柔耦合项（依赖于当前变形）"""
        params = self.params
        x = params.x_nodes
        weights = params.weights
        m = params.m

        # 计算当前变形场 y(x)
        y = np.zeros_like(x)
        for i in range(m):
            y += w[i] * self.mode_functions.phi(i, x)

        # 刚体惯量增量
        M_rr_flex = params.rho * params.A * np.sum(weights * y**2)
        M_rr = params.J_h + M_rr_flex

        # 刚-柔耦合向量
        M_rf = np.zeros(m)
        for i in range(m):
            phi_i = self.mode_functions.phi(i, x)
            integral = np.sum(weights * y * phi_i)
            M_rf[i] = params.rho * params.A * integral

        return M_rr, M_rf, M_rr_flex

    def compute_coriolis(self, theta_dot: float, w: np.ndarray, w_dot: np.ndarray) -> Tuple[float, np.ndarray]:
        """计算科里奥利力和离心力"""
        params = self.params
        x = params.x_nodes
        weights = params.weights
        m = params.m

        # 计算变形场和速度场
        y = np.zeros_like(x)
        y_dot = np.zeros_like(x)
        for i in range(m):
            phi_i = self.mode_functions.phi(i, x)
            y += w[i] * phi_i
            y_dot += w_dot[i] * phi_i

        # 科里奥利项（简化模型）
        integral_y2 = np.sum(weights * y**2)
        integral_y_ydot = np.sum(weights * y * y_dot)

        C_r = params.rho * params.A * (theta_dot * integral_y2 + 2 * integral_y_ydot)

        C_f = np.zeros(m)
        for i in range(m):
            phi_i = self.mode_functions.phi(i, x)
            integral = np.sum(weights * phi_i * (y_dot - 2 * theta_dot * y))
            C_f[i] = params.rho * params.A * theta_dot * integral

        return C_r, C_f

    def compute_potential(self, theta: float, w: np.ndarray) -> float:
        """计算势能（重力势能 + 弹性势能）"""
        params = self.params
        x = params.x_nodes
        weights = params.weights
        m = params.m

        # 计算变形场
        y = np.zeros_like(x)
        for i in range(m):
            y += w[i] * self.mode_functions.phi(i, x)

        # 重力势能
        V_gravity = params.rho * params.A * params.g * (np.sum(weights * x) * np.sin(theta) + np.sum(weights * y) * np.cos(theta))

        # 弹性势能
        V_elastic = 0.5 * np.dot(w, np.dot(self.K_ff, w))

        return V_gravity + V_elastic

    def compute_kinetic(self, theta_dot: float, w: np.ndarray, w_dot: np.ndarray, M_rr: float, M_rf: np.ndarray) -> float:
        """计算动能"""
        M_ff = self.M_ff
        T = 0.5 * M_rr * theta_dot**2
        T += theta_dot * np.dot(M_rf, w_dot)
        T += 0.5 * np.dot(w_dot, np.dot(M_ff, w_dot))
        return T

    def dynamics(self, t: float, state: np.ndarray, tau: float) -> np.ndarray:
        """
        动力学方程（DAE）
        state = [theta, theta_dot, w1, w1_dot, w2, w2_dot, ...]
        """
        n_r = self.n_r
        n_f = self.n_f

        # 解析状态
        theta = state[0]
        theta_dot = state[1]
        w = state[2 : 2 + n_f]
        w_dot = state[2 + n_f : 2 + 2 * n_f]

        # 计算耦合项
        M_rr, M_rf, _ = self.compute_rigid_flex_coupling(theta, w)
        C_r, C_f = self.compute_coriolis(theta_dot, w, w_dot)

        # 装配质量矩阵
        M = np.zeros((n_r + n_f, n_r + n_f))
        M[0, 0] = M_rr
        M[0, 1 : n_r + n_f] = M_rf
        M[1 : n_r + n_f, 0] = M_rf
        M[1 : n_r + n_f, 1 : n_r + n_f] = self.M_ff

        # 装配力向量
        F = np.zeros(n_r + n_f)
        F[0] = tau - C_r
        F[1 : n_r + n_f] = -C_f - np.dot(self.K_ff, w)

        # 求解加速度
        acc = solve(M, F)

        theta_ddot = acc[0]
        w_ddot = acc[1 : n_r + n_f]

        # 返回状态导数
        state_dot = np.zeros_like(state)
        state_dot[0] = theta_dot
        state_dot[1] = theta_ddot
        state_dot[2 : 2 + n_f] = w_dot
        state_dot[2 + n_f : 2 + 2 * n_f] = w_ddot

        return state_dot


# ==================== 4. 数值积分器 ====================


class MultiStepBlockSolver:
    """多步块谱方法求解器"""

    def __init__(self, system: FlexibleManipulatorSystem, dt: float = 0.001, r: int = 4):
        self.system = system
        self.dt = dt
        self.r = r  # 块内节点数

        # 计算Chebyshev节点和微分矩阵
        self._setup_chebyshev()

    def _setup_chebyshev(self):
        """设置Chebyshev节点"""
        self.tau = np.cos((2 * np.arange(1, self.r + 1) - 1) / (2 * self.r) * np.pi)
        # 映射到[0, dt]
        self.t = (self.tau + 1) / 2 * self.dt

    def step(self, state_n: np.ndarray, tau_func: Callable) -> np.ndarray:
        """
        单步积分（使用经典RK4作为基础积分器）
        完整多步块方法需要实现隐式格式
        """

        # 简化版本：使用RK4
        def f(t, state):
            return self.system.dynamics(t, state, tau_func(t))

        t = 0
        h = self.dt
        state = state_n.copy()

        k1 = f(t, state)
        k2 = f(t + h / 2, state + h / 2 * k1)
        k3 = f(t + h / 2, state + h / 2 * k2)
        k4 = f(t + h, state + h * k3)

        state_next = state + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return state_next


# ==================== 5. 数据生成器 ====================


class DataGenerator:
    """训练数据生成器"""

    def __init__(self, system: FlexibleManipulatorSystem, solver: MultiStepBlockSolver):
        self.system = system
        self.solver = solver

    def generate_trajectory(
        self, init_theta: float, init_theta_dot: float = 0.0, tau_func: Callable = None, t_span: Tuple[float, float] = (0, 10), save_interval: int = 10
    ) -> dict:
        """
        生成单条轨迹

        参数:
            init_theta: 初始角度 [rad]
            init_theta_dot: 初始角速度 [rad/s]
            tau_func: 驱动力矩函数 tau(t)
            t_span: 时间区间 (t0, t1)
            save_interval: 保存间隔（步数）
        """
        if tau_func is None:
            tau_func = lambda t: 0.0

        params = self.system.params
        m = params.m
        n_state = self.system.n_state

        # 初始状态
        state0 = np.zeros(n_state)
        state0[0] = init_theta
        state0[1] = init_theta_dot
        # w, w_dot 初始为0

        # 时间步数
        dt = self.solver.dt
        n_steps = int((t_span[1] - t_span[0]) / dt)

        # 存储轨迹
        times = []
        states = []
        torques = []

        state = state0.copy()
        for step in range(n_steps + 1):
            if step % save_interval == 0:
                times.append(step * dt)
                states.append(state.copy())
                torques.append(tau_func(step * dt))

            state = self.solver.step(state, tau_func)

        # 提取变量
        theta = [s[0] for s in states]
        theta_dot = [s[1] for s in states]
        w = [s[2 : 2 + m] for s in states]
        w_dot = [s[2 + m : 2 + 2 * m] for s in states]

        # 计算动量和能量（可选）
        momenta = []
        energies = []
        for i, s in enumerate(states):
            theta_i = s[0]
            theta_dot_i = s[1]
            w_i = s[2 : 2 + m]
            w_dot_i = s[2 + m : 2 + 2 * m]

            # 计算耦合项
            M_rr, M_rf, _ = self.system.compute_rigid_flex_coupling(theta_i, w_i)

            # 动量
            p_theta = M_rr * theta_dot_i + np.dot(M_rf, w_dot_i)
            p_w = np.dot(M_rf, theta_dot_i) + np.dot(self.system.M_ff, w_dot_i)
            momenta.append([p_theta] + list(p_w))

            # 能量
            kinetic = self.system.compute_kinetic(theta_dot_i, w_i, w_dot_i, M_rr, M_rf)
            potential = self.system.compute_potential(theta_i, w_i)
            energies.append(kinetic + potential)

        return {
            "t": np.array(times),
            "theta": np.array(theta),
            "theta_dot": np.array(theta_dot),
            "w": np.array(w),
            "w_dot": np.array(w_dot),
            "p": np.array(momenta),
            "energy": np.array(energies),
            "torque": np.array(torques),
        }

    def generate_dataset(self, n_trajectories: int = 800, t_span=(0, 10), tau_type: str = "free", save_path: str = None) -> dict:
        """
        生成完整数据集

        参数:
            n_trajectories: 轨迹数量
            t_span: 时间区间
            tau_type: 力矩类型 ('free', 'sine', 'chirp')
            save_path: 保存路径
        """
        trajectories = []
        init_angles = np.linspace(0.1, 0.8, n_trajectories)

        if tau_type == "free":
            tau_func = lambda t: 0.0
        elif tau_type == "sine":
            A = 0.5
            f = 2.0
            tau_func = lambda t: A * np.sin(2 * np.pi * f * t)
        elif tau_type == "chirp":
            A = 0.5
            f0 = 0.5
            f1 = 20.0
            T = t_span[1]
            tau_func = lambda t: A * np.sin(2 * np.pi * (f0 + (f1 - f0) * t / T) * t)
        else:
            tau_func = lambda t: 0.0

        for i, init_theta in enumerate(init_angles):
            print(f"生成轨迹 {i+1}/{n_trajectories}: theta0={init_theta:.3f} rad")
            traj = self.generate_trajectory(init_theta, tau_func=tau_func, t_span=t_span)
            trajectories.append(traj)

        dataset = {
            "trajectories": trajectories,
            "params": {
                "dt": self.solver.dt,
                "t_span": t_span,
                "tau_type": tau_type,
                "n_trajectories": n_trajectories,
                "system_params": {
                    "L": self.system.params.L,
                    "m": self.system.params.m,
                    "E": self.system.params.E,
                    "rho": self.system.params.rho,
                },
            },
        }

        if save_path:
            with open(save_path, "wb") as f:
                pickle.dump(dataset, f)
            print(f"数据集已保存至: {save_path}")

        return dataset


# ==================== 6. 数据预处理 ====================


class DataPreprocessor:
    """数据预处理（归一化、格式转换）"""

    @staticmethod
    def normalize(data: np.ndarray, mean: float = None, std: float = None):
        """归一化"""
        if mean is None:
            mean = np.mean(data)
            std = np.std(data)
        normalized = (data - mean) / std
        return normalized, mean, std

    @staticmethod
    def to_hnn_format(traj: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        转换为HNN格式
        输入: [θ, p_θ, w₁, p_w1, w₂, p_w2, ...]
        输出: [θ̇, ṗ_θ, ẇ₁, ṗ_w1, ẇ₂, ṗ_w2, ...]
        """
        m = traj["w"].shape[1]

        # 输入: 位置和动量
        X = np.hstack([traj["theta"].reshape(-1, 1), traj["p"][:, 0:1], traj["w"], traj["p"][:, 1:]])

        # 输出: 速度和动量导数（需要数值微分）
        dt = traj["t"][1] - traj["t"][0]
        theta_dot = traj["theta_dot"]
        w_dot = traj["w_dot"]
        p_dot = np.gradient(traj["p"], dt, axis=0)

        Y = np.hstack([theta_dot.reshape(-1, 1), p_dot[:, 0:1], w_dot, p_dot[:, 1:]])

        return X, Y

    @staticmethod
    def to_lnn_format(traj: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        转换为LNN格式
        输入: [θ, θ̇, w₁, ẇ₁, w₂, ẇ₂, ...]
        输出: [θ̈, ẅ₁, ẅ₂, ...]
        """
        m = traj["w"].shape[1]

        # 输入: 位置和速度
        X = np.hstack([traj["theta"].reshape(-1, 1), traj["theta_dot"].reshape(-1, 1), traj["w"], traj["w_dot"]])

        # 输出: 加速度（数值微分）
        dt = traj["t"][1] - traj["t"][0]
        theta_ddot = np.gradient(traj["theta_dot"], dt)
        w_ddot = np.gradient(traj["w_dot"], dt, axis=0)

        Y = np.hstack([theta_ddot.reshape(-1, 1), w_ddot])

        return X, Y

    @staticmethod
    def to_mdann_format(traj: dict, slow_modes: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        转换为MDANN格式（多尺度）
        慢变量: [θ, θ̇, w₁, w₂]（前2阶模态）
        快变量: [w₃, w₄, ẇ₃, ẇ₄]（高阶模态）
        """
        m = traj["w"].shape[1]

        # 慢变量（前2阶模态）
        y = np.hstack([traj["theta"].reshape(-1, 1), traj["theta_dot"].reshape(-1, 1), traj["w"][:, :slow_modes]])

        # 快变量（高阶模态）
        if m > slow_modes:
            z = np.hstack([traj["w"][:, slow_modes:], traj["w_dot"][:, slow_modes:]])
        else:
            z = np.zeros((len(traj["t"]), 1))

        # 输出加速度
        dt = traj["t"][1] - traj["t"][0]
        theta_ddot = np.gradient(traj["theta_dot"], dt)
        w_ddot = np.gradient(traj["w_dot"], dt, axis=0)

        y_dot = np.hstack([theta_ddot.reshape(-1, 1), w_ddot[:, :slow_modes]])
        z_dot = w_ddot[:, slow_modes:] if m > slow_modes else np.zeros((len(traj["t"]), 1))

        return (y, z), (y_dot, z_dot)


# ==================== 7. 可视化 ====================


class Visualizer:
    """结果可视化"""

    @staticmethod
    def plot_trajectory(traj: dict, save_path: str = None):
        """绘制单条轨迹"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        t = traj["t"]

        # 刚体角度
        axes[0, 0].plot(t, traj["theta"])
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("θ (rad)")
        axes[0, 0].set_title("Joint Angle")
        axes[0, 0].grid(True)

        # 刚体角速度
        axes[0, 1].plot(t, traj["theta_dot"])
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].set_ylabel("θ̇ (rad/s)")
        axes[0, 1].set_title("Joint Angular Velocity")
        axes[0, 1].grid(True)

        # 柔性模态
        m = traj["w"].shape[1]
        for i in range(m):
            axes[1, 0].plot(t, traj["w"][:, i], label=f"w{i+1}")
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_ylabel("Modal Displacement")
        axes[1, 0].set_title("Flexible Modes")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 能量
        axes[1, 1].plot(t, traj["energy"])
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("Energy (J)")
        axes[1, 1].set_title("Total Energy")
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    @staticmethod
    def plot_comparison(traj_pred: dict, traj_true: dict, save_path: str = None):
        """对比预测轨迹与真实轨迹"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        t_true = traj_true["t"]
        t_pred = traj_pred["t"]

        # 角度对比
        axes[0, 0].plot(t_true, traj_true["theta"], "b-", label="True", linewidth=2)
        axes[0, 0].plot(t_pred, traj_pred["theta"], "r--", label="Predicted", linewidth=2)
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("θ (rad)")
        axes[0, 0].set_title("Joint Angle")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 模态对比
        m = traj_true["w"].shape[1]
        axes[0, 1].plot(t_true, traj_true["w"][:, 0], "b-", label="True", linewidth=2)
        axes[0, 1].plot(t_pred, traj_pred["w"][:, 0], "r--", label="Predicted", linewidth=2)
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].set_ylabel("w₁")
        axes[0, 1].set_title("First Flexible Mode")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 误差
        # 插值到相同时间点
        from scipy.interpolate import interp1d

        theta_interp = interp1d(t_pred, traj_pred["theta"], fill_value="extrapolate")
        theta_pred_interp = theta_interp(t_true)
        error = traj_true["theta"] - theta_pred_interp

        axes[1, 0].plot(t_true, error)
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_ylabel("θ Error (rad)")
        axes[1, 0].set_title("Prediction Error")
        axes[1, 0].grid(True)

        # 能量对比
        axes[1, 1].plot(t_true, traj_true["energy"], "b-", label="True", linewidth=2)
        axes[1, 1].plot(t_pred, traj_pred["energy"], "r--", label="Predicted", linewidth=2)
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("Energy (J)")
        axes[1, 1].set_title("Total Energy")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()


# ==================== 8. 主程序 ====================


def main():
    """主程序：生成数据集并可视化"""

    print("=" * 60)
    print("柔性机械臂动力学建模与数据生成")
    print("=" * 60)

    # 1. 初始化系统
    print("\n[1] 初始化系统参数...")
    params = FlexibleManipulatorParams()
    system = FlexibleManipulatorSystem(params)
    print(f"    自由度: 刚体={system.n_r}, 柔性={system.n_f}")
    print(f"    状态维度: {system.n_state}")

    # 2. 初始化求解器
    print("\n[2] 初始化数值求解器...")
    solver = MultiStepBlockSolver(system, dt=0.001, r=4)
    print(f"    时间步长: {solver.dt} s")
    print(f"    块节点数: {solver.r}")

    # 3. 生成单条轨迹测试
    print("\n[3] 生成单条测试轨迹...")
    generator = DataGenerator(system, solver)
    traj = generator.generate_trajectory(init_theta=0.5, t_span=(0, 5))
    print(f"    轨迹点数: {len(traj['t'])}")
    print(f"    最终能量: {traj['energy'][-1]:.4f} J")
    print(f"    能量变化: {(traj['energy'][-1] - traj['energy'][0])/traj['energy'][0]*100:.2f}%")

    # 4. 可视化
    print("\n[4] 可视化轨迹...")
    visualizer = Visualizer()
    visualizer.plot_trajectory(traj, save_path="flexible_manipulator_trajectory.png")

    # 5. 生成完整数据集
    print("\n[5] 生成完整数据集...")
    dataset = generator.generate_dataset(n_trajectories=100, t_span=(0, 5), tau_type="free", save_path="flexible_manipulator_dataset.pkl")  # 测试用，实际可设为800

    # 6. 数据格式转换示例
    print("\n[6] 数据格式转换示例...")
    X_hnn, Y_hnn = DataPreprocessor.to_hnn_format(traj)
    X_lnn, Y_lnn = DataPreprocessor.to_lnn_format(traj)
    print(f"    HNN格式: X.shape={X_hnn.shape}, Y.shape={Y_hnn.shape}")
    print(f"    LNN格式: X.shape={X_lnn.shape}, Y.shape={Y_lnn.shape}")

    print("\n" + "=" * 60)
    print("数据生成完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
