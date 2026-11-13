# -*- coding: utf-8 -*-
"""
core/lie_algebra.py
===================
李群与李代数基础数学库 (SO(3), SE(3), Sim(3))
严格 1:1 对齐 C++ Eigen3 + Sophus / ORB-SLAM3 (g2o::SE3Quat) 数学定义。

约定说明 (与 ORB-SLAM3 / g2o::VertexSE3Expmap 保持一致):
- 6DoF 李代数切空间向量 xi = [omega_x, omega_y, omega_z, rho_x, rho_y, rho_z]^T
  其中前 3 维为旋转分量 omega in so(3)，后 3 维为平移分量 rho in R^3。
- 位姿左乘扰动更新模型: T_new = SE3.exp(d_xi) * T_old
"""

from __future__ import annotations
import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """
    向量反对称矩阵算子 [v]_x (对应 C++ Sophus::SO3d::hat)
    输入: v (3,)
    输出: (3, 3) 反对称矩阵
    """
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    return np.array([
        [0.0, -vz,  vy],
        [ vz, 0.0, -vx],
        [-vy,  vx, 0.0]
    ], dtype=np.float64)


def vee(M: np.ndarray) -> np.ndarray:
    """
    反对称矩阵转向量算子 (对应 C++ Sophus::SO3d::vee)
    """
    return np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]], dtype=np.float64) * 0.5


def exp_so3(omega: np.ndarray) -> np.ndarray:
    """
    SO(3) 指数映射: exp([omega]_x) -> R in SO(3) (罗德里格斯公式)
    对应 C++ Sophus::SO3d::exp(omega)
    """
    omega = np.asarray(omega, dtype=np.float64).reshape(3)
    theta_sq = float(np.dot(omega, omega))
    theta = np.sqrt(theta_sq)
    W = skew(omega)
    if theta < 1e-8:
        # 一阶与二阶泰勒展开: I + W + 0.5 * W^2
        return np.eye(3, dtype=np.float64) + W + 0.5 * (W @ W)
    else:
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / theta_sq
        return np.eye(3, dtype=np.float64) + A * W + B * (W @ W)


def log_so3(R: np.ndarray) -> np.ndarray:
    """
    SO(3) 对数映射: log(R) -> omega in R^3
    对应 C++ Sophus::SO3d::log()
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(R))
    cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return vee(R - np.eye(3, dtype=np.float64))
    elif np.pi - theta < 1e-5:
        # 接近 180 度奇异区处理
        eigvals, eigvecs = np.linalg.eig(R)
        idx = np.argmin(np.abs(eigvals - 1.0))
        axis = np.real(eigvecs[:, idx])
        axis /= (np.linalg.norm(axis) + 1e-12)
        if np.dot(axis, vee(R)) < 0:
            axis = -axis
        return axis * theta
    else:
        return (theta / np.sin(theta)) * vee(R)


def right_jacobian_so3(omega: np.ndarray) -> np.ndarray:
    """
    SO(3) 右雅可比矩阵 J_r(omega)
    满足: exp(omega + d_omega) approx exp(omega) * exp(J_r(omega) * d_omega)
    用于 IMU 预积分 (Forster 模型) 零偏一阶更新与误差传播。
    """
    omega = np.asarray(omega, dtype=np.float64).reshape(3)
    theta_sq = float(np.dot(omega, omega))
    theta = np.sqrt(theta_sq)
    W = skew(omega)
    I3 = np.eye(3, dtype=np.float64)
    if theta < 1e-8:
        return I3 - 0.5 * W + (1.0 / 6.0) * (W @ W)
    else:
        c1 = (1.0 - np.cos(theta)) / theta_sq
        c2 = (theta - np.sin(theta)) / (theta_sq * theta)
        return I3 - c1 * W + c2 * (W @ W)


def right_jacobian_inv_so3(omega: np.ndarray) -> np.ndarray:
    """
    SO(3) 右雅可比逆矩阵 J_r^{-1}(omega)
    """
    omega = np.asarray(omega, dtype=np.float64).reshape(3)
    theta_sq = float(np.dot(omega, omega))
    theta = np.sqrt(theta_sq)
    W = skew(omega)
    I3 = np.eye(3, dtype=np.float64)
    if theta < 1e-8:
        return I3 + 0.5 * W + (1.0 / 12.0) * (W @ W)
    else:
        half_theta = 0.5 * theta
        cot_half = np.cos(half_theta) / np.sin(half_theta)
        c2 = (1.0 / theta_sq) * (1.0 - half_theta * cot_half)
        return I3 + 0.5 * W + c2 * (W @ W)


class SE3:
    """
    刚体变换群 SE(3) 类 (对应 C++ Sophus::SE3d / g2o::SE3Quat)
    内部维护旋转矩阵 R (3x3) 与平移向量 t (3,)
    表示坐标变换: P_target = R * P_source + t
    """
    __slots__ = ("R", "t")

    def __init__(self, R: np.ndarray | None = None, t: np.ndarray | None = None):
        if R is None:
            self.R = np.eye(3, dtype=np.float64)
        else:
            self.R = np.asarray(R, dtype=np.float64).reshape(3, 3).copy()
        if t is None:
            self.t = np.zeros(3, dtype=np.float64)
        else:
            self.t = np.asarray(t, dtype=np.float64).reshape(3).copy()

    @classmethod
    def identity(cls) -> SE3:
        return cls()

    @classmethod
    def from_matrix(cls, T: np.ndarray) -> SE3:
        T = np.asarray(T, dtype=np.float64).reshape(4, 4)
        return cls(T[:3, :3], T[:3, 3])

    def matrix(self) -> np.ndarray:
        """返回 4x4 齐次变换矩阵"""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    def inverse(self) -> SE3:
        """位姿求逆: T^{-1} = [R^T, -R^T * t]"""
        R_inv = self.R.T
        t_inv = -(R_inv @ self.t)
        return SE3(R_inv, t_inv)

    def copy(self) -> SE3:
        return SE3(self.R.copy(), self.t.copy())

    def __mul__(self, other: SE3) -> SE3:
        """群乘法: T1 * T2"""
        return SE3(self.R @ other.R, self.R @ other.t + self.t)

    def transform(self, pts: np.ndarray) -> np.ndarray:
        """
        对 3D 点或点集进行坐标变换: P_out = R * P_in + t
        支持输入 (3,) 或 (N, 3)
        """
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim == 1:
            return self.R @ pts + self.t
        else:
            return (pts @ self.R.T) + self.t

    @classmethod
    def exp(cls, xi: np.ndarray) -> SE3:
        """
        SE(3) 指数映射: xi = [omega (3), rho (3)]^T -> SE3
        对应 g2o::SE3Quat::exp(xi)
        """
        xi = np.asarray(xi, dtype=np.float64).reshape(6)
        omega = xi[:3]
        rho = xi[3:6]

        theta_sq = float(np.dot(omega, omega))
        theta = np.sqrt(theta_sq)
        W = skew(omega)
        I3 = np.eye(3, dtype=np.float64)

        if theta < 1e-8:
            R = I3 + W + 0.5 * (W @ W)
            V = I3 + 0.5 * W + (1.0 / 6.0) * (W @ W)
        else:
            A = np.sin(theta) / theta
            B = (1.0 - np.cos(theta)) / theta_sq
            C = (1.0 - A) / theta_sq
            R = I3 + A * W + B * (W @ W)
            V = I3 + B * W + C * (W @ W)

        t = V @ rho
        return cls(R, t)

    def log(self) -> np.ndarray:
        """
        SE(3) 对数映射: SE3 -> xi = [omega (3), rho (3)]^T in R^6
        """
        omega = log_so3(self.R)
        theta_sq = float(np.dot(omega, omega))
        theta = np.sqrt(theta_sq)
        W = skew(omega)
        I3 = np.eye(3, dtype=np.float64)

        if theta < 1e-8:
            V_inv = I3 - 0.5 * W + (1.0 / 12.0) * (W @ W)
        else:
            half_theta = 0.5 * theta
            cot_half = np.cos(half_theta) / np.sin(half_theta)
            V_inv = I3 - 0.5 * W + (1.0 / theta_sq) * (1.0 - half_theta * cot_half) * (W @ W)

        rho = V_inv @ self.t
        return np.concatenate([omega, rho])

    def normalize(self) -> None:
        """利用 SVD 保持旋转矩阵正交性 (消除多次迭代累积舍入误差)"""
        U, _, Vt = np.linalg.svd(self.R)
        self.R = U @ Vt
        if np.linalg.det(self.R) < 0:
            U[:, -1] *= -1.0
            self.R = U @ Vt

    def __repr__(self) -> str:
        omega = log_so3(self.R)
        return f"SE3(rot_vec={np.round(omega, 4)}, t={np.round(self.t, 4)})"
