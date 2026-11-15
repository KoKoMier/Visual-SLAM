# -*- coding: utf-8 -*-
"""
frontend/imu_preintegration.py
==============================
流形上的 IMU 预积分器 (Forster et al., T-RO 2017 / ORB-SLAM3 ImuTypes.cc)
严格 1:1 对齐 C++ ORB_SLAM3::IMU::Preintegrated 类。

核心功能:
1. 在两帧图像之间对高频 IMU 角速度 gyro 与线加速度 acc 进行中点/欧拉流形积分，
   计算相对旋转 dR (3x3)、相对速度 dv (3,) 与相对位移 dp (3,)。
2. 递推传播 9x9 预积分噪声协方差矩阵 C (旋转、速度、位置) 及 6x6 零偏随机游走协方差。
3. 递推计算预积分量对陀螺仪零偏 bg 与加速度计零偏 ba 的一阶解析雅可比矩阵:
   J_R_bg, J_v_bg, J_v_ba, J_p_bg, J_p_ba。
4. 当后端优化更新零偏 (bg_new, ba_new) 时，通过一阶泰勒展开直接修正 dR, dv, dp，
   无需重新积分原始高频 IMU 数据！
"""

# Python 3.6 compatible
from dataclasses import dataclass
import numpy as np

from ..core.lie_algebra import SE3, skew, exp_so3, right_jacobian_so3


@dataclass
class ImuMeasurement:
    """单条 IMU 测量数据"""
    timestamp: float
    acc: np.ndarray   # (3,) m/s^2
    gyro: np.ndarray  # (3,) rad/s


class ImuPreintegrator:
    """
    Forster 流形预积分器 (对应 ORB-SLAM3 IMU::Preintegrated)
    """
    __slots__ = (
        "bg", "ba", "d_bg", "d_ba",
        "dR", "dv", "dp", "dt_total",
        "J_R_bg", "J_v_bg", "J_v_ba", "J_p_bg", "J_p_ba",
        "cov", "noise_gyro_cov", "noise_acc_cov",
        "measurements"
    )

    def __init__(
        self,
        bg = None,
        ba = None,
        gyro_noise_sigma = 1.6968e-4,
        acc_noise_sigma = 2.0000e-3
    ):
        self.bg = np.zeros(3, dtype=np.float64) if bg is None else np.asarray(bg, dtype=np.float64).reshape(3).copy()
        self.ba = np.zeros(3, dtype=np.float64) if ba is None else np.asarray(ba, dtype=np.float64).reshape(3).copy()
        self.d_bg = np.zeros(3, dtype=np.float64)
        self.d_ba = np.zeros(3, dtype=np.float64)

        self.noise_gyro_cov = (float(gyro_noise_sigma) ** 2) * np.eye(3, dtype=np.float64)
        self.noise_acc_cov = (float(acc_noise_sigma) ** 2) * np.eye(3, dtype=np.float64)

        self.dR = np.eye(3, dtype=np.float64)
        self.dv = np.zeros(3, dtype=np.float64)
        self.dp = np.zeros(3, dtype=np.float64)
        self.dt_total = 0.0

        # 对零偏的一阶解析雅可比矩阵 (3x3)
        self.J_R_bg = np.zeros((3, 3), dtype=np.float64)
        self.J_v_bg = np.zeros((3, 3), dtype=np.float64)
        self.J_v_ba = np.zeros((3, 3), dtype=np.float64)
        self.J_p_bg = np.zeros((3, 3), dtype=np.float64)
        self.J_p_ba = np.zeros((3, 3), dtype=np.float64)

        # 9x9 预积分误差协方差矩阵 [delta_theta (3), delta_v (3), delta_p (3)]
        self.cov = np.zeros((9, 9), dtype=np.float64)
        self.measurements = []

    def reset(self, bg = None, ba = None) :
        if bg is not None:
            self.bg = np.asarray(bg, dtype=np.float64).reshape(3).copy()
        if ba is not None:
            self.ba = np.asarray(ba, dtype=np.float64).reshape(3).copy()
        self.d_bg.fill(0.0)
        self.d_ba.fill(0.0)
        self.dR = np.eye(3, dtype=np.float64)
        self.dv.fill(0.0)
        self.dp.fill(0.0)
        self.dt_total = 0.0
        self.J_R_bg.fill(0.0)
        self.J_v_bg.fill(0.0)
        self.J_v_ba.fill(0.0)
        self.J_p_bg.fill(0.0)
        self.J_p_ba.fill(0.0)
        self.cov.fill(0.0)
        self.measurements.clear()

    def integrate_new_measurement(self, acc, gyro, dt) :
        """
        单步预积分递推 (对应 ORB-SLAM3 Preintegrated::IntegrateNewMeasurement)
        """
        dt = float(dt)
        if dt <= 0.0:
            return

        acc_v = np.asarray(acc, dtype=np.float64).reshape(3)
        gyro_v = np.asarray(gyro, dtype=np.float64).reshape(3)
        self.measurements.append((acc_v.copy(), gyro_v.copy(), dt))

        # 扣除当前零偏估计
        a_unbiased = acc_v - self.ba
        w_unbiased = gyro_v - self.bg

        # 1. 更新位置与速度 (使用更新前的 dR)
        dt2_half = 0.5 * dt * dt
        acc_skew = skew(a_unbiased)
        dR_a = self.dR @ a_unbiased

        self.dp = self.dp + self.dv * dt + dR_a * dt2_half
        self.dv = self.dv + dR_a * dt

        # 2. 计算当前步旋转增量 dR_step 与右雅可比 J_r
        omega_dt = w_unbiased * dt
        dR_step = exp_so3(omega_dt)
        Jr = right_jacobian_so3(omega_dt)

        # 3. 传播 9x9 噪声协方差矩阵: cov_{k+1} = A * cov_k * A^T + B * Q * B^T
        A = np.eye(9, dtype=np.float64)
        A[0:3, 0:3] = dR_step.T
        A[3:6, 0:3] = -self.dR @ acc_skew * dt
        A[6:9, 0:3] = -self.dR @ acc_skew * dt2_half
        A[6:9, 3:6] = np.eye(3, dtype=np.float64) * dt

        B = np.zeros((9, 6), dtype=np.float64)
        B[0:3, 0:3] = Jr * dt
        B[3:6, 3:6] = self.dR * dt
        B[6:9, 3:6] = self.dR * dt2_half

        Q = np.zeros((6, 6), dtype=np.float64)
        Q[0:3, 0:3] = (1.0 / dt) * self.noise_gyro_cov
        Q[3:6, 3:6] = (1.0 / dt) * self.noise_acc_cov
        self.cov = A @ self.cov @ A.T + B @ Q @ B.T

        # 4. 递推更新零偏一阶雅可比矩阵 (必须注意更新顺序: 先 p，再 v，最后 R)
        self.J_p_ba = self.J_p_ba + self.J_v_ba * dt - self.dR * dt2_half
        self.J_p_bg = self.J_p_bg + self.J_v_bg * dt - (self.dR @ acc_skew @ self.J_R_bg) * dt2_half
        self.J_v_ba = self.J_v_ba - self.dR * dt
        self.J_v_bg = self.J_v_bg - (self.dR @ acc_skew @ self.J_R_bg) * dt
        self.J_R_bg = (dR_step.T @ self.J_R_bg) - Jr * dt

        # 5. 更新累积旋转矩阵并保持正交性
        self.dR = self.dR @ dR_step
        self.dt_total += dt

    def copy(self) :
        """深拷贝当前预积分器状态 (对应 ORB-SLAM3 new IMU::Preintegrated(pInt))"""
        clone = ImuPreintegrator(bg=self.bg, ba=self.ba)
        clone.d_bg = self.d_bg.copy()
        clone.d_ba = self.d_ba.copy()
        clone.dR = self.dR.copy()
        clone.dv = self.dv.copy()
        clone.dp = self.dp.copy()
        clone.dt_total = float(self.dt_total)
        clone.J_R_bg = self.J_R_bg.copy()
        clone.J_v_bg = self.J_v_bg.copy()
        clone.J_v_ba = self.J_v_ba.copy()
        clone.J_p_bg = self.J_p_bg.copy()
        clone.J_p_ba = self.J_p_ba.copy()
        clone.cov = self.cov.copy()
        clone.noise_gyro_cov = self.noise_gyro_cov.copy()
        clone.noise_acc_cov = self.noise_acc_cov.copy()
        clone.measurements = [(a.copy(), g.copy(), float(dt)) for a, g, dt in self.measurements]
        return clone

    def set_new_bias(self, bg_new, ba_new) :
        """设置新的零偏并更新 delta_bias (对应 ORB-SLAM3 Preintegrated::SetNewBias)"""
        bg_v = np.asarray(bg_new, dtype=np.float64).reshape(3)
        ba_v = np.asarray(ba_new, dtype=np.float64).reshape(3)
        self.d_bg = bg_v - self.bg
        self.d_ba = ba_v - self.ba

    def reintegrate(self, bg_new = None, ba_new = None) :
        """当零偏更新幅度较大时重新积分缓存的原始测量 (对应 ORB-SLAM3 Preintegrated::Reintegrate)"""
        saved = list(self.measurements)
        new_bg = self.bg + self.d_bg if bg_new is None else np.asarray(bg_new, dtype=np.float64).reshape(3)
        new_ba = self.ba + self.d_ba if ba_new is None else np.asarray(ba_new, dtype=np.float64).reshape(3)
        self.reset(bg=new_bg, ba=new_ba)
        for acc_v, gyro_v, dt in saved:
            self.integrate_new_measurement(acc_v, gyro_v, dt)

    def merge_previous(self, prev_preint) :
        """
        关键帧剔除时将前一关键帧的预积分段合并至当前预积分器
        对应 ORB-SLAM3 Preintegrated::MergePrevious(IMU::Preintegrated* pPrev)
        """
        if prev_preint is None:
            return
        saved_curr = list(self.measurements)
        saved_prev = list(prev_preint.measurements)
        self.reset(bg=prev_preint.bg, ba=prev_preint.ba)
        for acc_v, gyro_v, dt in saved_prev:
            self.integrate_new_measurement(acc_v, gyro_v, dt)
        for acc_v, gyro_v, dt in saved_curr:
            self.integrate_new_measurement(acc_v, gyro_v, dt)

    def get_updated_delta_rotation(self, bg_new = None) :
        """零偏变化后的一阶修正旋转: dR_updated = dR * Exp(J_R_bg * delta_bg)"""
        dbg = self.d_bg if bg_new is None else (np.asarray(bg_new, dtype=np.float64).reshape(3) - self.bg)
        return self.dR @ exp_so3(self.J_R_bg @ dbg)

    def get_updated_delta_velocity(
        self,
        bg_new = None,
        ba_new = None
    ) :
        """零偏变化后的一阶修正速度: dv_updated = dv + J_v_bg * dbg + J_v_ba * dba"""
        dbg = self.d_bg if bg_new is None else (np.asarray(bg_new, dtype=np.float64).reshape(3) - self.bg)
        dba = self.d_ba if ba_new is None else (np.asarray(ba_new, dtype=np.float64).reshape(3) - self.ba)
        return self.dv + (self.J_v_bg @ dbg) + (self.J_v_ba @ dba)

    def get_updated_delta_position(
        self,
        bg_new = None,
        ba_new = None
    ) :
        """零偏变化后的一阶修正位置: dp_updated = dp + J_p_bg * dbg + J_p_ba * dba"""
        dbg = self.d_bg if bg_new is None else (np.asarray(bg_new, dtype=np.float64).reshape(3) - self.bg)
        dba = self.d_ba if ba_new is None else (np.asarray(ba_new, dtype=np.float64).reshape(3) - self.ba)
        return self.dp + (self.J_p_bg @ dbg) + (self.J_p_ba @ dba)

    def predict_state(
        self,
        T_wb_i,
        v_w_i,
        g_w = np.array([0.0, 0.0, -9.81], dtype=np.float64)
    ) :
        """
        根据上一帧 Body 位姿 T_wb_i 与世界系速度 v_w_i，利用预积分结果预测当前帧位姿 T_wb_j 与速度 v_w_j
        """
        if self.dt_total <= 0.0:
            return T_wb_i.copy(), v_w_i.copy()
        dt = self.dt_total
        R_wb_i = T_wb_i.R
        p_wb_i = T_wb_i.t

        dR = self.get_updated_delta_rotation()
        dv = self.get_updated_delta_velocity()
        dp = self.get_updated_delta_position()

        R_wb_j = R_wb_i @ dR
        v_w_j = v_w_i + g_w * dt + (R_wb_i @ dv)
        p_wb_j = p_wb_i + v_w_i * dt + 0.5 * g_w * (dt * dt) + (R_wb_i @ dp)
        return SE3(R_wb_j, p_wb_j), v_w_j
