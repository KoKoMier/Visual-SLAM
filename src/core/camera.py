# -*- coding: utf-8 -*-
"""
core/camera.py
==============
双目针孔相机模型与双目去畸变极线校正模块 (StereoCamera)
严格 1:1 对应未来 RK3566 C++ 端 StereoCamera.h / .cpp。

坐标系约定 (遵循 ORB-SLAM3 标准):
- 相机坐标系: X 向右, Y 向下, Z 向前 (光轴方向)
- 双目投影模型:
  u   = fx * (X / Z) + cx
  v   = fy * (Y / Z) + cy
  u_r = u - bf / Z   (其中 bf = baseline * fx, 视差 d = u - u_r = bf / Z)
"""

from __future__ import annotations
import numpy as np
import cv2


class StereoCamera:
    """
    双目相机模型 (支持已校正图像直接输入，或原始双目图像在线极线校正)
    """
    __slots__ = (
        "fx", "fy", "cx", "cy",
        "inv_fx", "inv_fy",
        "baseline", "bf",
        "width", "height",
        "th_depth", "fps",
        "K", "T_b_c1",
        "_map1_l", "_map2_l", "_map1_r", "_map2_r"
    )

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        baseline: float,
        width: int = 752,
        height: int = 480,
        th_depth_factor: float = 40.0,
        fps: float = 20.0,
        T_b_c1: np.ndarray | None = None
    ):
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.inv_fx = 1.0 / self.fx
        self.inv_fy = 1.0 / self.fy

        self.baseline = float(baseline)
        self.bf = self.baseline * self.fx
        self.width = int(width)
        self.height = int(height)
        # ORB-SLAM3 近点/远点阈值: 默认基线的 40 倍以内为强几何约束近点 (Close Stereo Points)
        self.th_depth = self.baseline * float(th_depth_factor)
        self.fps = float(fps)

        self.K = np.array([
            [self.fx, 0.0,     self.cx],
            [0.0,     self.fy, self.cy],
            [0.0,     0.0,     1.0    ]
        ], dtype=np.float64)

        # 左相机到 IMU (Body) 的外参矩阵 T_b_c1 (4x4)
        if T_b_c1 is None:
            self.T_b_c1 = np.eye(4, dtype=np.float64)
        else:
            self.T_b_c1 = np.asarray(T_b_c1, dtype=np.float64).reshape(4, 4).copy()

        self._map1_l = None
        self._map2_l = None
        self._map1_r = None
        self._map2_r = None

    def init_stereo_rectification(
        self,
        K_l: np.ndarray, D_l: np.ndarray,
        K_r: np.ndarray, D_r: np.ndarray,
        R_r_l: np.ndarray, t_r_l: np.ndarray
    ) -> None:
        """
        根据左右原始内参、畸变系数与左右外参 T_r_l 初始化双目立体极线校正映射表 (cv2.stereoRectify)
        校正后更新当前相机的有效内参 fx, fy, cx, cy, baseline, bf
        """
        img_size = (self.width, self.height)
        K_l = np.ascontiguousarray(K_l, dtype=np.float64).reshape(3, 3)
        D_l = np.ascontiguousarray(D_l, dtype=np.float64).reshape(-1, 1)
        K_r = np.ascontiguousarray(K_r, dtype=np.float64).reshape(3, 3)
        D_r = np.ascontiguousarray(D_r, dtype=np.float64).reshape(-1, 1)
        R_r_l = np.ascontiguousarray(R_r_l, dtype=np.float64).reshape(3, 3)
        t_r_l = np.ascontiguousarray(t_r_l, dtype=np.float64).reshape(3, 1)
        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
            K_l, D_l, K_r, D_r, img_size, R_r_l, t_r_l,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        self._map1_l, self._map2_l = cv2.initUndistortRectifyMap(
            K_l, D_l, R1, P1, img_size, cv2.CV_32FC1
        )
        self._map1_r, self._map2_r = cv2.initUndistortRectifyMap(
            K_r, D_r, R2, P2, img_size, cv2.CV_32FC1
        )

        # 更新校正后的虚拟相机参数
        self.fx = float(P1[0, 0])
        self.fy = float(P1[1, 1])
        self.cx = float(P1[0, 2])
        self.cy = float(P1[1, 2])
        self.inv_fx = 1.0 / self.fx
        self.inv_fy = 1.0 / self.fy
        self.bf = float(-P2[0, 3])
        self.baseline = self.bf / self.fx
        self.K = P1[:3, :3].copy()

        # 同步修正左相机到 Body 外参旋转: R_b_c1_rect = R_b_c1_raw * R1^T
        self.T_b_c1[:3, :3] = self.T_b_c1[:3, :3] @ R1.T

    def rectify_stereo(self, img_l: np.ndarray, img_r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """对原始双目图像进行去畸变与极线对齐"""
        if self._map1_l is None:
            return img_l, img_r
        rect_l = cv2.remap(img_l, self._map1_l, self._map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(img_r, self._map1_r, self._map2_r, cv2.INTER_LINEAR)
        return rect_l, rect_r

    def rectify_mono(self, img_l: np.ndarray) -> np.ndarray:
        """对纯单目图像执行去畸变校正"""
        if self._map1_l is None:
            return img_l
        return cv2.remap(img_l, self._map1_l, self._map2_l, cv2.INTER_LINEAR)

    def project_mono(self, p_c: np.ndarray) -> np.ndarray:
        """
        单目投影: P_c = [X, Y, Z]^T -> [u, v]^T
        支持 (3,) 或 (N, 3)
        """
        p_c = np.asarray(p_c, dtype=np.float64)
        if p_c.ndim == 1:
            inv_z = 1.0 / p_c[2]
            u = self.fx * p_c[0] * inv_z + self.cx
            v = self.fy * p_c[1] * inv_z + self.cy
            return np.array([u, v], dtype=np.float64)
        else:
            inv_z = 1.0 / p_c[:, 2]
            u = self.fx * p_c[:, 0] * inv_z + self.cx
            v = self.fy * p_c[:, 1] * inv_z + self.cy
            return np.stack([u, v], axis=-1)

    def project_stereo(self, p_c: np.ndarray) -> np.ndarray:
        """
        双目投影 (对应 ORB-SLAM3 EdgeStereoSE3ProjectXYZ):
        P_c = [X, Y, Z]^T -> [u, v, u_r]^T, 其中 u_r = u - bf / Z
        支持 (3,) 或 (N, 3)
        """
        p_c = np.asarray(p_c, dtype=np.float64)
        if p_c.ndim == 1:
            inv_z = 1.0 / p_c[2]
            u = self.fx * p_c[0] * inv_z + self.cx
            v = self.fy * p_c[1] * inv_z + self.cy
            u_r = u - self.bf * inv_z
            return np.array([u, v, u_r], dtype=np.float64)
        else:
            inv_z = 1.0 / p_c[:, 2]
            u = self.fx * p_c[:, 0] * inv_z + self.cx
            v = self.fy * p_c[:, 1] * inv_z + self.cy
            u_r = u - self.bf * inv_z
            return np.stack([u, v, u_r], axis=-1)

    def unproject_depth(self, u: float, v: float, depth: float) -> np.ndarray:
        """
        已知左目像素 (u, v) 与深度 Z 恢复相机坐标系下 3D 点 P_c = [X, Y, Z]^T
        """
        z = float(depth)
        x = (float(u) - self.cx) * z * self.inv_fx
        y = (float(v) - self.cy) * z * self.inv_fy
        return np.array([x, y, z], dtype=np.float64)

    def is_in_image(self, u: float, v: float, margin: float = 0.0) -> bool:
        """判断像素坐标是否位于图像有效视野内"""
        return (margin <= u < self.width - margin) and (margin <= v < self.height - margin)

    @classmethod
    def create_euroc_camera(cls) -> StereoCamera:
        """
        创建标准 EuRoC MAV 数据集 (MH_01~05 / V1 / V2) 双目相机配置
        包含原始 cam0/cam1 内参、畸变与外参，自动完成双目极线校正初始化
        """
        cam = cls(
            fx=458.654, fy=457.296, cx=367.215, cy=248.375,
            baseline=0.11007, width=752, height=480, th_depth_factor=35.0, fps=20.0
        )
        K_l = np.array([[458.654, 0.0, 367.215], [0.0, 457.296, 248.375], [0.0, 0.0, 1.0]], dtype=np.float64)
        D_l = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05], dtype=np.float64)
        K_r = np.array([[457.587, 0.0, 379.999], [0.0, 456.134, 255.238], [0.0, 0.0, 1.0]], dtype=np.float64)
        D_r = np.array([-0.28368365, 0.07451284, -0.00010473, -3.55590700e-05], dtype=np.float64)

        # EuRoC Body(IMU) 到 cam0 / cam1 的外参矩阵 T_b_c0, T_b_c1
        T_b_c0 = np.array([
            [ 0.0148655429818, -0.999880929698,  0.00414029679422, -0.0216401454975],
            [ 0.999557249008,   0.0149672133247, 0.025715529948,   -0.064676986768 ],
            [-0.0257744366974,  0.00375618835797, 0.999660727178,   0.00981073058949],
            [ 0.0,              0.0,              0.0,              1.0             ]
        ], dtype=np.float64)

        T_b_c1 = np.array([
            [ 0.0125552670891, -0.999755099723,  0.0182237714554, -0.0198435579556],
            [ 0.999598781151,   0.0130119051815, 0.0251588363115,  0.0453689425024],
            [-0.0253898008918,  0.0179005838253, 0.999517347078,   0.00786212447038],
            [ 0.0,              0.0,              0.0,              1.0             ]
        ], dtype=np.float64)

        T_c1_c0 = np.linalg.inv(T_b_c1) @ T_b_c0
        cam.T_b_c1 = T_b_c0.copy()
        cam.init_stereo_rectification(K_l, D_l, K_r, D_r, T_c1_c0[:3, :3], T_c1_c0[:3, 3])
        return cam

    @classmethod
    def from_calib_file(cls, calib_json_path: str) -> StereoCamera:
        """
        从 calibrate_stereo_camera.py 生成的 JSON 标定文件直接构建并初始化双目极线校正相机
        """
        import json
        with open(calib_json_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        width = int(cfg["width"])
        height = int(cfg["height"])
        fps = float(cfg.get("fps", 30.0))
        K_l = np.asarray(cfg["K_l"], dtype=np.float64).reshape(3, 3)
        D_l = np.asarray(cfg["D_l"], dtype=np.float64).reshape(-1)
        K_r = np.asarray(cfg["K_r"], dtype=np.float64).reshape(3, 3)
        D_r = np.asarray(cfg["D_r"], dtype=np.float64).reshape(-1)
        R_r_l = np.asarray(cfg["R_r_l"], dtype=np.float64).reshape(3, 3)
        t_r_l = np.asarray(cfg["t_r_l"], dtype=np.float64).reshape(3, 1)
        T_b_c1 = np.asarray(cfg.get("T_b_c1", np.eye(4).tolist()), dtype=np.float64).reshape(4, 4)

        baseline_init = float(np.linalg.norm(t_r_l))
        cam = cls(
            fx=float(K_l[0, 0]),
            fy=float(K_l[1, 1]),
            cx=float(K_l[0, 2]),
            cy=float(K_l[1, 2]),
            baseline=max(1e-3, baseline_init),
            width=width,
            height=height,
            th_depth_factor=float(cfg.get("th_depth_factor", 35.0)),
            fps=fps,
            T_b_c1=T_b_c1
        )
        cam.init_stereo_rectification(K_l, D_l, K_r, D_r, R_r_l, t_r_l)
        return cam
