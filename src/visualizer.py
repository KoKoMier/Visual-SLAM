# -*- coding: utf-8 -*-
"""
src/visualizer.py
=================
SLAM 摄像头 6DoF 运动视锥体 (Camera Frustum) 与实时算法状态流程图视频生成器
将每一帧的：
1. 左目去畸变相机画面 + 2D 特征跟踪光流场 + 深度着色内点
2. 动态 SLAM 算法流水线状态图 (高亮当前帧走的是 Path A 光流快速路径还是 Path B 关键帧 ORB+SAD 路径，以及是否触发 Local BA)
3. 3D 空间摄像头视锥体 (6DoF Camera Frustum Pyramid) + 实时观测射线 + 3D 地图点云 + 估计/真值轨迹对比
4. 俯视平面视场角扫掠图 (Top-Down FOV Cone) + X/Y/Z 三轴实时跟随曲线
合成为 1920x1080 全高清 MP4 视频、GIF 动图与关键阶段分镜流程图，并统一保存至 outputs/ 目录。
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class FrameVisRecord:
    """记录单帧 SLAM 运行状态以供生成摄像头运动与算法流程图视频"""
    frame_idx: int
    timestamp: float
    img_rect_l: np.ndarray                      # 去畸变后的左目灰度图
    R_wc_raw: np.ndarray                        # (3, 3) 原始估计旋转
    t_wc_raw: np.ndarray                        # (3,) 原始估计平移
    gt_pos: np.ndarray | None                   # (3,) 真值位置
    keypoints_uv: np.ndarray                    # (M, 2) 当前帧跟踪到的有效内点像素坐标
    keypoints_depth: np.ndarray                 # (M,) 对应深度
    prev_uv: np.ndarray                         # (K, 2) 光流跟踪上一帧像素坐标
    curr_uv: np.ndarray                         # (K, 2) 光流跟踪当前帧像素坐标
    observed_mp_xyz_raw: np.ndarray             # (M, 3) 当前帧观测到的 3D 地图点世界坐标
    all_mp_xyz_raw: np.ndarray                  # (P, 3) 截至当前帧的活跃 3D 地图点世界坐标
    is_keyframe: bool                           # 当前帧是否被判定为关键帧
    used_optical_flow: bool                     # 当前帧是否走了金字塔光流快速路径
    inlier_count: int                           # Motion-only BA 后的有效内点数
    num_keyframes: int                          # 当前地图活跃关键帧总数
    num_mappoints: int                          # 当前地图活跃 3D 点总数
    frontend_ms: float                          # 前端耗时 (ms)
    backend_ms: float                           # 后端耗时 (ms)


def _load_chinese_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """加载 Windows 微软雅黑或黑体以支持高清中文渲染"""
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


class SLAMVideoFlowVisualizer:
    """生成 1920x1080 摄像头 6DoF 运动与算法状态流程图视频"""

    def __init__(self, width: int = 1920, height: int = 1080) -> None:
        self.width = width
        self.height = height
        self.font_title = _load_chinese_font(24, bold=True)
        self.font_h2 = _load_chinese_font(18, bold=True)
        self.font_body = _load_chinese_font(15, bold=False)
        self.font_body_bold = _load_chinese_font(15, bold=True)
        self.font_small = _load_chinese_font(13, bold=False)
        self.font_tiny = _load_chinese_font(12, bold=False)

    @staticmethod
    def _project_3d_to_2d(
        pts_3d: np.ndarray,
        center_3d: np.ndarray,
        yaw_deg: float,
        pitch_deg: float,
        scale: float,
        canvas_cx: float,
        canvas_cy: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        将对齐后的世界系 3D 点 (X, Y, Z) 投影到 3D 轴测/透视画布平面
        使得轨迹、点云与 3D 摄像头视锥体具有清晰的立体纵深感
        """
        if len(pts_3d) == 0:
            return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)

        rel = pts_3d - center_3d.reshape(1, 3)
        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)

        # 绕 Z 轴(高度或水平轴)与俯仰轴旋转以获得最佳立体观察视角
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)

        # 将 (X, Y, Z) 映射为观察系坐标
        x1 = cy * rel[:, 0] - sy * rel[:, 1]
        y1 = sy * rel[:, 0] + cy * rel[:, 1]
        z1 = rel[:, 2]

        u_cam = x1
        v_cam = -sp * y1 - cp * z1
        depth = cp * y1 - sp * z1 + 8.5

        persp = 8.5 / np.clip(depth, 2.5, 25.0)
        u_pix = canvas_cx + u_cam * scale * persp
        v_pix = canvas_cy + v_cam * scale * persp
        return np.column_stack([u_pix, v_pix]), depth

    def _draw_camera_frustum_3d(
        self,
        canvas: np.ndarray,
        R_wc: np.ndarray,
        t_wc: np.ndarray,
        center_3d: np.ndarray,
        yaw_deg: float,
        pitch_deg: float,
        scale: float,
        cx: float,
        cy: float,
        frustum_size: float = 0.28,
        color_bgr: tuple[int, int, int] = (40, 60, 255),
        thickness: int = 2,
        draw_axes: bool = True,
    ) -> np.ndarray:
        """在 3D 视图中绘制摄像头的五面体视锥 (Camera Frustum Pyramid) 与局部坐标轴"""
        w = frustum_size * 0.75
        h = frustum_size * 0.50
        z = frustum_size * 1.10

        # 相机坐标系下的光心与四个成像平面角点
        pts_c = np.array([
            [0.0, 0.0, 0.0],
            [-w, -h, z],
            [w, -h, z],
            [w, h, z],
            [-w, h, z],
        ], dtype=np.float64)

        pts_w = (pts_c @ R_wc.T) + t_wc.reshape(1, 3)
        uv, _ = self._project_3d_to_2d(pts_w, center_3d, yaw_deg, pitch_deg, scale, cx, cy)
        uv_i = np.round(uv).astype(np.int32)

        o = tuple(uv_i[0])
        c1, c2, c3, c4 = tuple(uv_i[1]), tuple(uv_i[2]), tuple(uv_i[3]), tuple(uv_i[4])

        # 四条视锥侧棱与底面矩形框
        for c in (c1, c2, c3, c4):
            cv2.line(canvas, o, c, color_bgr, thickness, cv2.LINE_AA)
        cv2.line(canvas, c1, c2, color_bgr, thickness, cv2.LINE_AA)
        cv2.line(canvas, c2, c3, color_bgr, thickness, cv2.LINE_AA)
        cv2.line(canvas, c3, c4, color_bgr, thickness, cv2.LINE_AA)
        cv2.line(canvas, c4, c1, color_bgr, thickness, cv2.LINE_AA)

        if draw_axes:
            axis_len = frustum_size * 1.15
            axes_c = np.array([
                [0.0, 0.0, 0.0],
                [axis_len, 0.0, 0.0],  # X轴 (红)
                [0.0, axis_len, 0.0],  # Y轴 (绿)
                [0.0, 0.0, axis_len],  # Z轴/主光轴 (蓝/青)
            ], dtype=np.float64)
            axes_w = (axes_c @ R_wc.T) + t_wc.reshape(1, 3)
            ax_uv, _ = self._project_3d_to_2d(axes_w, center_3d, yaw_deg, pitch_deg, scale, cx, cy)
            ax_i = np.round(ax_uv).astype(np.int32)
            cv2.arrowedLine(canvas, tuple(ax_i[0]), tuple(ax_i[1]), (60, 60, 255), 2, cv2.LINE_AA, tipLength=0.25)
            cv2.arrowedLine(canvas, tuple(ax_i[0]), tuple(ax_i[2]), (60, 220, 60), 2, cv2.LINE_AA, tipLength=0.25)
            cv2.arrowedLine(canvas, tuple(ax_i[0]), tuple(ax_i[3]), (255, 200, 40), 2, cv2.LINE_AA, tipLength=0.25)

        return uv_i[0]

    def render_all_outputs(
        self,
        records: list[FrameVisRecord],
        R_align: np.ndarray,
        t_align: np.ndarray,
        ate_metrics: dict[str, float],
        output_dir: str,
        prefix: str = "euroc",
        fps: int = 15,
    ) -> dict[str, str]:
        """
        渲染完整视频流程图 (.mp4)、动图 (.gif) 与多阶段分镜汇总图 (.png)
        """
        os.makedirs(output_dir, exist_ok=True)
        mp4_path = os.path.join(output_dir, f"{prefix}_camera_motion_flow.mp4")
        gif_path = os.path.join(output_dir, f"{prefix}_camera_motion_flow.gif")
        storyboard_path = os.path.join(output_dir, f"{prefix}_camera_motion_storyboard.png")

        n_frames = len(records)
        if n_frames == 0:
            return {}

        # 预先将所有估计位姿与真值对齐到同一物理坐标系
        aligned_t_wc = np.zeros((n_frames, 3), dtype=np.float64)
        aligned_R_wc = np.zeros((n_frames, 3, 3), dtype=np.float64)
        gt_positions = np.zeros((n_frames, 3), dtype=np.float64)

        for i, rec in enumerate(records):
            aligned_t_wc[i] = R_align @ rec.t_wc_raw + t_align
            aligned_R_wc[i] = R_align @ rec.R_wc_raw
            if rec.gt_pos is not None:
                gt_positions[i] = rec.gt_pos
            else:
                gt_positions[i] = aligned_t_wc[i]

        # 计算 3D 视图与俯视图的全局包围盒中心与缩放比例
        all_traj_pts = np.vstack([aligned_t_wc, gt_positions])
        center_3d = np.mean(all_traj_pts, axis=0)
        span_max = float(np.max(np.ptp(all_traj_pts, axis=0)))
        scale_3d = 470.0 / max(2.2, span_max)

        # 初始化 MP4 编码器
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(mp4_path, fourcc, float(fps), (self.width, self.height))

        gif_frames: list[Image.Image] = []
        storyboard_snapshots: list[tuple[int, np.ndarray]] = []
        snapshot_indices = set(np.linspace(0, n_frames - 1, min(6, n_frames), dtype=int).tolist())

        # 记录历史关键帧的对齐位姿列表
        kf_history: list[tuple[np.ndarray, np.ndarray]] = []

        for idx, rec in enumerate(records):
            if rec.is_keyframe or idx == 0:
                kf_history.append((aligned_R_wc[idx], aligned_t_wc[idx]))

            frame_bgr = self._render_single_dashboard_frame(
                idx=idx,
                total_frames=n_frames,
                rec=rec,
                aligned_t_wc=aligned_t_wc,
                aligned_R_wc=aligned_R_wc,
                gt_positions=gt_positions,
                kf_history=kf_history,
                R_align=R_align,
                t_align=t_align,
                center_3d=center_3d,
                scale_3d=scale_3d,
                ate_rmse_cm=ate_metrics["aligned_rmse_m"] * 100.0,
                records=records,
            )

            writer.write(frame_bgr)

            # 每隔 2 帧采样一帧制作轻量流畅的 GIF 动图
            if idx % 2 == 0 or idx == n_frames - 1:
                rgb_small = cv2.cvtColor(
                    cv2.resize(frame_bgr, (960, 540), interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2RGB,
                )
                gif_frames.append(Image.fromarray(rgb_small))

            if idx in snapshot_indices:
                storyboard_snapshots.append((idx, frame_bgr.copy()))

        writer.release()

        # 保存 GIF 动图
        if len(gif_frames) > 0:
            gif_frames[0].save(
                gif_path,
                save_all=True,
                append_images=gif_frames[1:],
                duration=int(1000 * 2 / max(1, fps)),
                loop=0,
                optimize=False,
            )

        # 保存 6 阶段关键帧分镜流程图
        self._save_storyboard_image(storyboard_snapshots, storyboard_path, ate_metrics)

        print(f"  [视频已保存] 摄像头运动与算法流程图 MP4 : {os.path.abspath(mp4_path)}")
        print(f"  [动图已保存] 摄像头运动与算法流程图 GIF : {os.path.abspath(gif_path)}")
        print(f"  [分镜已保存] 摄像头运动关键阶段分镜图   : {os.path.abspath(storyboard_path)}")

        return {
            "mp4": mp4_path,
            "gif": gif_path,
            "storyboard": storyboard_path,
        }

    def _render_single_dashboard_frame(
        self,
        idx: int,
        total_frames: int,
        rec: FrameVisRecord,
        aligned_t_wc: np.ndarray,
        aligned_R_wc: np.ndarray,
        gt_positions: np.ndarray,
        kf_history: list[tuple[np.ndarray, np.ndarray]],
        R_align: np.ndarray,
        t_align: np.ndarray,
        center_3d: np.ndarray,
        scale_3d: float,
        ate_rmse_cm: float,
        records: list[FrameVisRecord],
    ) -> np.ndarray:
        """合成单帧 1920x1080 四分区可视化画面"""
        # 深色科技感背景画布 (BGR)
        canvas = np.full((self.height, self.width, 3), (24, 20, 16), dtype=np.uint8)

        # 四个面板圆角/边框区域背景
        # 1. 顶部状态栏 (0..1920, 0..62)
        cv2.rectangle(canvas, (0, 0), (self.width, 62), (38, 30, 24), -1)
        cv2.line(canvas, (0, 62), (self.width, 62), (90, 75, 60), 2)

        # 2. 左上：实时摄像头画面 + 特征跟踪光流场 (x: 18..878, y: 76..576)
        cv2.rectangle(canvas, (18, 76), (878, 576), (34, 28, 22), -1)
        cv2.rectangle(canvas, (18, 76), (878, 576), (85, 70, 55), 1)

        # 3. 左下：动态 SLAM 算法状态流程图 (x: 18..878, y: 590..1062)
        cv2.rectangle(canvas, (18, 590), (878, 1062), (34, 28, 22), -1)
        cv2.rectangle(canvas, (18, 590), (878, 1062), (85, 70, 55), 1)

        # 4. 右上：3D 摄像头视锥体 (Frustum) 运动与点云重建 (x: 894..1902, y: 76..690)
        cv2.rectangle(canvas, (894, 76), (1902, 690), (30, 25, 20), -1)
        cv2.rectangle(canvas, (894, 76), (1902, 690), (85, 70, 55), 1)

        # 5. 右下：俯视平面 FOV 扫掠图 + 三轴位移实时曲线 (x: 894..1902, y: 704..1062)
        cv2.rectangle(canvas, (894, 704), (1902, 1062), (30, 25, 20), -1)
        cv2.rectangle(canvas, (894, 704), (1902, 1062), (85, 70, 55), 1)

        # =====================================================================
        # Panel A: 左上 — 实时去畸变摄像头画面 + 光流场 + 深度着色内点
        # =====================================================================
        cam_vis = cv2.cvtColor(rec.img_rect_l, cv2.COLOR_GRAY2BGR)
        img_h, img_w = cam_vis.shape[:2]
        target_w, target_h = 832, 435
        sx = target_w / float(img_w)
        sy = target_h / float(img_h)
        cam_vis = cv2.resize(cam_vis, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # 绘制帧间光流运动矢量 (绿色尾迹线)
        if len(rec.prev_uv) > 0 and len(rec.curr_uv) == len(rec.prev_uv):
            for p0, p1 in zip(rec.prev_uv, rec.curr_uv):
                u0, v0 = int(round(p0[0] * sx)), int(round(p0[1] * sy))
                u1, v1 = int(round(p1[0] * sx)), int(round(p1[1] * sy))
                cv2.line(cam_vis, (u0, v0), (u1, v1), (80, 255, 120), 1, cv2.LINE_AA)

        # 绘制当前帧有效跟踪到的 3D 地图点投影 (按深度从青绿到橙红渐变)
        if len(rec.keypoints_uv) > 0:
            for uv, d in zip(rec.keypoints_uv, rec.keypoints_depth):
                u, v = int(round(uv[0] * sx)), int(round(uv[1] * sy))
                ratio = float(np.clip((d - 1.0) / 5.5, 0.0, 1.0))
                # 近处亮绿/青色，远处蓝紫/橙色
                color = (
                    int(255 * ratio + 60 * (1.0 - ratio)),
                    int(240 * (1.0 - ratio) + 90 * ratio),
                    int(60 * (1.0 - ratio) + 255 * ratio),
                )
                cv2.circle(cam_vis, (u, v), 3, color, -1, cv2.LINE_AA)
                cv2.rectangle(cam_vis, (u - 4, v - 4), (u + 4, v + 4), color, 1)

        # 若当前帧为新建关键帧，在相机画面外围加亮琥珀色提示框
        if rec.is_keyframe:
            cv2.rectangle(cam_vis, (2, 2), (target_w - 3, target_h - 3), (0, 180, 255), 3)

        canvas[126 : 126 + target_h, 32 : 32 + target_w] = cam_vis

        # =====================================================================
        # Panel B: 右上 — 3D 空间摄像头视锥体 (Frustum) 运动与点云重建视图
        # =====================================================================
        p3d_cx, p3d_cy = 1398.0, 405.0
        # 随帧号微幅旋转观察角 (-38° -> -18°)，增强立体观察感
        yaw_deg = -35.0 + 16.0 * (idx / max(1, total_frames - 1))
        pitch_deg = 26.0

        # 1. 绘制 3D 地面参考网格
        grid_lines = []
        for gx in np.linspace(center_3d[0] - 2.5, center_3d[0] + 2.5, 9):
            pts_g = np.array([[gx, center_3d[1] - 2.2, center_3d[2] - 0.6], [gx, center_3d[1] + 2.2, center_3d[2] - 0.6]])
            uv_g, _ = self._project_3d_to_2d(pts_g, center_3d, yaw_deg, pitch_deg, scale_3d, p3d_cx, p3d_cy)
            cv2.line(canvas, tuple(np.round(uv_g[0]).astype(int)), tuple(np.round(uv_g[1]).astype(int)), (48, 40, 34), 1, cv2.LINE_AA)
        for gy in np.linspace(center_3d[1] - 2.2, center_3d[1] + 2.2, 9):
            pts_g = np.array([[center_3d[0] - 2.5, gy, center_3d[2] - 0.6], [center_3d[0] + 2.5, gy, center_3d[2] - 0.6]])
            uv_g, _ = self._project_3d_to_2d(pts_g, center_3d, yaw_deg, pitch_deg, scale_3d, p3d_cx, p3d_cy)
            cv2.line(canvas, tuple(np.round(uv_g[0]).astype(int)), tuple(np.round(uv_g[1]).astype(int)), (48, 40, 34), 1, cv2.LINE_AA)

        # 2. 绘制全局活跃 3D 地图点云 (浅蓝灰)
        if len(rec.all_mp_xyz_raw) > 0:
            mp_aligned = (rec.all_mp_xyz_raw @ R_align.T) + t_align.reshape(1, 3)
            dists = np.linalg.norm(mp_aligned - center_3d.reshape(1, 3), axis=1)
            mp_vis = mp_aligned[dists < 6.0]
            if len(mp_vis) > 0:
                uv_mp, _ = self._project_3d_to_2d(mp_vis, center_3d, yaw_deg, pitch_deg, scale_3d, p3d_cx, p3d_cy)
                for pt in np.round(uv_mp).astype(int):
                    if 900 <= pt[0] <= 1895 and 120 <= pt[1] <= 682:
                        cv2.circle(canvas, (int(pt[0]), int(pt[1])), 2, (165, 150, 130), -1, cv2.LINE_AA)

        # 3. 绘制当前帧正在观测的共视 3D 地图点及相机观测射线 (亮青绿色)
        cam_center_2d, _ = self._project_3d_to_2d(
            aligned_t_wc[idx : idx + 1], center_3d, yaw_deg, pitch_deg, scale_3d, p3d_cx, p3d_cy
        )
        cam_c_pt = tuple(np.round(cam_center_2d[0]).astype(int))

        if len(rec.observed_mp_xyz_raw) > 0:
            obs_aligned = (rec.observed_mp_xyz_raw @ R_align.T) + t_align.reshape(1, 3)
            dists = np.linalg.norm(obs_aligned - aligned_t_wc[idx].reshape(1, 3), axis=1)
            obs_vis = obs_aligned[dists < 5.0]
            if len(obs_vis) > 0:
                uv_obs, _ = self._project_3d_to_2d(obs_vis, center_3d, yaw_deg, pitch_deg, scale_3d, p3d_cx, p3d_cy)
                uv_obs_i = np.round(uv_obs).astype(int)
                # 抽样绘制相机光心到 3D 特征点的观测视线束
                step_ray = max(1, len(uv_obs_i) // 18)
                for pt in uv_obs_i[::step_ray]:
                    if 900 <= pt[0] <= 1895 and 120 <= pt[1] <= 682:
                        cv2.line(canvas, cam_c_pt, (int(pt[0]), int(pt[1])), (80, 140, 60), 1, cv2.LINE_AA)
                for pt in uv_obs_i:
                    if 900 <= pt[0] <= 1895 and 120 <= pt[1] <= 682:
                        cv2.circle(canvas, (int(pt[0]), int(pt[1])), 3, (100, 255, 140), -1, cv2.LINE_AA)

        # 4. 绘制历史关键帧视锥体 (小号蓝色视锥)
        for R_kf, t_kf in kf_history[:-1]:
            self._draw_camera_frustum_3d(
                canvas, R_kf, t_kf, center_3d, yaw_deg, pitch_deg, scale_3d,
                p3d_cx, p3d_cy, frustum_size=0.14, color_bgr=(220, 150, 50), thickness=1, draw_axes=False
            )

        # 5. 绘制真值轨迹 (白色虚线) 与 SLAM 估计轨迹 (亮红实线)
        uv_gt_all, _ = self._project_3d_to_2d(gt_positions[: idx + 1], center_3d, yaw_deg, pitch_deg, scale_3d, p3d_cx, p3d_cy)
        uv_est_all, _ = self._project_3d_to_2d(aligned_t_wc[: idx + 1], center_3d, yaw_deg, pitch_deg, scale_3d, p3d_cx, p3d_cy)
        pts_gt_i = np.round(uv_gt_all).astype(np.int32)
        pts_est_i = np.round(uv_est_all).astype(np.int32)

        for k_line in range(1, len(pts_gt_i)):
            if k_line % 2 == 0:
                cv2.line(canvas, tuple(pts_gt_i[k_line - 1]), tuple(pts_gt_i[k_line]), (210, 210, 210), 2, cv2.LINE_AA)
            cv2.line(canvas, tuple(pts_est_i[k_line - 1]), tuple(pts_est_i[k_line]), (50, 70, 255), 3, cv2.LINE_AA)

        # 起点标记
        cv2.circle(canvas, tuple(pts_gt_i[0]), 6, (80, 230, 80), -1, cv2.LINE_AA)

        # 6. 绘制当前帧摄像头 3D 视锥体与三轴姿态 (醒目亮金/红视锥)
        self._draw_camera_frustum_3d(
            canvas, aligned_R_wc[idx], aligned_t_wc[idx], center_3d, yaw_deg, pitch_deg, scale_3d,
            p3d_cx, p3d_cy, frustum_size=0.28, color_bgr=(0, 215, 255), thickness=2, draw_axes=True
        )

        # =====================================================================
        # Panel C: 右下左半 — 俯视平面 (Top-Down X-Y) 摄像头视场角扫掠与轨迹
        # =====================================================================
        td_cx, td_cy = 1130.0, 900.0
        td_scale = 82.0

        def to_td_uv(xyz: np.ndarray) -> np.ndarray:
            rel = xyz - center_3d.reshape(-1, 3)
            u = td_cx + rel[:, 0] * td_scale
            v = td_cy - rel[:, 1] * td_scale
            return np.round(np.column_stack([u, v])).astype(np.int32)

        # 俯视网格圆环
        for rad_m in (1.0, 2.0):
            cv2.circle(canvas, (int(td_cx), int(td_cy)), int(rad_m * td_scale), (48, 40, 34), 1, cv2.LINE_AA)

        # 俯视视场角扇形 (Camera FOV Cone)
        cur_td = to_td_uv(aligned_t_wc[idx : idx + 1])[0]
        fov_pts_c = np.array([[0.0, 0.0, 0.0], [-0.65, 0.0, 1.1], [0.65, 0.0, 1.1]], dtype=np.float64)
        fov_pts_w = (fov_pts_c @ aligned_R_wc[idx].T) + aligned_t_wc[idx].reshape(1, 3)
        fov_td = to_td_uv(fov_pts_w)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [fov_td], (50, 160, 230))
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
        cv2.polylines(canvas, [fov_td], True, (0, 215, 255), 2, cv2.LINE_AA)

        td_gt = to_td_uv(gt_positions[: idx + 1])
        td_est = to_td_uv(aligned_t_wc[: idx + 1])
        for k_line in range(1, len(td_gt)):
            if k_line % 2 == 0:
                cv2.line(canvas, tuple(td_gt[k_line - 1]), tuple(td_gt[k_line]), (200, 200, 200), 2, cv2.LINE_AA)
            cv2.line(canvas, tuple(td_est[k_line - 1]), tuple(td_est[k_line]), (50, 90, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, tuple(cur_td), 5, (0, 215, 255), -1, cv2.LINE_AA)

        # =====================================================================
        # Panel D: 右下右半 — X/Y/Z 三轴位置实时跟踪曲线 (x: 1385..1885, y: 755..1035)
        # =====================================================================
        plot_x0, plot_y0, plot_w, plot_h = 1395, 755, 485, 280
        cv2.rectangle(canvas, (plot_x0, plot_y0), (plot_x0 + plot_w, plot_y0 + plot_h), (22, 18, 15), -1)
        cv2.rectangle(canvas, (plot_x0, plot_y0), (plot_x0 + plot_w, plot_y0 + plot_h), (65, 55, 45), 1)

        val_min = float(min(np.min(aligned_t_wc), np.min(gt_positions))) - 0.2
        val_max = float(max(np.max(aligned_t_wc), np.max(gt_positions))) + 0.2

        def val_to_py(v: float) -> int:
            norm = (v - val_min) / max(1e-5, val_max - val_min)
            return int(round(plot_y0 + plot_h - 12 - norm * (plot_h - 24)))

        def idx_to_px(k_idx: int) -> int:
            return int(round(plot_x0 + 10 + (k_idx / max(1, total_frames - 1)) * (plot_w - 20)))

        axis_colors_gt = [(120, 120, 180), (120, 180, 120), (180, 140, 100)]
        axis_colors_est = [(60, 70, 255), (80, 220, 80), (255, 170, 40)]

        # 先画全序列浅色真值参考虚线
        for ax_i in range(3):
            for k_line in range(1, total_frames):
                if k_line % 2 == 0:
                    p0 = (idx_to_px(k_line - 1), val_to_py(float(gt_positions[k_line - 1, ax_i])))
                    p1 = (idx_to_px(k_line), val_to_py(float(gt_positions[k_line, ax_i])))
                    cv2.line(canvas, p0, p1, axis_colors_gt[ax_i], 1, cv2.LINE_AA)

        # 再画截至当前帧 idx 的实测估计曲线
        for ax_i in range(3):
            for k_line in range(1, idx + 1):
                p0 = (idx_to_px(k_line - 1), val_to_py(float(aligned_t_wc[k_line - 1, ax_i])))
                p1 = (idx_to_px(k_line), val_to_py(float(aligned_t_wc[k_line, ax_i])))
                cv2.line(canvas, p0, p1, axis_colors_est[ax_i], 2, cv2.LINE_AA)

        # 当前帧时间游标竖线
        cur_px = idx_to_px(idx)
        cv2.line(canvas, (cur_px, plot_y0), (cur_px, plot_y0 + plot_h), (0, 215, 255), 1, cv2.LINE_AA)

        # =====================================================================
        # Panel E: 左下 — 动态 SLAM 算法状态流程图 (Dynamic Pipeline Flowchart)
        # =====================================================================
        self._draw_pipeline_flowchart_boxes(canvas, rec, idx, total_frames, records)

        # =====================================================================
        # 使用 PIL 一次性绘制全图高清抗锯齿中文标注与状态数值
        # =====================================================================
        pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # 顶部主标题栏
        draw.text((24, 16), "视觉/惯性 SLAM 摄像头 6DoF 运动与算法流程图 (RK3566 架构)", font=self.font_title, fill=(245, 240, 230))
        cur_err_cm = float(np.linalg.norm(aligned_t_wc[idx] - gt_positions[idx])) * 100.0
        status_str = (
            f"帧号: {idx+1:03d}/{total_frames:03d}   |   "
            f"活跃关键帧: {rec.num_keyframes:2d}   |   "
            f"3D 地图点: {rec.num_mappoints:3d}   |   "
            f"当前帧位置误差: {cur_err_cm:5.2f} cm   |   "
            f"序列 ATE RMSE: {ate_rmse_cm:5.2f} cm"
        )
        draw.text((950, 20), status_str, font=self.font_body_bold, fill=(120, 230, 255))

        # 左上面板标题与 HUD
        mode_tag = "【关键帧路径: 四叉树 ORB + 双目匹配】" if rec.is_keyframe else (
            "【普通帧快速路径: 金字塔 LK 光流跟踪】" if rec.used_optical_flow else "【普通帧重匹配路径: ORB 描述子投影】"
        )
        mode_color = (255, 190, 60) if rec.is_keyframe else (100, 245, 140)
        draw.text((32, 86), f"1. 实时左目去畸变相机视图 & 2D 特征跟踪光流场   {mode_tag}", font=self.font_h2, fill=mode_color)
        draw.text(
            (44, 134),
            f"时间戳: {rec.timestamp:.3f} s  |  BA 有效内点: {rec.inlier_count:3d}  |  前端耗时: {rec.frontend_ms:5.1f} ms",
            font=self.font_small,
            fill=(230, 255, 230),
        )

        # 右上面板标题与图例
        draw.text((912, 86), "3. 3D 空间摄像头视锥体 (6DoF Camera Frustum) 运动轨迹与稀疏地图重建", font=self.font_h2, fill=(240, 230, 215))
        draw.text((912, 114), "── 红色实线: SLAM 估计轨迹   -- 白色虚线: Vicon 动捕真值   ▲ 金色视锥: 当前摄像头 6DoF 姿态   ● 绿点/射线: 当前共视 3D 点", font=self.font_small, fill=(195, 185, 175))
        pos_txt = f"当前摄像头 3D 坐标: X={aligned_t_wc[idx, 0]:+.2f}m, Y={aligned_t_wc[idx, 1]:+.2f}m, Z={aligned_t_wc[idx, 2]:+.2f}m"
        draw.text((912, 656), pos_txt, font=self.font_body_bold, fill=(255, 215, 90))

        # 右下面板标题
        draw.text((912, 714), "4A. 俯视平面 (X-Y) 摄像头视场角扫掠 (FOV Cone)", font=self.font_h2, fill=(240, 230, 215))
        draw.text((1395, 714), "4B. 三轴位置随时间跟踪曲线 (实线 Est vs 虚线 GT)", font=self.font_h2, fill=(240, 230, 215))
        draw.text((1410, 762), "■ 红: X轴(m)   ■ 绿: Y轴(m)   ■ 橙: Z轴(m)", font=self.font_tiny, fill=(220, 220, 220))

        # 左下面板：动态 SLAM 算法状态流程图文字
        self._draw_pipeline_flowchart_texts(draw, rec)

        return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)

    def _draw_pipeline_flowchart_boxes(
        self,
        canvas: np.ndarray,
        rec: FrameVisRecord,
        idx: int,
        total_frames: int,
        records: list[FrameVisRecord],
    ) -> None:
        """在左下面板绘制动态高亮的算法流程图节点框与连接箭头"""
        # 节点坐标定义 (x1, y1, x2, y2)
        b_input = (36, 640, 240, 735)
        b_imu = (36, 765, 240, 860)

        b_path_a = (285, 640, 565, 735)   # Path A: 金字塔 LK 光流快速路径
        b_path_b = (285, 765, 565, 860)   # Path B: 四叉树 ORB + 零均值 SAD 双目匹配

        b_moba = (610, 640, 860, 735)     # 局部地图投影 + 4轮鲁棒 Motion-only BA
        b_localba = (610, 765, 860, 860)  # 后端局部建图 (Schur 消元 Local BA + 冗余剔除)

        def draw_node(box: tuple[int, int, int, int], active: bool, active_bgr: tuple[int, int, int]) -> None:
            x1, y1, x2, y2 = box
            if active:
                # 激活态发光底色与高亮粗边框
                fill = (int(active_bgr[0] * 0.25 + 30), int(active_bgr[1] * 0.25 + 25), int(active_bgr[2] * 0.25 + 20))
                cv2.rectangle(canvas, (x1, y1), (x2, y2), fill, -1)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), active_bgr, 3)
            else:
                # 待机态暗色框
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (42, 36, 30), -1)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (85, 75, 65), 1)

        path_a_active = (not rec.is_keyframe) and rec.used_optical_flow
        path_b_active = rec.is_keyframe or (not rec.used_optical_flow)
        local_ba_active = rec.is_keyframe and (rec.backend_ms > 1.0)

        draw_node(b_input, True, (255, 190, 60))
        draw_node(b_imu, True, (255, 190, 60))
        draw_node(b_path_a, path_a_active, (100, 245, 120))
        draw_node(b_path_b, path_b_active, (0, 180, 255))
        draw_node(b_moba, True, (230, 220, 70))
        draw_node(b_localba, local_ba_active, (180, 90, 255))

        # 绘制流程连接箭头
        c_act_a = (100, 245, 120) if path_a_active else (90, 80, 70)
        c_act_b = (0, 180, 255) if path_b_active else (90, 80, 70)
        c_act_ba = (180, 90, 255) if local_ba_active else (90, 80, 70)

        # Input -> IMU
        cv2.arrowedLine(canvas, (138, 735), (138, 765), (255, 190, 60), 2, cv2.LINE_AA, tipLength=0.3)
        # Input/IMU -> Path A & Path B
        cv2.arrowedLine(canvas, (240, 687), (285, 687), c_act_a, 2, cv2.LINE_AA, tipLength=0.25)
        cv2.arrowedLine(canvas, (240, 812), (285, 812), c_act_b, 2, cv2.LINE_AA, tipLength=0.25)
        # Path A & Path B -> Motion-only BA
        cv2.arrowedLine(canvas, (565, 687), (610, 687), c_act_a, 2, cv2.LINE_AA, tipLength=0.25)
        cv2.arrowedLine(canvas, (565, 812), (610, 710), c_act_b, 2, cv2.LINE_AA, tipLength=0.15)
        # Motion-only BA -> Local Mapping
        cv2.arrowedLine(canvas, (735, 735), (735, 765), c_act_ba, 2, cv2.LINE_AA, tipLength=0.3)

        # 底部：逐帧实时耗时波形条 (x: 36..860, y: 905..1045)
        rx0, ry0, rw, rh = 36, 905, 824, 140
        cv2.rectangle(canvas, (rx0, ry0), (rx0 + rw, ry0 + rh), (22, 18, 15), -1)
        cv2.rectangle(canvas, (rx0, ry0), (rx0 + rw, ry0 + rh), (65, 55, 45), 1)

        max_ms = 220.0
        # 33.3ms (30 FPS) 实时预算基准线
        y_30fps = int(round(ry0 + rh - 10 - (33.3 / max_ms) * (rh - 24)))
        cv2.line(canvas, (rx0 + 5, y_30fps), (rx0 + rw - 5, y_30fps), (70, 70, 230), 1, cv2.LINE_AA)

        bar_w = max(2, int(rw / max(1, total_frames)) - 1)
        for k_t in range(idx + 1):
            r_k = records[k_t]
            px = int(round(rx0 + 8 + (k_t / max(1, total_frames - 1)) * (rw - 20)))
            fe_h = int(np.clip((r_k.frontend_ms / max_ms) * (rh - 24), 2, rh - 16))
            c_bar = (0, 180, 255) if r_k.is_keyframe else (100, 230, 120)
            cv2.rectangle(canvas, (px, ry0 + rh - 10 - fe_h), (px + bar_w, ry0 + rh - 10), c_bar, -1)

    def _draw_pipeline_flowchart_texts(self, draw: ImageDraw.ImageDraw, rec: FrameVisRecord) -> None:
        """在左下面板流程图框内绘制文字说明"""
        draw.text((32, 598), "2. 实时算法执行状态流程图 (高亮当前帧激活算子与耗时分布)", font=self.font_h2, fill=(240, 230, 215))

        # 节点 1: 双目输入
        draw.text((48, 650), "① 双目图像极线校正", font=self.font_body_bold, fill=(255, 220, 130))
        draw.text((48, 676), "cv::remap 去畸变共面", font=self.font_tiny, fill=(210, 205, 195))
        draw.text((48, 696), "2D 结构张量抑制伪角点", font=self.font_tiny, fill=(210, 205, 195))

        # 节点 2: IMU 预积分
        draw.text((48, 775), "② Forster IMU 预积分", font=self.font_body_bold, fill=(255, 220, 130))
        draw.text((48, 801), "200Hz 陀螺仪 ΔR 先验", font=self.font_tiny, fill=(210, 205, 195))
        draw.text((48, 821), "预测当前帧初值 T_cw", font=self.font_tiny, fill=(210, 205, 195))

        # 分支 A: 光流快速路径
        path_a_active = (not rec.is_keyframe) and rec.used_optical_flow
        col_a = (130, 255, 150) if path_a_active else (140, 135, 130)
        status_a = "● [当前激活]" if path_a_active else "○ [本帧待机]"
        draw.text((297, 650), f"③-A 普通帧光流快速路径 {status_a}", font=self.font_body_bold, fill=col_a)
        draw.text((297, 676), "金字塔 LK 光流直接跟踪 3D 地图点", font=self.font_tiny, fill=col_a)
        draw.text((297, 696), "省去全图 ORB 提取 (耗时 ~33ms)", font=self.font_tiny, fill=col_a)

        # 分支 B: 关键帧特征路径
        path_b_active = rec.is_keyframe or (not rec.used_optical_flow)
        col_b = (255, 200, 70) if path_b_active else (140, 135, 130)
        status_b = "● [当前激活]" if path_b_active else "○ [本帧待机]"
        draw.text((297, 775), f"③-B 关键帧特征提取路径 {status_b}", font=self.font_body_bold, fill=col_b)
        draw.text((297, 801), "8层金字塔四叉树 ORB + 零均值SAD双目", font=self.font_tiny, fill=col_b)
        draw.text((297, 821), "补充新视角高各向同性 3D 特征点", font=self.font_tiny, fill=col_b)

        # 节点 4: Motion-only BA
        draw.text((622, 650), "④ 局部地图 + Motion-only BA", font=self.font_body_bold, fill=(130, 240, 245))
        draw.text((622, 676), "4轮 Huber 鲁棒核 + 卡方外点剔除", font=self.font_tiny, fill=(215, 225, 225))
        draw.text((622, 696), f"解析雅可比优化 6DoF (内点: {rec.inlier_count})", font=self.font_tiny, fill=(215, 225, 225))

        # 节点 5: Local Mapping
        local_ba_active = rec.is_keyframe and (rec.backend_ms > 1.0)
        col_ba = (255, 140, 255) if local_ba_active else (140, 135, 130)
        status_ba = f"● [触发建图 {rec.backend_ms:.0f}ms]" if local_ba_active else "○ [仅关键帧触发]"
        draw.text((622, 775), f"⑤ 后端局部建图 {status_ba}", font=self.font_body_bold, fill=col_ba)
        draw.text((622, 801), "共视图更新 + 双目卡方校验新建点", font=self.font_tiny, fill=col_ba)
        draw.text((622, 821), "稀疏舒尔补 Local BA + 90%冗余剔除", font=self.font_tiny, fill=col_ba)

        # 底部耗时条图例
        draw.text(
            (46, 878),
            f"逐帧前端耗时实时监控 (当前帧: {rec.frontend_ms:.1f} ms)   |   ■ 绿柱: 普通帧光流跟踪   ■ 橙柱: 关键帧 ORB 提取   -- 红虚线: 30 FPS 实时线 (33.3ms)",
            font=self.font_tiny,
            fill=(210, 205, 195),
        )

    def _save_storyboard_image(
        self,
        snapshots: list[tuple[int, np.ndarray]],
        out_path: str,
        ate_metrics: dict[str, float],
    ) -> None:
        """将 6 个关键阶段的完整仪表盘画面拼接成一张高清运动过程分镜大图"""
        if len(snapshots) == 0:
            return

        # 缩略图尺寸 940 x 528，排成 3行 x 2列
        tw, th = 940, 528
        header_h = 72
        grid_w = tw * 2 + 30
        grid_h = header_h + th * 3 + 40
        board = np.full((grid_h, grid_w, 3), (20, 16, 12), dtype=np.uint8)

        for i, (f_idx, img_bgr) in enumerate(snapshots[:6]):
            r = i // 2
            c = i % 2
            x0 = 10 + c * (tw + 10)
            y0 = header_h + r * (th + 10)
            thumb = cv2.resize(img_bgr, (tw, th), interpolation=cv2.INTER_AREA)
            cv2.rectangle(thumb, (0, 0), (tw - 1, th - 1), (110, 95, 80), 2)
            board[y0 : y0 + th, x0 : x0 + tw] = thumb

        pil_board = Image.fromarray(cv2.cvtColor(board, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_board)
        title = (
            f"SLAM 摄像头 6DoF 运动全过程分镜流程图 (序列 ATE RMSE = {ate_metrics['aligned_rmse_m']*100:.2f} cm, "
            f"中位数误差 = {ate_metrics['aligned_median_m']*100:.2f} cm)"
        )
        draw.text((24, 20), title, font=self.font_title, fill=(255, 235, 180))
        final_bgr = cv2.cvtColor(np.asarray(pil_board), cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, final_bgr)


class Live3DTrajectorySketch:
    """
    轻量级高帧率【实时 3D 轨迹与 6DoF 相机位姿简图渲染器】
    专为实机双目+IMU 实时运行 (--live) 设计，单帧渲染耗时 < 2.5 ms：
      - 左区 (520x500): 3D 透视网格、3D 相机运动轨迹、关键帧、稀疏地图点云与实时 6DoF 摄像头视锥体 (Frustum)
      - 中区 (380x500): X-Z 俯视平面雷达简图 (Top-Down Mini-Map) + 实时相机视场角扇形 (FOV Cone)
      - 右区 (380x500): 3D 姿态地平仪 (Roll/Pitch/Yaw) + X/Y/Z 三轴实时位移滚动曲线与传感器状态栏
    支持与 1280x480 实时双目画面上下拼接为 1280x980 一体化监控大屏，并实时输出高清简图文件。
    """

    def __init__(self, width: int = 1280, height: int = 500) -> None:
        self.width = int(width)
        self.height = int(height)
        self.yaw_deg = -35.0
        self.pitch_deg = 26.0
        self.traj_history: list[np.ndarray] = []
        self.rot_history: list[np.ndarray] = []
        self.kf_positions: list[np.ndarray] = []
        self.euler_history: list[np.ndarray] = []
        self.ts_history: list[float] = []

    def reset(self) -> None:
        self.traj_history.clear()
        self.rot_history.clear()
        self.kf_positions.clear()
        self.euler_history.clear()
        self.ts_history.clear()

    def handle_key(self, key: int) -> bool:
        """处理键盘交互: A/D 旋转 3D 视角偏航角, W/S 调整俯仰角, R 重置轨迹, Q 退出"""
        if key in (ord("q"), ord("Q")):
            return False
        elif key in (ord("a"), ord("A")):
            self.yaw_deg -= 8.0
        elif key in (ord("d"), ord("D")):
            self.yaw_deg += 8.0
        elif key in (ord("w"), ord("W")):
            self.pitch_deg = min(75.0, self.pitch_deg + 6.0)
        elif key in (ord("s"), ord("S")):
            self.pitch_deg = max(-20.0, self.pitch_deg - 6.0)
        elif key in (ord("r"), ord("R")):
            self.reset()
        return True

    def _project_3d(
        self,
        pts_w: np.ndarray,
        center_w: np.ndarray,
        scale_px: float,
        cx: float,
        cy: float,
    ) -> np.ndarray:
        """将世界系 (X右, Y下, Z前) 3D 点投影到 2D 轴测透视画布"""
        if len(pts_w) == 0:
            return np.empty((0, 2), dtype=np.int32)
        rel = pts_w - center_w.reshape(1, 3)
        # 转换为视觉直观的 (X右, Z前深, -Y上高) 观察空间
        X = rel[:, 0]
        Depth = rel[:, 2]
        Up = -rel[:, 1]

        yaw = np.deg2rad(self.yaw_deg)
        pitch = np.deg2rad(self.pitch_deg)
        cy_a, sy_a = np.cos(yaw), np.sin(yaw)
        cp_a, sp_a = np.cos(pitch), np.sin(pitch)

        x1 = cy_a * X + sy_a * Depth
        z1 = -sy_a * X + cy_a * Depth
        y1 = Up

        u_3d = x1
        v_3d = -sp_a * z1 + cp_a * y1
        d_cam = cp_a * z1 + sp_a * y1 + 6.0
        persp = 6.0 / np.clip(d_cam, 1.8, 25.0)

        u_px = cx + u_3d * scale_px * persp
        v_px = cy - v_3d * scale_px * persp
        return np.round(np.column_stack([u_px, v_px])).astype(np.int32)

    def update_and_render(
        self,
        frame_idx: int,
        timestamp: float,
        R_wc: np.ndarray,
        t_wc: np.ndarray,
        map_points_w: np.ndarray,
        is_keyframe: bool,
        tracking_state_name: str,
        num_kfs: int,
        num_mps: int,
        inliers: int,
        imu_acc: np.ndarray,
        imu_gyro: np.ndarray,
        imu_euler_deg: np.ndarray,
        fps: float,
        stereo_top_hud: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        更新当前帧 6DoF 位姿并渲染 1280x500 实时 3D 轨迹简图面板
        (若传入 stereo_top_hud 则自动拼接为上双目画面 + 下 3D 简图的一体化画布)
        """
        pos = np.asarray(t_wc, dtype=np.float64).reshape(3).copy()
        rot = np.asarray(R_wc, dtype=np.float64).reshape(3, 3).copy()
        eul = np.asarray(imu_euler_deg, dtype=np.float64).reshape(3).copy()

        self.traj_history.append(pos)
        self.rot_history.append(rot)
        self.euler_history.append(eul)
        self.ts_history.append(float(timestamp))
        if is_keyframe:
            self.kf_positions.append(pos)

        traj_arr = np.asarray(self.traj_history, dtype=np.float64)
        panel = np.full((self.height, self.width, 3), (22, 24, 29), dtype=np.uint8)

        # 三大子面板边框与标题栏
        # 左: [10, 520], 中: [530, 900], 右: [910, 1270]
        boxes = [
            (10, 10, 520, self.height - 10, "3D Live Trajectory & 6DoF Camera Frustum [A/D/W/S Orbit]"),
            (530, 10, 900, self.height - 10, "Top-Down X-Z Mini-Map & FOV Cone"),
            (910, 10, 1270, self.height - 10, "3D Attitude Horizon & XYZ Position Curves"),
        ]
        for x0, y0, x1, y1, title in boxes:
            cv2.rectangle(panel, (x0, y0), (x1, y1), (42, 46, 56), -1)
            cv2.rectangle(panel, (x0, y0), (x1, y1), (85, 95, 115), 1)
            cv2.rectangle(panel, (x0, y0), (x1, y0 + 28), (35, 50, 72), -1)
            cv2.putText(panel, title, (x0 + 10, y0 + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 235, 255), 1, cv2.LINE_AA)

        # ====================================================================
        # 1. 左面板: 3D 透视网格 + 3D 轨迹 + 地图点 + 6DoF 相机视锥体
        # ====================================================================
        cx_3d, cy_3d = 265.0, 275.0
        span_3d = float(np.max(np.ptp(traj_arr, axis=0))) if len(traj_arr) > 1 else 0.0
        half_extent = max(0.35, min(6.0, span_3d * 0.85 + 0.25))
        scale_3d = 175.0 / half_extent
        center_3d = traj_arr.mean(axis=0) if len(traj_arr) > 1 else np.zeros(3, dtype=np.float64)

        # 1a. 绘制 3D 水平基准网格 (Y = center_y + 0.15m 地面线)
        grid_y = float(center_3d[1]) + half_extent * 0.35
        n_grid = 5
        grid_vals = np.linspace(-half_extent, half_extent, n_grid * 2 + 1)
        for gv in grid_vals:
            p_line1 = np.array([
                [center_3d[0] - half_extent, grid_y, center_3d[2] + gv],
                [center_3d[0] + half_extent, grid_y, center_3d[2] + gv],
            ])
            p_line2 = np.array([
                [center_3d[0] + gv, grid_y, center_3d[2] - half_extent],
                [center_3d[0] + gv, grid_y, center_3d[2] + half_extent],
            ])
            uv1 = self._project_3d(p_line1, center_3d, scale_3d, cx_3d, cy_3d)
            uv2 = self._project_3d(p_line2, center_3d, scale_3d, cx_3d, cy_3d)
            cv2.line(panel, tuple(uv1[0]), tuple(uv1[1]), (55, 62, 74), 1, cv2.LINE_AA)
            cv2.line(panel, tuple(uv2[0]), tuple(uv2[1]), (55, 62, 74), 1, cv2.LINE_AA)

        # 1b. 绘制世界系原点坐标轴 O-XYZ
        axis_len = half_extent * 0.35
        origin_pts = np.array([
            [0.0, 0.0, 0.0],
            [axis_len, 0.0, 0.0],
            [0.0, -axis_len, 0.0],
            [0.0, 0.0, axis_len],
        ], dtype=np.float64)
        uv_orig = self._project_3d(origin_pts, center_3d, scale_3d, cx_3d, cy_3d)
        o_pt = tuple(uv_orig[0])
        if 15 < o_pt[0] < 515 and 42 < o_pt[1] < self.height - 15:
            cv2.arrowedLine(panel, o_pt, tuple(uv_orig[1]), (70, 70, 240), 2, cv2.LINE_AA, tipLength=0.2)
            cv2.arrowedLine(panel, o_pt, tuple(uv_orig[2]), (70, 220, 70), 2, cv2.LINE_AA, tipLength=0.2)
            cv2.arrowedLine(panel, o_pt, tuple(uv_orig[3]), (240, 140, 50), 2, cv2.LINE_AA, tipLength=0.2)
            cv2.putText(panel, "X", tuple(uv_orig[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 255), 1, cv2.LINE_AA)
            cv2.putText(panel, "Y", tuple(uv_orig[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 240, 90), 1, cv2.LINE_AA)
            cv2.putText(panel, "Z", tuple(uv_orig[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 170, 70), 1, cv2.LINE_AA)

        # 1c. 绘制 3D 稀疏地图点云
        if len(map_points_w) > 0:
            dists = np.linalg.norm(map_points_w - center_3d.reshape(1, 3), axis=1)
            valid_mp = map_points_w[dists < half_extent * 2.8]
            if len(valid_mp) > 400:
                valid_mp = valid_mp[:: max(1, len(valid_mp) // 400)]
            if len(valid_mp) > 0:
                uv_mp = self._project_3d(valid_mp, center_3d, scale_3d, cx_3d, cy_3d)
                for u_p, v_p in uv_mp:
                    if 14 <= u_p <= 516 and 42 <= v_p <= self.height - 14:
                        cv2.circle(panel, (int(u_p), int(v_p)), 2, (180, 195, 120), -1)

        # 1d. 绘制 3D 历史轨迹曲线 (渐变色)
        uv_traj = self._project_3d(traj_arr, center_3d, scale_3d, cx_3d, cy_3d)
        n_t = len(uv_traj)
        for i in range(1, n_t):
            p0 = (int(np.clip(uv_traj[i - 1, 0], 14, 516)), int(np.clip(uv_traj[i - 1, 1], 42, self.height - 14)))
            p1 = (int(np.clip(uv_traj[i, 0], 14, 516)), int(np.clip(uv_traj[i, 1], 42, self.height - 14)))
            ratio = float(i) / float(max(1, n_t - 1))
            col = (int(60 * (1.0 - ratio)), int(220 * (1.0 - 0.5 * ratio)), int(80 + 175 * ratio))
            cv2.line(panel, p0, p1, col, 2, cv2.LINE_AA)

        # 起点标记
        p_start = (int(np.clip(uv_traj[0, 0], 16, 514)), int(np.clip(uv_traj[0, 1], 44, self.height - 16)))
        cv2.circle(panel, p_start, 5, (0, 230, 100), -1, cv2.LINE_AA)

        # 1e. 绘制当前帧 3D 6DoF 相机视锥体 (Camera Frustum Pyramid)
        f_sz = half_extent * 0.24
        fw, fh, fz = f_sz * 0.75, f_sz * 0.50, f_sz * 1.15
        frustum_c = np.array([
            [0.0, 0.0, 0.0],
            [-fw, -fh, fz],
            [fw, -fh, fz],
            [fw, fh, fz],
            [-fw, fh, fz],
            [0.0, 0.0, fz * 1.6],  # 光轴指向箭头端点
        ], dtype=np.float64)
        frustum_w = (frustum_c @ rot.T) + pos.reshape(1, 3)
        uv_f = self._project_3d(frustum_w, center_3d, scale_3d, cx_3d, cy_3d)
        uv_f[:, 0] = np.clip(uv_f[:, 0], 14, 516)
        uv_f[:, 1] = np.clip(uv_f[:, 1], 42, self.height - 14)

        fc_origin = tuple(uv_f[0])
        c1, c2, c3, c4 = tuple(uv_f[1]), tuple(uv_f[2]), tuple(uv_f[3]), tuple(uv_f[4])
        for corner in (c1, c2, c3, c4):
            cv2.line(panel, fc_origin, corner, (0, 215, 255), 2, cv2.LINE_AA)
        cv2.line(panel, c1, c2, (0, 180, 255), 2, cv2.LINE_AA)
        cv2.line(panel, c2, c3, (0, 180, 255), 2, cv2.LINE_AA)
        cv2.line(panel, c3, c4, (0, 180, 255), 2, cv2.LINE_AA)
        cv2.line(panel, c4, c1, (0, 180, 255), 2, cv2.LINE_AA)
        cv2.arrowedLine(panel, fc_origin, tuple(uv_f[5]), (80, 100, 255), 2, cv2.LINE_AA, tipLength=0.25)

        # 左面板底部比例尺与坐标信息
        dist_total = float(np.sum(np.linalg.norm(np.diff(traj_arr, axis=0), axis=1))) if len(traj_arr) > 1 else 0.0
        cv2.putText(
            panel,
            f"Pos: X={pos[0]:+.3f}m, Y={pos[1]:+.3f}m, Z={pos[2]:+.3f}m | Travel={dist_total*100:.1f}cm",
            (20, self.height - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (200, 220, 240),
            1,
            cv2.LINE_AA,
        )

        # ====================================================================
        # 2. 中面板: X-Z 俯视平面雷达简图 (Top-Down Bird's-Eye Mini-Map)
        # ====================================================================
        cx_td, cy_td = 715, 265
        radius_px = 165
        # 同心圆雷达网格与十字准星
        for r_ring in (55, 110, 165):
            cv2.circle(panel, (cx_td, cy_td), r_ring, (58, 65, 78), 1, cv2.LINE_AA)
        cv2.line(panel, (cx_td - radius_px, cy_td), (cx_td + radius_px, cy_td), (58, 65, 78), 1)
        cv2.line(panel, (cx_td, cy_td - radius_px), (cx_td, cy_td + radius_px), (58, 65, 78), 1)

        xz_pts = traj_arr[:, [0, 2]]
        xz_center = xz_pts.mean(axis=0) if len(xz_pts) > 1 else np.zeros(2)
        xz_span = max(0.30, float(np.max(np.ptp(xz_pts, axis=0))) * 1.3 + 0.15)
        scale_td = (radius_px * 0.82) / (0.5 * xz_span)

        def _to_td_px(x_val: float, z_val: float) -> tuple[int, int]:
            u = int(np.clip(cx_td + (x_val - xz_center[0]) * scale_td, 538, 892))
            v = int(np.clip(cy_td - (z_val - xz_center[1]) * scale_td, 46, self.height - 20))
            return u, v

        # 绘制俯视地图点
        if len(map_points_w) > 0:
            for mp_pt in map_points_w[:: max(1, len(map_points_w) // 250)]:
                u_m, v_m = _to_td_px(float(mp_pt[0]), float(mp_pt[2]))
                cv2.circle(panel, (u_m, v_m), 1, (130, 145, 110), -1)

        # 绘制当前相机视场角扇区 (FOV Cone)
        cam_cur_td = _to_td_px(float(pos[0]), float(pos[2]))
        forward_w = rot @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right_w = rot @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        fov_dir = np.array([forward_w[0], forward_w[2]], dtype=np.float64)
        if np.linalg.norm(fov_dir) < 1e-3:
            fov_dir = np.array([0.0, 1.0], dtype=np.float64)
        else:
            fov_dir /= np.linalg.norm(fov_dir)
        fov_perp = np.array([-fov_dir[1], fov_dir[0]], dtype=np.float64)
        cone_len = 52.0
        cone_p1 = (
            int(np.clip(cam_cur_td[0] + (fov_dir[0] * cone_len - fov_perp[0] * 28.0), 538, 892)),
            int(np.clip(cam_cur_td[1] - (fov_dir[1] * cone_len - fov_perp[1] * 28.0), 46, self.height - 20)),
        )
        cone_p2 = (
            int(np.clip(cam_cur_td[0] + (fov_dir[0] * cone_len + fov_perp[0] * 28.0), 538, 892)),
            int(np.clip(cam_cur_td[1] - (fov_dir[1] * cone_len + fov_perp[1] * 28.0), 46, self.height - 20)),
        )
        overlay = panel.copy()
        cv2.fillConvexPoly(overlay, np.array([cam_cur_td, cone_p1, cone_p2], dtype=np.int32), (0, 190, 255))
        cv2.addWeighted(overlay, 0.25, panel, 0.75, 0, panel)
        cv2.line(panel, cam_cur_td, cone_p1, (0, 215, 255), 1, cv2.LINE_AA)
        cv2.line(panel, cam_cur_td, cone_p2, (0, 215, 255), 1, cv2.LINE_AA)

        # 绘制俯视轨迹线
        td_pts = [_to_td_px(float(p[0]), float(p[2])) for p in traj_arr]
        for i in range(1, len(td_pts)):
            cv2.line(panel, td_pts[i - 1], td_pts[i], (60, 210, 255), 2, cv2.LINE_AA)
        cv2.circle(panel, td_pts[0], 5, (0, 230, 100), -1, cv2.LINE_AA)
        cv2.circle(panel, cam_cur_td, 5, (40, 60, 255), -1, cv2.LINE_AA)

        cv2.putText(
            panel,
            f"Scale Ring: {xz_span*33.3:.1f} cm/ring | KFs={num_kfs} | MPs={num_mps}",
            (542, self.height - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (200, 220, 240),
            1,
            cv2.LINE_AA,
        )

        # ====================================================================
        # 3. 右面板: 3D 姿态地平仪 (Artificial Horizon) + 三轴实时曲线
        # ====================================================================
        # 3a. 姿态球 (中心: (1005, 145), 半径 68)
        ah_cx, ah_cy, ah_r = 1000, 142, 64
        cv2.circle(panel, (ah_cx, ah_cy), ah_r, (32, 38, 48), -1, cv2.LINE_AA)
        roll_rad = np.deg2rad(eul[0])
        pitch_px = float(np.clip(np.sin(np.deg2rad(eul[1])) * ah_r * 0.85, -ah_r * 0.75, ah_r * 0.75))
        dx_h = int(np.cos(roll_rad) * (ah_r - 4))
        dy_h = int(np.sin(roll_rad) * (ah_r - 4))
        h_p1 = (ah_cx - dx_h, int(np.clip(ah_cy + pitch_px - dy_h, ah_cy - ah_r, ah_cy + ah_r)))
        h_p2 = (ah_cx + dx_h, int(np.clip(ah_cy + pitch_px + dy_h, ah_cy - ah_r, ah_cy + ah_r)))
        cv2.line(panel, h_p1, h_p2, (0, 220, 120), 2, cv2.LINE_AA)
        cv2.circle(panel, (ah_cx, ah_cy), ah_r, (110, 130, 160), 2, cv2.LINE_AA)
        cv2.line(panel, (ah_cx - 12, ah_cy), (ah_cx + 12, ah_cy), (0, 215, 255), 2)
        cv2.line(panel, (ah_cx, ah_cy - 8), (ah_cx, ah_cy + 8), (0, 215, 255), 2)

        # 右侧传感器与状态读数
        st_col = (0, 230, 110) if tracking_state_name == "OK" else (0, 180, 255)
        cv2.putText(panel, f"State: {tracking_state_name}", (1080, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.48, st_col, 1, cv2.LINE_AA)
        cv2.putText(panel, f"Frame: {frame_idx + 1} ({fps:.1f} FPS)", (1080, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 225, 235), 1, cv2.LINE_AA)
        cv2.putText(panel, f"Inliers: {inliers}", (1080, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 225, 235), 1, cv2.LINE_AA)
        cv2.putText(panel, f"Roll : {eul[0]:+6.1f} deg", (1080, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 210, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, f"Pitch: {eul[1]:+6.1f} deg", (1080, 176), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 255, 160), 1, cv2.LINE_AA)
        cv2.putText(panel, f"Yaw  : {eul[2]:+6.1f} deg", (1080, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 180, 120), 1, cv2.LINE_AA)
        cv2.putText(panel, f"|g|  : {np.linalg.norm(imu_acc):.2f} m/s2", (1080, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 220), 1, cv2.LINE_AA)

        # 3b. 下半部分: X / Y / Z 三轴位移实时滚动曲线框 [924, 245] -> [1256, 455]
        gx0, gy0, gx1, gy1 = 925, 245, 1255, self.height - 38
        cv2.rectangle(panel, (gx0, gy0), (gx1, gy1), (28, 32, 40), -1)
        cv2.rectangle(panel, (gx0, gy0), (gx1, gy1), (70, 78, 92), 1)
        gy_mid = (gy0 + gy1) // 2
        cv2.line(panel, (gx0, gy_mid), (gx1, gy_mid), (55, 62, 74), 1)

        recent_pos = traj_arr[-120:]
        max_abs_xyz = max(0.05, float(np.max(np.abs(recent_pos))) * 1.15)
        n_rec = len(recent_pos)
        if n_rec >= 2:
            xs_px = np.linspace(gx0 + 6, gx1 - 6, n_rec).astype(np.int32)
            axis_colors = [(70, 80, 245), (70, 220, 90), (245, 155, 60)]  # X(Red), Y(Green), Z(Blue)
            for ax_i, col in enumerate(axis_colors):
                ys_px = np.clip(
                    gy_mid - (recent_pos[:, ax_i] / max_abs_xyz) * ((gy1 - gy0) * 0.42),
                    gy0 + 4,
                    gy1 - 4,
                ).astype(np.int32)
                pts_curve = np.column_stack([xs_px, ys_px]).reshape(-1, 1, 2)
                cv2.polylines(panel, [pts_curve], False, col, 2, cv2.LINE_AA)

        cv2.putText(
            panel,
            " Curves: X(Red)  Y(Green)  Z(Blue)  |  [Q] Quit [R] Reset",
            (922, self.height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (190, 205, 225),
            1,
            cv2.LINE_AA,
        )

        if stereo_top_hud is not None:
            if stereo_top_hud.shape[1] != self.width:
                h_top = int(stereo_top_hud.shape[0] * (self.width / stereo_top_hud.shape[1]))
                stereo_top_hud = cv2.resize(stereo_top_hud, (self.width, h_top))
            return np.vstack([stereo_top_hud, panel])
        return panel

