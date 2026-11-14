# -*- coding: utf-8 -*-
"""
frontend/stereo_matcher.py
==========================
双目极线匹配器、帧间光流跟踪器与 ORB 描述子投影匹配器 (StereoMatcher & ORBMatcher)
严格 1:1 对齐 ORB-SLAM3 Frame::ComputeStereoMatches 与 ORBmatcher，
并集成针对 RK3566 Cortex-A55 优化的双向金字塔 LK 光流快速跟踪路径。
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import cv2

if TYPE_CHECKING:
    from ..core.camera import StereoCamera
    from ..core.frame import Frame
    from ..core.keyframe import KeyFrame
    from ..core.mappoint import MapPoint


# 预计算 0~255 字节的二进制 1 比特计数查表 (对应 C++ ARM NEON vcntq_u8 汉明距离加速)
_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


def descriptor_distance(d1: np.ndarray, d2: np.ndarray) -> int:
    """
    计算两个 256-bit (32 bytes) ORB 描述子的汉明距离 (0 ~ 256)
    对应 ORB-SLAM3 ORBmatcher::DescriptorDistance
    """
    return int(_POPCOUNT_TABLE[np.bitwise_xor(d1, d2)].sum())


def descriptor_distance_batch(d1: np.ndarray, d2_mat: np.ndarray) -> np.ndarray:
    """
    计算单个描述子 d1 (32,) 与一组描述子 d2_mat (M, 32) 的汉明距离向量 (M,)
    """
    xor_res = np.bitwise_xor(d2_mat, d1[None, :])
    return _POPCOUNT_TABLE[xor_res].sum(axis=1)


class StereoMatcher:
    """
    提供:
    1. 双目极线亚像素立体匹配 (compute_stereo_matches)
    2. 帧间双向金字塔 LK 光流跟踪 (track_optical_flow) —— RK3566 普通帧高速路径
    3. 局部地图点投影匹配 (search_by_projection)
    4. 共视关键帧对极几何三角化匹配 (search_for_triangulation)
    """
    TH_HIGH: int = 100
    TH_LOW: int = 50

    @staticmethod
    def compute_stereo_matches(
        img_left: np.ndarray,
        img_right: np.ndarray,
        kp_pts: np.ndarray,
        camera: StereoCamera,
        min_disparity: float = 2.0,
        max_disparity: float = 80.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        对左目特征点 kp_pts (N, 2) 计算右目极线匹配横坐标 u_right 与物理深度 depths
        严格对齐 ORB-SLAM3 Frame::ComputeStereoMatches 极线扫描 + 归一化块匹配 + 左右一致性校验 + 抛物线亚像素精化:
        1. 对每个左目特征点 (u_l, v_l) 提取 (2R+1)x(2R+1) 局部图像块 (R=5, 即 11x11 窗口)。
        2. 在右目同一水平极线扫描带 [u_l - max_d, u_l - min_d] 上执行 SIMD 加速的 1D 归一化平方差匹配 (TM_SQDIFF_NORMED)，
           天然克服真实双目相机 (如 EuRoC cam0/cam1) 之间的曝光/增益差异以及大视差近点问题。
        3. 执行峰值显著性检验、左右双向一致性校验 (LRCC) 与三点抛物线亚像素插值精化 u_right，由 Z = bf / (u_l - u_right) 恢复深度。
        """
        n_kps = len(kp_pts)
        u_right = np.full(n_kps, -1.0, dtype=np.float32)
        depths = np.full(n_kps, -1.0, dtype=np.float32)
        if n_kps == 0:
            return u_right, depths

        gray_l = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY) if img_left.ndim == 3 else img_left
        gray_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY) if img_right.ndim == 3 else img_right

        h, w = gray_l.shape[:2]
        bf = float(camera.bf)
        half_w = 5  # 11x11 匹配窗口
        max_d_int = int(np.ceil(max_disparity))
        min_d_int = max(1, int(np.floor(min_disparity)))

        for i in range(n_kps):
            ul, vl = float(kp_pts[i, 0]), float(kp_pts[i, 1])
            ui_l, vi_l = int(np.round(ul)), int(np.round(vl))

            if vi_l < half_w or vi_l >= h - half_w or ui_l < half_w + min_d_int or ui_l >= w - half_w:
                continue

            u_r_min = max(half_w, ui_l - max_d_int)
            u_r_max = ui_l - min_d_int
            if u_r_max <= u_r_min + 2:
                continue

            patch_l = gray_l[vi_l - half_w : vi_l + half_w + 1, ui_l - half_w : ui_l + half_w + 1]
            # 二维结构张量检验 (Shi-Tomasi 最小特征值与极线水平梯度): 剔除纯白墙及水平/垂直一维直线边缘 (消除孔径问题与极线退化)
            p_f = patch_l.astype(np.float32)
            gx = p_f[1:-1, 2:] - p_f[1:-1, :-2]
            gy = p_f[2:, 1:-1] - p_f[:-2, 1:-1]
            ixx = float(np.mean(gx * gx))
            iyy = float(np.mean(gy * gy))
            if ixx < 25.0 or iyy < 15.0:
                continue
            ixy = float(np.mean(gx * gy))
            tr = ixx + iyy
            det = ixx * iyy - ixy * ixy
            lambda_min = 0.5 * (tr - float(np.sqrt(max(0.0, tr * tr - 4.0 * det))))
            if lambda_min < 12.0 or lambda_min < 0.055 * tr:
                continue

            strip_r = gray_r[vi_l - half_w : vi_l + half_w + 1, u_r_min - half_w : u_r_max + half_w + 1]
            res = cv2.matchTemplate(strip_r, patch_l, cv2.TM_SQDIFF_NORMED)[0]  # 1D 响应曲线

            best_idx = int(np.argmin(res))
            best_val = float(res[best_idx])
            if best_val > 0.12:
                continue

            # 唯一性校验: 排除周期性重复纹理伪峰 (要求非邻域次优值明显大于最优值)
            if len(res) > 6:
                mask_other = np.ones(len(res), dtype=bool)
                mask_other[max(0, best_idx - 2) : min(len(res), best_idx + 3)] = False
                if np.any(mask_other):
                    second_val = float(np.min(res[mask_other]))
                    if best_val > 0.70 * second_val:
                        continue

            ui_r = u_r_min + best_idx
            ur = float(ui_r)
            # 抛物线亚像素插值精化 (对应 ORB-SLAM3 Frame::ComputeStereoMatches)
            if 0 < best_idx < len(res) - 1:
                c_m1 = float(res[best_idx - 1])
                c_0 = best_val
                c_p1 = float(res[best_idx + 1])
                denom = 2.0 * (c_m1 + c_p1 - 2.0 * c_0)
                if abs(denom) > 1e-6:
                    delta = (c_m1 - c_p1) / denom
                    if abs(delta) < 1.0:
                        ur += delta

            # 亚像素左目坐标小数部分补偿
            ur += (ul - float(ui_l))
            disp = ul - ur
            if min_disparity <= disp <= max_disparity:
                u_right[i] = ur
                depths[i] = bf / disp

        return u_right, depths

    @staticmethod
    def track_optical_flow(
        prev_frame: Frame,
        curr_img_left: np.ndarray,
        curr_img_right: np.ndarray,
        camera: StereoCamera,
        T_cw_pred: object | None = None,
        fb_threshold: float = 1.5
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[MapPoint | None]]:
        """
        RK3566 普通帧快速路径:
        对上一帧中已关联有效 MapPoint 的特征点执行带 IMU/恒速运动先验的双向金字塔 LK 光流跟踪，
        并同步计算当前帧右目双目深度，直接传递 3D-2D/3D 双目地图点匹配关系。
        """
        valid_indices = [
            i for i, mp in enumerate(prev_frame.map_points)
            if mp is not None and not mp.is_bad and not prev_frame.outliers[i]
        ]
        if len(valid_indices) < 15:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                []
            )

        prev_gray = cv2.cvtColor(prev_frame.img_left, cv2.COLOR_BGR2GRAY) if prev_frame.img_left.ndim == 3 else prev_frame.img_left
        curr_gray = cv2.cvtColor(curr_img_left, cv2.COLOR_BGR2GRAY) if curr_img_left.ndim == 3 else curr_img_left

        pts_prev = prev_frame.kp_pts[valid_indices].reshape(-1, 1, 2).astype(np.float32)

        # 利用 IMU / 恒速预测位姿 T_cw_pred 将 3D 地图点重投影至当前帧，作为金字塔光流初始迭代位置 (克服快速旋转与大位移)
        if T_cw_pred is not None:
            pts_init = pts_prev.copy()
            for idx_local, orig_idx in enumerate(valid_indices):
                mp = prev_frame.map_points[orig_idx]
                if mp is not None:
                    p_c = T_cw_pred.transform(mp.pos_w)
                    if p_c[2] > 0.25:
                        uv = camera.project_mono(p_c)
                        if camera.is_in_image(float(uv[0]), float(uv[1]), margin=2.0):
                            pts_init[idx_local, 0, 0] = float(uv[0])
                            pts_init[idx_local, 0, 1] = float(uv[1])
            pts_curr, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, pts_prev, pts_init,
                winSize=(21, 21), maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                flags=cv2.OPTFLOW_USE_INITIAL_FLOW
            )
        else:
            pts_curr, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, pts_prev, None,
                winSize=(21, 21), maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )

        if pts_curr is None or st_fwd is None:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                []
            )

        # 反向光流校验 curr -> prev (剔除遮挡与漂移外点)
        pts_back_init = pts_prev.copy()
        pts_back, st_bwd, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, prev_gray, pts_curr, pts_back_init,
            winSize=(21, 21), maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW
        )

        pts_prev_2d = pts_prev.reshape(-1, 2)
        pts_curr_2d = pts_curr.reshape(-1, 2)
        pts_back_2d = pts_back.reshape(-1, 2)
        fb_err = np.linalg.norm(pts_prev_2d - pts_back_2d, axis=1)

        tracked_pts = []
        tracked_mps: list[MapPoint | None] = []
        for idx_local, orig_idx in enumerate(valid_indices):
            if st_fwd[idx_local, 0] == 1 and st_bwd[idx_local, 0] == 1 and fb_err[idx_local] <= fb_threshold:
                u, v = float(pts_curr_2d[idx_local, 0]), float(pts_curr_2d[idx_local, 1])
                if camera.is_in_image(u, v, margin=4.0):
                    tracked_pts.append((u, v))
                    tracked_mps.append(prev_frame.map_points[orig_idx])

        if len(tracked_pts) == 0:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                []
            )

        kp_pts_arr = np.asarray(tracked_pts, dtype=np.float32)
        if curr_img_right is not curr_img_left:
            curr_gray_r = cv2.cvtColor(curr_img_right, cv2.COLOR_BGR2GRAY) if curr_img_right.ndim == 3 else curr_img_right
            u_right, depths = StereoMatcher.compute_stereo_matches(curr_gray, curr_gray_r, kp_pts_arr, camera)
        else:
            u_right = np.full(len(kp_pts_arr), -1.0, dtype=np.float32)
            depths = np.full(len(kp_pts_arr), -1.0, dtype=np.float32)
        return kp_pts_arr, u_right, depths, tracked_mps

    @classmethod
    def search_by_projection(
        cls,
        frame: Frame,
        local_mappoints: list[MapPoint],
        th_radius: float = 5.0
    ) -> int:
        """
        将局部地图点投影到当前帧，在 64x48 空间网格内进行 ORB 描述子汉明距离匹配
        对应 ORB-SLAM3 ORBmatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints)
        """
        if frame.descriptors is None or frame.num_kps == 0:
            return 0

        added_matches = 0
        for mp in local_mappoints:
            if not mp.track_in_view or mp.is_bad or mp.descriptor is None:
                continue

            pred_level = mp.track_scale_level
            r = th_radius * float(frame.scale_factors[pred_level])
            if mp.track_view_cos < 0.998:
                r *= 1.4

            cand_indices = frame.get_features_in_area(
                mp.track_proj_x, mp.track_proj_y, r,
                min_level= max(0, pred_level - 1),
                max_level= min(len(frame.scale_factors) - 1, pred_level + 1)
            )
            if len(cand_indices) == 0:
                continue

            best_dist = 256
            best_dist2 = 256
            best_idx = -1

            for idx in cand_indices:
                if frame.map_points[idx] is not None:
                    continue
                if frame.u_right[idx] > 0.0:
                    er = abs(mp.track_proj_xr - float(frame.u_right[idx]))
                    if er > r:
                        continue
                dist = descriptor_distance(mp.descriptor, frame.descriptors[idx])
                if dist < best_dist:
                    best_dist2 = best_dist
                    best_dist = dist
                    best_idx = idx
                elif dist < best_dist2:
                    best_dist2 = dist

            if best_dist <= cls.TH_HIGH:
                if best_dist < 0.8 * float(best_dist2):
                    frame.map_points[best_idx] = mp
                    added_matches += 1

        return added_matches

    @classmethod
    def search_for_triangulation(
        cls,
        kf1: KeyFrame,
        kf2: KeyFrame
    ) -> list[tuple[int, int]]:
        """
        在两个共视关键帧之间搜索未匹配的特征点对，利用极线几何约束 (Essential Matrix)
        与 ORB 汉明距离筛选用于三角化新 3D 地图点的匹配对 (idx1, idx2)。
        对应 ORB-SLAM3 ORBmatcher::SearchForTriangulation
        """
        if kf1.descriptors is None or kf2.descriptors is None:
            return []

        # 计算从 kf2 到 kf1 的相对变换与对极极点/本质矩阵
        T_1w = kf1.T_cw
        T_2w = kf2.T_cw
        T_12 = T_1w * T_2w.inverse()
        R_12 = T_12.R
        t_12 = T_12.t

        # 基础矩阵 F_12 = K1^{-T} * [t_12]_x * R_12 * K2^{-1}
        K_inv = np.linalg.inv(kf1.camera.K)
        tx = np.array([
            [0.0, -t_12[2], t_12[1]],
            [t_12[2], 0.0, -t_12[0]],
            [-t_12[1], t_12[0], 0.0]
        ], dtype=np.float64)
        F_12 = K_inv.T @ tx @ R_12 @ K_inv

        unmatched_1 = [i for i in range(kf1.num_kps) if kf1.map_points[i] is None]
        unmatched_2 = [j for j in range(kf2.num_kps) if kf2.map_points[j] is None]
        if len(unmatched_1) == 0 or len(unmatched_2) == 0:
            return []

        # 计算 kf1 光心 o1 在 kf2 像素平面上的对极点 (ex, ey)，对应 ORB-SLAM3 CheckDistEpipolarLine 极点退化剔除
        o1_w = kf1.get_camera_center()
        o1_c2 = T_2w.transform(o1_w)
        has_epipole_in_kf2 = o1_c2[2] > 1e-4
        if has_epipole_in_kf2:
            uv_e2 = kf2.camera.project_mono(o1_c2)
            ex, ey = float(uv_e2[0]), float(uv_e2[1])
        else:
            ex, ey = -1e6, -1e6

        desc1_sub = kf1.descriptors[unmatched_1]
        desc2_sub = kf2.descriptors[unmatched_2]
        # 一次性向量化计算 N1 x N2 汉明距离矩阵，省去 Python 逐点全量广播与完整排序开销
        dist_mat = _POPCOUNT_TABLE[desc1_sub[:, None, :] ^ desc2_sub[None, :, :]].sum(axis=-1)
        used_2 = set()
        matches: list[tuple[int, int]] = []

        for row_i, idx1 in enumerate(unmatched_1):
            lvl1 = int(kf1.kp_octaves[idx1])
            u1, v1 = float(kf1.kp_pts[idx1, 0]), float(kf1.kp_pts[idx1, 1])
            p1_h = np.array([u1, v1, 1.0], dtype=np.float64)

            dists = dist_mat[row_i]
            k_top = min(5, len(dists))
            cand_k = np.argpartition(dists, k_top - 1)[:k_top]
            sorted_order = cand_k[np.argsort(dists[cand_k])]

            for k in sorted_order:
                dist = int(dists[k])
                if dist > cls.TH_LOW:
                    break
                idx2 = unmatched_2[int(k)]
                if idx2 in used_2:
                    continue
                lvl2 = int(kf2.kp_octaves[idx2])
                if abs(lvl1 - lvl2) > 2:
                    continue

                u2, v2 = float(kf2.kp_pts[idx2, 0]), float(kf2.kp_pts[idx2, 1])
                # 对齐 ORB-SLAM3 ORBmatcher::SearchForTriangulation: 剔除距离对极点过近 (<10px * scale) 的退化射线对
                if has_epipole_in_kf2 and kf2.u_right[idx2] < 0.0:
                    dist_epipole_sq = (u2 - ex) ** 2 + (v2 - ey) ** 2
                    if dist_epipole_sq < 100.0 * float(kf2.scale_factors[lvl2]):
                        continue

                # 极线几何距离校验: p1^T * F_12 * p2 = 0
                l1 = F_12 @ np.array([u2, v2, 1.0], dtype=np.float64)
                num = float(np.dot(p1_h, l1))
                denom = l1[0] * l1[0] + l1[1] * l1[1]
                if denom <= 1e-10:
                    continue
                epipolar_err_sq = (num * num) / denom
                sigma2 = float(kf1.level_sigma2[lvl1])
                if epipolar_err_sq < 3.84 * sigma2:
                    matches.append((idx1, idx2))
                    used_2.add(idx2)
                    break

        return matches
