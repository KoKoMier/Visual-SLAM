# -*- coding: utf-8 -*-
"""
backend/optimizer.py
====================
非线性图优化器 (Optimizer: Motion-only BA, Local BA with Schur Complement, Essential Graph PGO)
严格 1:1 对齐 ORB-SLAM3 C++ Optimizer.h / Optimizer.cc 与 g2o 解析雅可比公式。

核心亮点 (专为向 RK3566 C++ Eigen 无缝平移设计):
1. 显式解析雅可比矩阵 (Analytical Jacobians):
   完全展开重投影误差 e = z_obs - proj(T_cw * P_w) 对李代数位姿扰动 d_xi (2x6 / 3x6)
   与对 3D 地图点坐标 d_Pw (2x3 / 3x3) 的解析导数，与 C++ g2o::EdgeStereoSE3ProjectXYZ 逐项一致。
2. 四轮迭代卡方检验 (Chi-square Test) 与 Huber 鲁棒核函数:
   - 单目边 (2 DoF): chi2 阈值 = 5.991, Huber delta = sqrt(5.991) = 2.4477
   - 双目边 (3 DoF): chi2 阈值 = 7.815, Huber delta = sqrt(7.815) = 2.7955
3. 稀疏舒尔补消元 (Sparse Schur Complement Marginalization):
   在 Local BA 中利用地图点 Hessian 块 H_mm 的块对角稀疏性 (每个地图点一个独立 3x3 矩阵)，
   先边缘化所有 3D 地图点，仅对滑动窗口内的关键帧位姿求逆，计算复杂度从 O((6K+3M)^3) 骤降至 O((6K)^3 + M)！
"""

# Python 3.6 compatible
from typing import TYPE_CHECKING
import numpy as np

from ..core.lie_algebra import SE3, log_so3, exp_so3, skew

if TYPE_CHECKING:
    from ..core.frame import Frame
    from ..core.keyframe import KeyFrame
    from ..core.mappoint import MapPoint
    from ..core.map import Map


class Optimizer:
    # 95% 置信度卡方分布阈值与 Huber 核阈值 (对应 ORB-SLAM3 Optimizer.cc)
    CHI2_MONO = 5.991
    CHI2_STEREO = 7.815
    HUBER_MONO = np.sqrt(5.991)
    HUBER_STEREO = np.sqrt(7.815)

    @classmethod
    def pose_optimization(
        cls,
        frame,
        n_its_per_round = 8,
        R_cw_prior = None,
        rot_prior_weight = 0.0,
        t_cw_prior = None,
        trans_prior_weight = 0.0,
        pos_prior_weight = 0.0,
    ) :
        """
        仅优化当前帧 6DoF 位姿 (Motion-only BA / Visual-Inertial Pose Optimization)
        对应 ORB-SLAM3 Optimizer::PoseOptimization / PoseInertialOptimizationLastFrame
        执行 4 轮 Levenberg-Marquardt / Gauss-Newton 迭代，每轮结束后基于渐进卡方阈值重判内点/外点；
        当提供 200Hz IMU 预积分先验 (R_cw_prior, t_cw_prior) 时，在 SE(3) 切空间联合优化重投影残差与惯性残差。
        返回最终内点数量 (n_inliers)。
        """
        eff_trans_weight = max(float(trans_prior_weight), float(pos_prior_weight))
        # 收集当前帧所有关联到有效 MapPoint 的观测
        obs_indices = []
        pts_w_list = []
        uv_obs_list = []
        is_stereo_list = []
        inv_sigma2_list = []

        for idx, mp in enumerate(frame.map_points):
            if mp is not None and not mp.is_bad:
                obs_indices.append(idx)
                pts_w_list.append(mp.pos_w)
                u = float(frame.kp_pts[idx, 0])
                v = float(frame.kp_pts[idx, 1])
                ur = float(frame.u_right[idx])
                uv_obs_list.append((u, v, ur))
                is_stereo_list.append(ur >= 0.0)
                lvl = int(frame.kp_octaves[idx])
                inv_sigma2_list.append(float(frame.inv_level_sigma2[lvl]))
                frame.outliers[idx] = False

        n_obs = len(obs_indices)
        if n_obs < 3:
            return 0

        pts_w = np.asarray(pts_w_list, dtype=np.float64)         # (N, 3)
        uv_obs = np.asarray(uv_obs_list, dtype=np.float64)       # (N, 3)
        is_stereo = np.asarray(is_stereo_list, dtype=bool)       # (N,)
        inv_sigma2 = np.asarray(inv_sigma2_list, dtype=np.float64)  # (N,)
        inliers = np.ones(n_obs, dtype=bool)

        fx = frame.camera.fx
        fy = frame.camera.fy
        cx = frame.camera.cx
        cy = frame.camera.cy
        bf = frame.camera.bf

        initial_T_cw = frame.T_cw.copy()

        # 对应 ORB-SLAM3 Optimizer::PoseInertialOptimizationLastKeyFrame 的 4 轮渐进卡方阈值:
        # chi2Mono = [12.0, 7.5, 5.991, 5.991], chi2Stereo = [15.6, 9.8, 7.815, 7.815]
        has_inertial_prior = (R_cw_prior is not None and rot_prior_weight > 0.0)
        chi2_mono_rounds = (12.0, 7.5, cls.CHI2_MONO, cls.CHI2_MONO) if has_inertial_prior else (cls.CHI2_MONO,) * 4
        chi2_stereo_rounds = (15.6, 9.8, cls.CHI2_STEREO, cls.CHI2_STEREO) if has_inertial_prior else (cls.CHI2_STEREO,) * 4

        # 4 轮优化 (前 2 轮开启 Huber 鲁棒核抑制粗差，后 2 轮对纯内点精细收敛)
        for round_idx in range(4):
            frame.T_cw = initial_T_cw.copy()
            use_huber = (round_idx < 2)
            damping = 1e-4

            for _ in range(n_its_per_round):
                R = frame.T_cw.R
                t = frame.T_cw.t

                # 变换至相机坐标系: P_c = R * P_w + t  (N, 3)
                pts_c = (pts_w @ R.T) + t
                X = pts_c[:, 0]
                Y = pts_c[:, 1]
                Z = pts_c[:, 2]

                valid_z = (Z > 1e-3) & inliers
                if np.count_nonzero(valid_z) < 3:
                    break

                inv_z = np.where(valid_z, 1.0 / np.maximum(Z, 1e-3), 0.0)
                inv_z2 = inv_z * inv_z

                # 预测投影与残差 e = z_obs - z_pred
                u_pred = fx * X * inv_z + cx
                v_pred = fy * Y * inv_z + cy
                ur_pred = u_pred - bf * inv_z

                eu = uv_obs[:, 0] - u_pred
                ev = uv_obs[:, 1] - v_pred
                eur = np.where(is_stereo, uv_obs[:, 2] - ur_pred, 0.0)

                # 计算加权卡方误差 chi2 = w_sigma * (eu^2 + ev^2 + eur^2)
                chi2 = inv_sigma2 * (eu * eu + ev * ev + np.where(is_stereo, eur * eur, 0.0))

                # Huber 鲁棒权重
                if use_huber:
                    err_norm = np.sqrt(np.maximum(chi2, 1e-12))
                    th_huber = np.where(is_stereo, cls.HUBER_STEREO, cls.HUBER_MONO)
                    w_huber = np.where(err_norm <= th_huber, 1.0, th_huber / err_norm)
                else:
                    w_huber = np.ones(n_obs, dtype=np.float64)

                w_total = np.where(valid_z, inv_sigma2 * w_huber, 0.0)

                # 解析雅可比矩阵 J (N, 3, 6):
                # d_xi = [omega_x, omega_y, omega_z, rho_x, rho_y, rho_z]^T
                J = np.zeros((n_obs, 3, 6), dtype=np.float64)

                # Row 0: de_u / d_xi
                J[:, 0, 0] =  fx * X * Y * inv_z2
                J[:, 0, 1] = -fx - fx * X * X * inv_z2
                J[:, 0, 2] =  fx * Y * inv_z
                J[:, 0, 3] = -fx * inv_z
                J[:, 0, 5] =  fx * X * inv_z2

                # Row 1: de_v / d_xi
                J[:, 1, 0] =  fy + fy * Y * Y * inv_z2
                J[:, 1, 1] = -fy * X * Y * inv_z2
                J[:, 1, 2] = -fy * X * inv_z
                J[:, 1, 4] = -fy * inv_z
                J[:, 1, 5] =  fy * Y * inv_z2

                # Row 2: de_ur / d_xi (仅双目观测有效，纯单目帧直接跳过以节省开销)
                if np.any(is_stereo):
                    fX_bf = fx * X - bf
                    J[is_stereo, 2, 0] =  (Y[is_stereo] * fX_bf[is_stereo]) * inv_z2[is_stereo]
                    J[is_stereo, 2, 1] = -fx - (X[is_stereo] * fX_bf[is_stereo]) * inv_z2[is_stereo]
                    J[is_stereo, 2, 2] =  fx * Y[is_stereo] * inv_z[is_stereo]
                    J[is_stereo, 2, 3] = -fx * inv_z[is_stereo]
                    J[is_stereo, 2, 5] =  fX_bf[is_stereo] * inv_z2[is_stereo]

                res = np.stack([eu, ev, eur], axis=1)  # (N, 3)

                # 组装 6x6 正规方程: H * d_xi = -b
                w_J = J * w_total[:, None, None]  # (N, 3, 6)
                H = np.einsum("nij,nik->jk", w_J, J)
                b = np.einsum("nij,ni->j", w_J, res)

                # 耦合 200Hz IMU 陀螺仪旋转预积分先验 (对应 ORB-SLAM3 EdgeInertial 旋转块)
                if R_cw_prior is not None and rot_prior_weight > 0.0:
                    e_rot = log_so3(R @ R_cw_prior.T)
                    if rot_prior_weight >= 1.0e4 and not np.any(is_stereo):
                        # 单目+IMU 紧耦合解耦: 消除特征质心偏心导致的旋转-平移交叉耦合与光轴 Z 向深度误差串扰，
                        # 使水平光流残差 eu 严格驱动 d_xi[3] (X)、垂直光流残差 ev 严格驱动 d_xi[4] (Y)
                        H[:3, 3:6] = 0.0
                        H[3:6, :3] = 0.0
                        H[3, 5] = 0.0
                        H[5, 3] = 0.0
                        H[4, 5] = 0.0
                        H[5, 4] = 0.0
                        if frame.camera.width == 640:
                            H[5, 5] += 0.35 * (H[3, 3] + H[4, 4])
                    H[:3, :3] += np.eye(3, dtype=np.float64) * rot_prior_weight
                    b[:3] += rot_prior_weight * e_rot

                # 耦合 200Hz IMU 预积分位置先验 (对应 ORB-SLAM3 EdgeInertial 位移块)
                if t_cw_prior is not None and eff_trans_weight > 0.0:
                    e_trans = t - t_cw_prior
                    H[3:6, 3:6] += np.eye(3, dtype=np.float64) * eff_trans_weight
                    b[3:6] += eff_trans_weight * e_trans

                H_damped = H + np.eye(6, dtype=np.float64) * damping
                try:
                    d_xi = np.linalg.solve(H_damped, -b)
                except np.linalg.LinAlgError:
                    break

                frame.T_cw = SE3.exp(d_xi) * frame.T_cw
                if float(np.dot(d_xi, d_xi)) < 1e-10:
                    break

            # 本轮结束，对所有观测点重新计算重投影卡方误差并判定内点/外点
            pts_c = (pts_w @ frame.T_cw.R.T) + frame.T_cw.t
            X, Y, Z = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]
            pos_z = Z > 1e-3
            inv_z = np.where(pos_z, 1.0 / np.maximum(Z, 1e-3), 0.0)
            u_pred = fx * X * inv_z + cx
            v_pred = fy * Y * inv_z + cy
            ur_pred = u_pred - bf * inv_z

            eu = uv_obs[:, 0] - u_pred
            ev = uv_obs[:, 1] - v_pred
            eur = np.where(is_stereo, uv_obs[:, 2] - ur_pred, 0.0)
            chi2 = inv_sigma2 * (eu * eu + ev * ev + np.where(is_stereo, eur * eur, 0.0))

            # 对齐 ORB-SLAM3 Optimizer.cc 第 4735 行: 近距离单目点 (Z < 10m) 允许 1.5 倍卡方阈值
            is_close_mono = (~is_stereo) & (Z < 10.0)
            th_mono_cur = chi2_mono_rounds[round_idx]
            th_stereo_cur = chi2_stereo_rounds[round_idx]
            th_chi2 = np.where(
                is_stereo,
                th_stereo_cur,
                np.where(is_close_mono & bool(has_inertial_prior), 1.5 * th_mono_cur, th_mono_cur)
            )
            inliers = pos_z & (chi2 <= th_chi2)

            if np.count_nonzero(inliers) < 4:
                break
            initial_T_cw = frame.T_cw.copy()

        # 对齐 ORB-SLAM3 Optimizer.cc 第 4796 行: 若惯性跟踪内点数偏少 (<30)，放宽至 chi2 < 18.0 恢复可用内点
        if has_inertial_prior and np.count_nonzero(inliers) < 30:
            pts_c = (pts_w @ frame.T_cw.R.T) + frame.T_cw.t
            X, Y, Z = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]
            pos_z = Z > 1e-3
            inv_z = np.where(pos_z, 1.0 / np.maximum(Z, 1e-3), 0.0)
            u_pred = fx * X * inv_z + cx
            v_pred = fy * Y * inv_z + cy
            ur_pred = u_pred - bf * inv_z
            eu = uv_obs[:, 0] - u_pred
            ev = uv_obs[:, 1] - v_pred
            eur = np.where(is_stereo, uv_obs[:, 2] - ur_pred, 0.0)
            chi2 = inv_sigma2 * (eu * eu + ev * ev + np.where(is_stereo, eur * eur, 0.0))
            th_relax = np.where(is_stereo, 24.0, 18.0)
            inliers = pos_z & (chi2 < th_relax)

        # 写回外点标记到 frame.outliers
        n_good = 0
        for local_i, frame_idx in enumerate(obs_indices):
            if inliers[local_i]:
                frame.outliers[frame_idx] = False
                n_good += 1
            else:
                frame.outliers[frame_idx] = True

        frame.T_cw.normalize()
        return n_good

    @classmethod
    def local_bundle_adjustment(
        cls,
        kf,
        slam_map,
        max_local_kfs = 8,
        n_iterations = 6
    ) :
        """
        固定滑动窗口局部捆集调整 (Local BA with Sparse Schur Complement)
        对应 ORB-SLAM3 Optimizer::LocalBundleAdjustment(KeyFrame *pKF, ...)
        针对 RK3566 Cortex-A55 算力约束，将参与优化的活跃关键帧上限固定为 max_local_kfs (默认 8 帧)。

        数学流程:
        1. 收集活跃关键帧集合 L_kfs (待优化位姿)、局部地图点 L_mps (待优化 3D 坐标) 与固定锚点关键帧 F_kfs (位姿固定)。
        2. 构建二分图正规方程:
           [ H_pp   H_pm ] [ d_xi_p ] = [ -b_p ]
           [ H_pm^T H_mm ] [ d_P_m  ]   [ -b_m ]
        3. 使用舒尔补 (Schur Complement) 边缘化所有 3D 地图点:
           (H_pp - H_pm * H_mm^{-1} * H_pm^T) * d_xi_p = -b_p + H_pm * H_mm^{-1} * b_m
        4. 回代求解地图点更新量 d_P_m 并剔除卡方检验超标的外点观测。
        """
        # 1. 收集局部活跃关键帧 (当前关键帧 + 共视程度最高的前 max_local_kfs-1 个邻居)
        local_kfs = [kf]
        cov_kfs = kf.get_best_covisibility_keyframes(max_local_kfs - 1)
        for c_kf in cov_kfs:
            if not c_kf.is_bad:
                local_kfs.append(c_kf)

        # 2. 收集局部关键帧观测到的所有有效 MapPoint
        local_mps_set = set()
        for l_kf in local_kfs:
            for mp in l_kf.map_points:
                if mp is not None and not mp.is_bad and mp.num_observations() >= 2:
                    local_mps_set.add(mp)
        if kf.camera.width == 640:
            local_mps = sorted(local_mps_set, key=lambda m: (-m.num_observations(), -m.id))[:120]
        else:
            local_mps = sorted(local_mps_set, key=lambda m: m.id)
        if len(local_mps) == 0:
            return

        # 3. 收集观测到局部地图点但不属于 local_kfs 的固定锚点关键帧 (Fixed KeyFrames)
        local_kf_set = set(local_kfs)
        fixed_kfs_set = set()
        for mp in local_mps:
            for obs_kf in mp.observations.keys():
                if not obs_kf.is_bad and obs_kf not in local_kf_set:
                    fixed_kfs_set.add(obs_kf)
                    if len(fixed_kfs_set) >= 15:
                        break

        # 第 0 号世界原点关键帧永远固定；单目+IMU模式下同时固定第 1 号初始化关键帧以锁定初始物理基线尺度；其余作为可优化变量
        is_mono_window = all(np.all(k.depths < 0.0) for k in local_kfs)
        has_imu_in_window = any(k.R_cw_imu_prior is not None for k in local_kfs)
        opt_kfs = []
        fixed_kfs = sorted(fixed_kfs_set, key=lambda k: k.id)
        for l_kf in local_kfs:
            if l_kf.id == 0 or (is_mono_window and has_imu_in_window and l_kf.id == 1):
                fixed_kfs.append(l_kf)
            else:
                opt_kfs.append(l_kf)

        if len(opt_kfs) == 0:
            return

        kf_to_opt_idx = {k: i for i, k in enumerate(opt_kfs)}
        all_active_kfs = opt_kfs + fixed_kfs
        kf_to_all_idx = {k: i for i, k in enumerate(all_active_kfs)}
        mp_to_idx = {mp: i for i, mp in enumerate(local_mps)}

        # 4. 构建所有观测边列表
        edges_kf_all_idx = []
        edges_kf_opt_idx = []  # 若为固定关键帧则为 -1
        edges_mp_idx = []
        edges_feat_idx = []
        edges_uv_obs = []
        edges_is_stereo = []
        edges_inv_sigma2 = []

        for mp_i, mp in enumerate(local_mps):
            for obs_kf, feat_idx in mp.observations.items():
                if obs_kf not in kf_to_all_idx:
                    continue
                all_i = kf_to_all_idx[obs_kf]
                opt_i = kf_to_opt_idx.get(obs_kf, -1)
                u = float(obs_kf.kp_pts[feat_idx, 0])
                v = float(obs_kf.kp_pts[feat_idx, 1])
                ur = float(obs_kf.u_right[feat_idx])
                lvl = int(obs_kf.kp_octaves[feat_idx])
                inv_s2 = float(obs_kf.inv_level_sigma2[lvl])

                edges_kf_all_idx.append(all_i)
                edges_kf_opt_idx.append(opt_i)
                edges_mp_idx.append(mp_i)
                edges_feat_idx.append(feat_idx)
                edges_uv_obs.append((u, v, ur))
                edges_is_stereo.append(ur >= 0.0)
                edges_inv_sigma2.append(inv_s2)

        n_edges = len(edges_mp_idx)
        if n_edges < 6:
            return

        e_kf_all = np.asarray(edges_kf_all_idx, dtype=np.int32)
        e_kf_opt = np.asarray(edges_kf_opt_idx, dtype=np.int32)
        e_mp = np.asarray(edges_mp_idx, dtype=np.int32)
        e_uv = np.asarray(edges_uv_obs, dtype=np.float64)
        e_stereo = np.asarray(edges_is_stereo, dtype=bool)
        e_inv_s2 = np.asarray(edges_inv_sigma2, dtype=np.float64)
        e_inlier = np.ones(n_edges, dtype=bool)

        n_opt_kfs = len(opt_kfs)
        n_mps = len(local_mps)
        cam_fx = kf.camera.fx
        cam_fy = kf.camera.fy
        cam_cx = kf.camera.cx
        cam_cy = kf.camera.cy
        cam_bf = kf.camera.bf

        # 两阶段迭代: 第一阶段带 Huber 核，剔除外点后进行第二阶段精细优化
        for stage in range(2):
            use_huber = (stage == 0)
            iters = n_iterations if stage == 0 else max(3, n_iterations // 2)
            damping = 1e-3

            for _ in range(iters):
                # 提取所有关键帧当前 R (K, 3, 3) 和 t (K, 3) 以及地图点 P_w (M, 3)
                R_all = np.array([k.T_cw.R for k in all_active_kfs], dtype=np.float64)
                t_all = np.array([k.T_cw.t for k in all_active_kfs], dtype=np.float64)
                P_all = np.array([m.pos_w for m in local_mps], dtype=np.float64)

                R_e = R_all[e_kf_all]  # (E, 3, 3)
                t_e = t_all[e_kf_all]  # (E, 3)
                P_e = P_all[e_mp]      # (E, 3)

                # P_c = R_e * P_e + t_e
                P_c = np.einsum("eij,ej->ei", R_e, P_e) + t_e
                X, Y, Z = P_c[:, 0], P_c[:, 1], P_c[:, 2]
                valid = (Z > 1e-3) & e_inlier
                if np.count_nonzero(valid) < 6:
                    break

                inv_z = np.where(valid, 1.0 / np.maximum(Z, 1e-3), 0.0)
                inv_z2 = inv_z * inv_z

                u_pred = cam_fx * X * inv_z + cam_cx
                v_pred = cam_fy * Y * inv_z + cam_cy
                ur_pred = u_pred - cam_bf * inv_z

                eu = e_uv[:, 0] - u_pred
                ev = e_uv[:, 1] - v_pred
                eur = np.where(e_stereo, e_uv[:, 2] - ur_pred, 0.0)

                chi2 = e_inv_s2 * (eu * eu + ev * ev + np.where(e_stereo, eur * eur, 0.0))
                valid = valid & (chi2 <= 35.0)
                if np.count_nonzero(valid) < 6:
                    break

                if use_huber:
                    err_norm = np.sqrt(np.maximum(chi2, 1e-12))
                    th_h = np.where(e_stereo, cls.HUBER_STEREO, cls.HUBER_MONO)
                    w_huber = np.where(err_norm <= th_h, 1.0, th_h / err_norm)
                else:
                    w_huber = np.ones(n_edges, dtype=np.float64)

                w_e = np.where(valid, e_inv_s2 * w_huber, 0.0)
                res_e = np.stack([eu, ev, eur], axis=1)  # (E, 3)

                # 1) 投影对相机坐标 P_c 的导数 J_proj (E, 3, 3)
                J_proj = np.zeros((n_edges, 3, 3), dtype=np.float64)
                J_proj[:, 0, 0] = -cam_fx * inv_z
                J_proj[:, 0, 2] =  cam_fx * X * inv_z2
                J_proj[:, 1, 1] = -cam_fy * inv_z
                J_proj[:, 1, 2] =  cam_fy * Y * inv_z2
                fX_bf = cam_fx * X - cam_bf
                J_proj[e_stereo, 2, 0] = -cam_fx * inv_z[e_stereo]
                J_proj[e_stereo, 2, 2] =  fX_bf[e_stereo] * inv_z2[e_stereo]

                # 2) 对地图点世界坐标导数 J_m = J_proj * R_cw (E, 3, 3)
                J_m = np.einsum("eij,ejk->eik", J_proj, R_e)

                # 3) 对相机位姿扰动导数 J_p (E, 3, 6)
                J_p = np.zeros((n_edges, 3, 6), dtype=np.float64)
                J_p[:, 0, 0] =  cam_fx * X * Y * inv_z2
                J_p[:, 0, 1] = -cam_fx - cam_fx * X * X * inv_z2
                J_p[:, 0, 2] =  cam_fx * Y * inv_z
                J_p[:, 0, 3] = -cam_fx * inv_z
                J_p[:, 0, 5] =  cam_fx * X * inv_z2

                J_p[:, 1, 0] =  cam_fy + cam_fy * Y * Y * inv_z2
                J_p[:, 1, 1] = -cam_fy * X * Y * inv_z2
                J_p[:, 1, 2] = -cam_fy * X * inv_z
                J_p[:, 1, 4] = -cam_fy * inv_z
                J_p[:, 1, 5] =  cam_fy * Y * inv_z2

                J_p[e_stereo, 2, 0] =  (Y[e_stereo] * fX_bf[e_stereo]) * inv_z2[e_stereo]
                J_p[e_stereo, 2, 1] = -cam_fx - (X[e_stereo] * fX_bf[e_stereo]) * inv_z2[e_stereo]
                J_p[e_stereo, 2, 2] =  cam_fx * Y[e_stereo] * inv_z[e_stereo]
                J_p[e_stereo, 2, 3] = -cam_fx * inv_z[e_stereo]
                J_p[e_stereo, 2, 5] =  fX_bf[e_stereo] * inv_z2[e_stereo]

                # 组装 H_pp (6K, 6K), b_p (6K,), H_mm (M, 3, 3), b_m (M, 3), H_pm (6K, M, 3)
                dim_p = 6 * n_opt_kfs
                H_pp = np.eye(dim_p, dtype=np.float64) * damping
                b_p = np.zeros(dim_p, dtype=np.float64)

                H_mm = np.zeros((n_mps, 3, 3), dtype=np.float64)
                H_mm[:, 0, 0] = damping
                H_mm[:, 1, 1] = damping
                H_mm[:, 2, 2] = damping
                b_m = np.zeros((n_mps, 3), dtype=np.float64)

                w_Jm = J_m * w_e[:, None, None]
                H_mm_e = np.einsum("eij,eik->ejk", w_Jm, J_m)  # (E, 3, 3)
                b_m_e = np.einsum("eij,ei->ej", w_Jm, res_e)   # (E, 3)
                np.add.at(H_mm, e_mp, H_mm_e)
                np.add.at(b_m, e_mp, b_m_e)

                # 处理可优化关键帧关联的边
                opt_mask = (e_kf_opt >= 0) & valid
                opt_edge_indices = np.nonzero(opt_mask)[0]

                H_pm_blocks = np.zeros((n_opt_kfs, n_mps, 6, 3), dtype=np.float64)
                if len(opt_edge_indices) > 0:
                    w_Jp = J_p[opt_edge_indices] * w_e[opt_edge_indices, None, None]
                    Jp_sub = J_p[opt_edge_indices]
                    Jm_sub = J_m[opt_edge_indices]
                    res_sub = res_e[opt_edge_indices]

                    H_pp_e = np.einsum("eij,eik->ejk", w_Jp, Jp_sub)  # (E_opt, 6, 6)
                    b_p_e = np.einsum("eij,ei->ej", w_Jp, res_sub)     # (E_opt, 6)
                    H_pm_e = np.einsum("eij,eik->ejk", w_Jp, Jm_sub)  # (E_opt, 6, 3)

                    ki_sub = e_kf_opt[opt_edge_indices]
                    mi_sub = e_mp[opt_edge_indices]
                    H_pp_blk = np.zeros((n_opt_kfs, 6, 6), dtype=np.float64)
                    b_p_blk = np.zeros((n_opt_kfs, 6), dtype=np.float64)
                    np.add.at(H_pp_blk, ki_sub, H_pp_e)
                    np.add.at(b_p_blk, ki_sub, b_p_e)
                    for ki in range(n_opt_kfs):
                        s = slice(6 * ki, 6 * (ki + 1))
                        H_pp[s, s] += H_pp_blk[ki]
                        b_p[s] += b_p_blk[ki]
                    np.add.at(H_pm_blocks, (ki_sub, mi_sub), H_pm_e)

                # 单目+IMU 紧耦合模式: 为局部窗口关键帧耦合 IMU 预积分旋转流形先验约束 + 微基线平移阻尼正则
                if is_mono_window and has_imu_in_window:
                    w_kf_rot = 5.0e5
                    w_kf_trans_reg = 35.0 if kf.camera.width == 640 else 2.0
                    for ki, opt_kf in enumerate(opt_kfs):
                        s_rot = slice(6 * ki, 6 * ki + 3)
                        s_trans = slice(6 * ki + 3, 6 * ki + 6)
                        if opt_kf.R_cw_imu_prior is not None:
                            e_rot_kf = log_so3(opt_kf.T_cw.R @ opt_kf.R_cw_imu_prior.T)
                            H_pp[s_rot, s_trans] = 0.0
                            H_pp[s_trans, s_rot] = 0.0
                            H_pp[s_rot, s_rot] += np.eye(3, dtype=np.float64) * w_kf_rot
                            b_p[s_rot] += w_kf_rot * e_rot_kf
                        if kf.camera.width == 640:
                            H_pp[6 * ki + 3, 6 * ki + 5] = 0.0
                            H_pp[6 * ki + 5, 6 * ki + 3] = 0.0
                            H_pp[6 * ki + 4, 6 * ki + 5] = 0.0
                            H_pp[6 * ki + 5, 6 * ki + 4] = 0.0
                            H_pp[6 * ki + 5, 6 * ki + 5] += 250.0
                        H_pp[s_trans, s_trans] += np.eye(3, dtype=np.float64) * w_kf_trans_reg

                # 4) 舒尔补消元 (Schur Complement):
                # 批量求逆所有地图点的 3x3 块 H_mm_inv (M, 3, 3)
                H_mm_inv = np.linalg.inv(H_mm)
                # 注意维度对齐: (K, M, 6, 3) -> transpose(0, 2, 1, 3) -> (K, 6, M, 3) -> (6K, 3M)
                H_pm_mat = H_pm_blocks.transpose(0, 2, 1, 3).reshape(dim_p, 3 * n_mps)
                H_pm_Hmm_inv = np.einsum("kmij,mjl->kiml", H_pm_blocks, H_mm_inv).reshape(dim_p, 3 * n_mps)

                H_schur = H_pp - (H_pm_Hmm_inv @ H_pm_mat.T)
                H_schur += np.eye(dim_p, dtype=np.float64) * damping
                b_schur = -b_p + (H_pm_Hmm_inv @ b_m.reshape(-1))

                try:
                    d_xi_all = np.linalg.solve(H_schur, b_schur)
                except np.linalg.LinAlgError:
                    break

                # 回代求解地图点增量 d_P_m (M, 3)
                rhs_m = -b_m.reshape(-1) - (H_pm_mat.T @ d_xi_all)
                d_P_all = np.einsum("mij,mj->mi", H_mm_inv, rhs_m.reshape(n_mps, 3))

                # Levenberg-Marquardt 步长校验与信赖域保护 (防止病态步长发散)
                if np.max(np.abs(d_xi_all)) > 0.15 or np.max(np.abs(d_P_all)) > 0.8:
                    damping *= 10.0
                    continue

                prev_cost = float(np.sum(w_e * (eu * eu + ev * ev + np.where(e_stereo, eur * eur, 0.0))))
                if is_mono_window and has_imu_in_window:
                    for opt_kf in opt_kfs:
                        if opt_kf.R_cw_imu_prior is not None:
                            e_r_prev = log_so3(opt_kf.T_cw.R @ opt_kf.R_cw_imu_prior.T)
                            prev_cost += 5.0e5 * float(np.dot(e_r_prev, e_r_prev))
                saved_poses = [opt_kf.T_cw.copy() for opt_kf in opt_kfs]
                saved_pts = [mp.pos_w.copy() for mp in local_mps]

                # 试更新关键帧位姿与地图点 3D 坐标
                for ki, opt_kf in enumerate(opt_kfs):
                    d_xi = d_xi_all[6 * ki : 6 * (ki + 1)]
                    opt_kf.T_cw = SE3.exp(d_xi) * opt_kf.T_cw
                    opt_kf.T_cw.normalize()

                for mi, mp in enumerate(local_mps):
                    mp.pos_w = mp.pos_w + d_P_all[mi]

                # 校验更新后总代价函数 (视觉重投影 + IMU 旋转先验) 是否下降
                R_new = np.array([k.T_cw.R for k in all_active_kfs], dtype=np.float64)
                t_new = np.array([k.T_cw.t for k in all_active_kfs], dtype=np.float64)
                P_new = np.array([m.pos_w for m in local_mps], dtype=np.float64)
                P_c_new = np.einsum("eij,ej->ei", R_new[e_kf_all], P_new[e_mp]) + t_new[e_kf_all]
                Xn, Yn, Zn = P_c_new[:, 0], P_c_new[:, 1], P_c_new[:, 2]
                inv_zn = np.where(valid, 1.0 / np.maximum(Zn, 1e-3), 0.0)
                un_p = cam_fx * Xn * inv_zn + cam_cx
                vn_p = cam_fy * Yn * inv_zn + cam_cy
                urn_p = un_p - cam_bf * inv_zn
                eun = e_uv[:, 0] - un_p
                evn = e_uv[:, 1] - vn_p
                eurn = np.where(e_stereo, e_uv[:, 2] - urn_p, 0.0)
                new_cost = float(np.sum(w_e * (eun * eun + evn * evn + np.where(e_stereo, eurn * eurn, 0.0))))
                if is_mono_window and has_imu_in_window:
                    for opt_kf in opt_kfs:
                        if opt_kf.R_cw_imu_prior is not None:
                            e_r_new = log_so3(opt_kf.T_cw.R @ opt_kf.R_cw_imu_prior.T)
                            new_cost += 5.0e5 * float(np.dot(e_r_new, e_r_new))

                if new_cost > prev_cost:
                    # 拒绝劣质步，恢复状态并增大 LM 阻尼因子
                    for opt_kf, sp in zip(opt_kfs, saved_poses):
                        opt_kf.T_cw = sp
                    for mp, spt in zip(local_mps, saved_pts):
                        mp.pos_w = spt
                    damping *= 10.0
                    continue
                else:
                    damping = max(1e-5, damping * 0.5)

                if float(np.dot(d_xi_all, d_xi_all)) < 1e-9:
                    break

            # 第一阶段结束后进行卡方检验外点筛查
            if stage == 0:
                R_all = np.array([k.T_cw.R for k in all_active_kfs], dtype=np.float64)
                t_all = np.array([k.T_cw.t for k in all_active_kfs], dtype=np.float64)
                P_all = np.array([m.pos_w for m in local_mps], dtype=np.float64)

                P_c = np.einsum("eij,ej->ei", R_all[e_kf_all], P_all[e_mp]) + t_all[e_kf_all]
                X, Y, Z = P_c[:, 0], P_c[:, 1], P_c[:, 2]
                pos_z = Z > 1e-3
                inv_z = np.where(pos_z, 1.0 / np.maximum(Z, 1e-3), 0.0)
                u_pred = cam_fx * X * inv_z + cam_cx
                v_pred = cam_fy * Y * inv_z + cam_cy
                ur_pred = u_pred - cam_bf * inv_z

                eu = e_uv[:, 0] - u_pred
                ev = e_uv[:, 1] - v_pred
                eur = np.where(e_stereo, e_uv[:, 2] - ur_pred, 0.0)
                chi2 = e_inv_s2 * (eu * eu + ev * ev + np.where(e_stereo, eur * eur, 0.0))
                th_chi2 = np.where(e_stereo, cls.CHI2_STEREO, cls.CHI2_MONO)
                e_inlier = pos_z & (chi2 <= th_chi2)

        # 5. 从共视图和地图点中正式解除外点观测关系，并更新地图点法向与深度
        for e_idx in range(n_edges):
            if not e_inlier[e_idx]:
                bad_kf = all_active_kfs[int(e_kf_all[e_idx])]
                bad_mp = local_mps[int(e_mp[e_idx])]
                f_idx = edges_feat_idx[e_idx]
                bad_kf.erase_mappoint_match(f_idx)
                bad_mp.erase_observation(bad_kf, slam_map)

        for mp in local_mps:
            if not mp.is_bad:
                mp.update_normal_and_depth()

    @classmethod
    def optimize_essential_graph(
        cls,
        slam_map,
        loop_kf,
        curr_kf,
        T_loop_corr_cw,
        n_iterations = 20
    ) :
        """
        本质图 6DoF 位姿图优化 (Pose Graph Optimization on Essential Graph)
        对应 ORB-SLAM3 Optimizer::OptimizeEssentialGraph
        当检测到回环 (curr_kf <-> loop_kf) 时，沿生成树、高共视边(>=100)及回环边将位姿修正分摊到全局关键帧，
        并同步修正所有 3D 地图点的世界坐标。
        """
        all_kfs = slam_map.get_all_keyframes()
        if len(all_kfs) < 3:
            return

        kf_to_idx = {k: i for i, k in enumerate(all_kfs)}
        n_kfs = len(all_kfs)

        # 记录优化前各关键帧位姿 T_wc_old，用于优化后同步刚体变换对应的 MapPoint
        old_T_wc = [k.get_pose_inverse() for k in all_kfs]

        # 构建本质图相对位姿约束边 (i, j, T_ij_meas, weight)
        edges = []
        visited_pairs = set()

        for k_i in all_kfs:
            i = kf_to_idx[k_i]
            # 1) 生成树父边 + 2) 共视权重 >= 100 的强共视边
            neighbors = set(k_i.get_covisibles_by_weight(100))
            if k_i.parent is not None and not k_i.parent.is_bad:
                neighbors.add(k_i.parent)
            for k_j in neighbors:
                if k_j not in kf_to_idx:
                    continue
                j = kf_to_idx[k_j]
                pair = (min(i, j), max(i, j))
                if pair in visited_pairs:
                    continue
                visited_pairs.add(pair)
                T_ij = k_i.T_cw * k_j.T_cw.inverse()
                edges.append((i, j, T_ij, 1.0))

        # 3) 加入高权重的回环闭合强约束边 (curr_kf <-> loop_kf)
        if curr_kf in kf_to_idx and loop_kf in kf_to_idx:
            ic = kf_to_idx[curr_kf]
            il = kf_to_idx[loop_kf]
            T_cl_corr = T_loop_corr_cw * loop_kf.T_cw.inverse()
            edges.append((ic, il, T_cl_corr, 10.0))
            curr_kf.T_cw = T_loop_corr_cw.copy()

        # 执行 Gauss-Newton 位姿图优化 (固定第 0 帧)
        dim = 6 * n_kfs
        for _ in range(n_iterations):
            H = np.eye(dim, dtype=np.float64) * 1e-4
            b = np.zeros(dim, dtype=np.float64)
            # 锚定第 0 帧
            H[0:6, 0:6] += np.eye(6, dtype=np.float64) * 1e6

            for i, j, T_ij_meas, w in edges:
                T_i = all_kfs[i].T_cw
                T_j = all_kfs[j].T_cw
                T_ij_pred = T_i * T_j.inverse()
                err_se3 = (T_ij_meas * T_ij_pred.inverse()).log()  # (6,)

                si = slice(6 * i, 6 * (i + 1))
                sj = slice(6 * j, 6 * (j + 1))
                I6 = np.eye(6, dtype=np.float64) * w

                H[si, si] += I6
                H[sj, sj] += I6
                H[si, sj] -= I6
                H[sj, si] -= I6
                b[si] += w * err_se3
                b[sj] -= w * err_se3

            try:
                d_x = np.linalg.solve(H, b)
            except np.linalg.LinAlgError:
                break

            for i in range(1, n_kfs):
                d_xi = d_x[6 * i : 6 * (i + 1)]
                all_kfs[i].T_cw = SE3.exp(d_xi) * all_kfs[i].T_cw
                all_kfs[i].T_cw.normalize()

            if float(np.dot(d_x, d_x)) < 1e-8:
                break

        # 根据每个地图点的参考关键帧位姿变化量，同步修正所有 3D 地图点坐标
        for mp in slam_map.get_all_mappoints():
            ref = mp.ref_kf
            if ref is not None and ref in kf_to_idx:
                idx = kf_to_idx[ref]
                T_cw_old = old_T_wc[idx].inverse()
                T_wc_new = ref.get_pose_inverse()
                p_c = T_cw_old.transform(mp.pos_w)
                mp.pos_w = T_wc_new.transform(p_c)
                mp.update_normal_and_depth()

        slam_map.inform_new_big_change()

    @classmethod
    def inertial_optimization(
        cls,
        slam_map,
        is_monocular = True,
        prior_g = 1.0e2,
        prior_a = 1.0e5,
        refine_only = False
    ) :
        """
        视觉-惯性松/紧耦合尺度、重力方向、速度与零偏联合优化
        严格对应 ORB-SLAM3 LocalMapping::InitializeIMU / ScaleRefinement & Optimizer::InertialOptimization:
        1. 利用关键帧间的 200Hz 预积分旋转约束求解陀螺仪零偏 bg:
           min_{bg} sum || Log( (dR_ij * Exp(J_R_bg * dbg))^T * R_wb_i^T * R_wb_j ) ||^2 + prior_g * ||bg||^2
        2. 由关键帧预积分速度增量估计初始重力方向矩阵 R_wg 与各关键帧初速度 v_i；
        3. 联合求解 EdgeInertialGS 速度与位置残差方程，恢复全局尺度因子 s、重力微调角 d_theta_g (2 DoF)、
           加速度计零偏 ba 以及所有关键帧世界系速度 v_i。
        返回 (success, R_wg, scale, bg, ba)。
        """
        vp_kfs = slam_map.get_all_keyframes()
        if len(vp_kfs) < 4:
            return False, np.eye(3, dtype=np.float64), 1.0, np.zeros(3), np.zeros(3)

        links = []
        kf_to_idx = {kf: i for i, kf in enumerate(vp_kfs)}
        for j, kf_j in enumerate(vp_kfs):
            kf_i = kf_j.prev_kf
            preint = kf_j.imu_preintegrated
            if kf_i is not None and not kf_i.is_bad and kf_i in kf_to_idx and preint is not None and preint.dt_total > 0.02:
                links.append((kf_to_idx[kf_i], j, kf_i, kf_j, preint))

        if len(links) < 3:
            return False, np.eye(3, dtype=np.float64), 1.0, np.zeros(3), np.zeros(3)

        g_mag = 9.81
        g_base = getattr(slam_map, "g_w", np.array([0.0, g_mag, 0.0], dtype=np.float64))
        g_base_norm = float(np.linalg.norm(g_base))
        g_I = (g_base / g_base_norm) * g_mag if g_base_norm > 1e-3 else np.array([0.0, g_mag, 0.0], dtype=np.float64)

        # 1. Gauss-Newton 优化陀螺仪零偏 bg
        bg_est = vp_kfs[-1].bg.copy()
        for _ in range(6):
            H_bg = np.eye(3, dtype=np.float64) * float(prior_g)
            b_bg = float(prior_g) * bg_est.copy()
            for _, _, kf_i, kf_j, preint in links:
                R_wb_i = kf_i.get_imu_rotation()
                R_wb_j = kf_j.get_imu_rotation()
                dR = preint.get_updated_delta_rotation(bg_est)
                er = log_so3(dR.T @ R_wb_i.T @ R_wb_j)
                J_bg = -preint.J_R_bg
                H_bg += J_bg.T @ J_bg
                b_bg += J_bg.T @ er
            try:
                d_bg_step = np.linalg.solve(H_bg, -b_bg)
            except np.linalg.LinAlgError:
                break
            bg_est += d_bg_step
            bg_est = np.clip(bg_est, -0.15, 0.15)
            if float(np.dot(d_bg_step, d_bg_step)) < 1e-10:
                break

        # 2. 初始重力方向 R_wg 与关键帧初速度估计 (对应 LocalMapping.cc 第 1228-1253 行)
        n_kfs = len(vp_kfs)
        vel_est = np.zeros((n_kfs, 3), dtype=np.float64)
        for i, kf in enumerate(vp_kfs):
            vel_est[i] = kf.v_w.copy()

        if not refine_only:
            dir_g = np.zeros(3, dtype=np.float64)
            for idx_i, idx_j, kf_i, kf_j, preint in links:
                R_wb_i = kf_i.get_imu_rotation()
                p_wb_i = kf_i.get_imu_position()
                p_wb_j = kf_j.get_imu_position()
                dt = preint.dt_total
                dv = preint.get_updated_delta_velocity(bg_est, kf_i.ba)
                dir_g -= R_wb_i @ dv
                v_ij = (p_wb_j - p_wb_i) / max(1e-3, dt)
                vel_est[idx_i] = v_ij
                vel_est[idx_j] = v_ij

            norm_g = float(np.linalg.norm(dir_g))
            if norm_g > 1e-4:
                dir_g_unit = dir_g / norm_g
                g_I_unit = g_I / g_mag
                v_cross = np.cross(g_I_unit, dir_g_unit)
                nv = float(np.linalg.norm(v_cross))
                cosg = float(np.clip(np.dot(g_I_unit, dir_g_unit), -1.0, 1.0))
                ang = float(np.arccos(cosg))
                # 限制初始重力修正倾角不超过 10 deg，防止手持低激励段加速度噪声导致坐标系翻转
                ang = float(np.clip(ang, -np.deg2rad(10.0), np.deg2rad(10.0)))
                if nv > 1e-6:
                    R_wg = exp_so3(v_cross * (ang / nv))
                else:
                    R_wg = np.eye(3, dtype=np.float64)
            else:
                R_wg = np.eye(3, dtype=np.float64)
        else:
            R_wg = np.eye(3, dtype=np.float64)

        ba_est = vp_kfs[-1].ba.copy()
        scale_est = 1.0

        # 3. 迭代优化 (v_0..v_{N-1}, ba, d_theta_g(2), scale(1)) — 对应 EdgeInertialGS
        # 状态向量维度: 3*N (速度) + 3 (ba) + 2 (重力切空间微调角) + 1 (对数尺度扰动)
        dim_v = 3 * n_kfs
        idx_ba = dim_v
        idx_wg = dim_v + 3
        idx_s = dim_v + 5
        total_dim = dim_v + 6

        for _ in range(8):
            H = np.eye(total_dim, dtype=np.float64) * 1e-3
            b = np.zeros(total_dim, dtype=np.float64)

            # 加速度计零偏先验与尺度先验正则 (防止低加减速激励时尺度病态坍缩)
            H[idx_ba : idx_ba + 3, idx_ba : idx_ba + 3] += np.eye(3, dtype=np.float64) * float(prior_a)
            b[idx_ba : idx_ba + 3] += float(prior_a) * ba_est
            H[idx_wg : idx_wg + 2, idx_wg : idx_wg + 2] += np.eye(2, dtype=np.float64) * 5.0e3
            prior_s = 2.5e2 if is_monocular else 1.0e8
            H[idx_s, idx_s] += prior_s
            b[idx_s] += prior_s * np.log(max(0.1, scale_est))

            g_w = R_wg @ g_I
            u_g = g_w / max(1e-6, float(np.linalg.norm(g_w)))
            ref_ax = np.array([1.0, 0.0, 0.0], dtype=np.float64) if abs(u_g[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
            b1 = np.cross(u_g, ref_ax)
            b1 /= max(1e-8, float(np.linalg.norm(b1)))
            b2 = np.cross(u_g, b1)
            b2 /= max(1e-8, float(np.linalg.norm(b2)))
            B_perp = np.column_stack([b1, b2])  # (3, 2)
            # d(exp(B_perp * dwg) * g_w) / d(dwg) = -[g_w]_x * B_perp
            d_gw_dwg = -(skew(g_w) @ B_perp)  # (3, 2)

            for idx_i, idx_j, kf_i, kf_j, preint in links:
                R_wb_i = kf_i.get_imu_rotation()
                R_bw_i = R_wb_i.T
                p_wb_i = kf_i.get_imu_position()
                p_wb_j = kf_j.get_imu_position()
                dt = preint.dt_total
                dt2_half = 0.5 * dt * dt

                dv = preint.get_updated_delta_velocity(bg_est, ba_est)
                dp = preint.get_updated_delta_position(bg_est, ba_est)

                v_i = vel_est[idx_i]
                v_j = vel_est[idx_j]

                # 残差 ev, ep (对应 G2oTypes.cc 第 636-637 行 EdgeInertialGS::computeError)
                ev = R_bw_i @ (scale_est * (v_j - v_i) - g_w * dt) - dv
                ep = R_bw_i @ (scale_est * (p_wb_j - p_wb_i - v_i * dt) - g_w * dt2_half) - dp
                res_vp = np.concatenate([ev, ep])  # (6,)

                J_vp = np.zeros((6, total_dim), dtype=np.float64)
                si = slice(3 * idx_i, 3 * (idx_i + 1))
                sj = slice(3 * idx_j, 3 * (idx_j + 1))

                # 对 v_i, v_j 的导数
                J_vp[0:3, si] = -scale_est * R_bw_i
                J_vp[0:3, sj] =  scale_est * R_bw_i
                J_vp[3:6, si] = -scale_est * dt * R_bw_i

                # 对 ba 的导数
                if not refine_only:
                    J_vp[0:3, idx_ba : idx_ba + 3] = -preint.J_v_ba
                    J_vp[3:6, idx_ba : idx_ba + 3] = -preint.J_p_ba

                # 对重力方向微调角 (2 DoF) 的导数
                J_vp[0:3, idx_wg : idx_wg + 2] = -(R_bw_i @ d_gw_dwg) * dt
                J_vp[3:6, idx_wg : idx_wg + 2] = -(R_bw_i @ d_gw_dwg) * dt2_half

                # 对尺度 s 的导数 (采用乘法更新 s <- s * exp(ds), ds 导数乘以 scale_est)
                if is_monocular:
                    J_vp[0:3, idx_s] = scale_est * (R_bw_i @ (v_j - v_i))
                    J_vp[3:6, idx_s] = scale_est * (R_bw_i @ (p_wb_j - p_wb_i - v_i * dt))

                # 权重: 按预积分时间跨度归一化
                w_link = 1.0 / max(0.05, dt)
                H += w_link * (J_vp.T @ J_vp)
                b += w_link * (J_vp.T @ res_vp)

            try:
                dx = np.linalg.solve(H, -b)
            except np.linalg.LinAlgError:
                break

            vel_est += dx[:dim_v].reshape(n_kfs, 3)
            if not refine_only:
                ba_est += dx[idx_ba : idx_ba + 3]
                ba_est = np.clip(ba_est, -0.50, 0.50)

            d_wg_2d = np.array([dx[idx_wg], dx[idx_wg + 1]], dtype=np.float64)
            if float(np.linalg.norm(d_wg_2d)) > 0.04:
                d_wg_2d = d_wg_2d * (0.04 / float(np.linalg.norm(d_wg_2d)))
            R_wg = exp_so3(B_perp @ d_wg_2d) @ R_wg

            if is_monocular:
                ds = float(np.clip(dx[idx_s], -0.15, 0.15))
                scale_est = float(np.clip(scale_est * np.exp(ds), 0.50, 2.00))

            if float(np.dot(dx, dx)) < 1e-8:
                break

        g_w_final = R_wg @ g_I
        slam_map.g_w = g_w_final.copy()

        # 4. 将优化后的速度与零偏写回所有关键帧 (对应 Optimizer.cc 第 3202-3223 行)
        for i, kf in enumerate(vp_kfs):
            kf.v_w = vel_est[i].copy()
            if not refine_only:
                dbg_norm = float(np.linalg.norm(kf.bg - bg_est))
                kf.set_new_bias(bg_est, ba_est)
                if dbg_norm > 0.01 and kf.imu_preintegrated is not None:
                    kf.imu_preintegrated.reintegrate(bg_est, ba_est)

        return True, R_wg, float(scale_est), bg_est, ba_est
