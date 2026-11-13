# -*- coding: utf-8 -*-
"""
run_phase1_demo.py
==================
阶段 1：Windows 纯 Python 算法原型全面验证与端到端评测脚本
所有评测报告、轨迹文件与摄像头运动视频流程图统一输出至 `outputs/` 文件夹。

支持两种运行模式:
1. 默认模式 (零外部数据集依赖):
   `python run_phase1_demo.py`
2. EuRoC 真实数据集模式:
   `python run_phase1_demo.py --euroc data/EuRoC_MH_01_easy/mav0 --max_frames 150`
"""

from __future__ import annotations
import argparse
import os
import numpy as np

from src.core.lie_algebra import SE3, log_so3
from src.core.camera import StereoCamera
from src.frontend.imu_preintegration import ImuPreintegrator
from src.system import SLAMSystem
from src.dataset import SyntheticStereoInertialDataset, EuRoCDataset
from src.visualizer import FrameVisRecord, SLAMVideoFlowVisualizer


def verify_core_operators() -> None:
    """验证 1:1 对齐 C++ 的核心数学与优化算子精度"""
    print("=" * 72)
    print("[Step 1] 开始验证底层核心算子 (1:1 对齐 C++ Eigen / Sophus / g2o)")
    print("=" * 72)

    # 1. 验证 SO(3) / SE(3) Exp & Log 互逆精度
    xi_test = np.array([0.25, -0.4, 0.15, 1.2, -0.8, 2.5], dtype=np.float64)
    T = SE3.exp(xi_test)
    xi_rec = T.log()
    err_se3 = float(np.linalg.norm(xi_test - xi_rec))
    assert err_se3 < 1e-10, f"SE(3) Exp/Log 误差过大: {err_se3}"
    print(f"  [PASS] 1. 李群李代数 SO(3)/SE(3) Exp/Log 往返误差: {err_se3:.3e} (< 1e-10)")

    # 2. 验证 Forster IMU 预积分对零偏的一阶解析雅可比 vs 数值差分
    preint = ImuPreintegrator(bg=np.zeros(3), ba=np.zeros(3))
    rng = np.random.default_rng(123)
    imu_seq = [(rng.normal([0.2, -0.1, 9.8], 0.3), rng.normal([0.05, -0.08, 0.12], 0.05), 0.005) for _ in range(20)]
    for acc, gyro, dt in imu_seq:
        preint.integrate_new_measurement(acc, gyro, dt)

    d_bg = np.array([1.0e-3, -0.8e-3, 0.6e-3], dtype=np.float64)
    d_ba = np.array([5.0e-3, -4.0e-3, 6.0e-3], dtype=np.float64)

    dR_approx = preint.get_updated_delta_rotation(d_bg)
    dv_approx = preint.get_updated_delta_velocity(d_bg, d_ba)
    dp_approx = preint.get_updated_delta_position(d_bg, d_ba)

    preint_exact = ImuPreintegrator(bg=d_bg, ba=d_ba)
    for acc, gyro, dt in imu_seq:
        preint_exact.integrate_new_measurement(acc, gyro, dt)

    err_dR = float(np.linalg.norm(log_so3(preint_exact.dR.T @ dR_approx)))
    err_dv = float(np.linalg.norm(preint_exact.dv - dv_approx))
    err_dp = float(np.linalg.norm(preint_exact.dp - dp_approx))
    assert err_dR < 1e-5 and err_dv < 1e-4 and err_dp < 1e-4
    print(
        f"  [PASS] 2. Forster IMU 预积分零偏一阶雅可比误差: "
        f"dR={err_dR:.2e} rad, dv={err_dv:.2e} m/s, dp={err_dp:.2e} m"
    )


def _collect_frame_vis_record(
    slam: SLAMSystem,
    frame_idx: int,
    timestamp: float,
    gt_pos: np.ndarray | None,
) -> FrameVisRecord:
    """从当前跟踪帧提取用于视频流程图的特征、光流、共视 3D 点与算法状态"""
    cf = slam.tracker.curr_frame
    pf = getattr(slam.tracker, "prev_frame", None)
    T_wc = cf.get_pose_inverse()

    kp_uv_list = []
    kp_depth_list = []
    obs_mp_xyz = []
    curr_mp_to_uv: dict[int, np.ndarray] = {}

    for idx_kp, mp in enumerate(cf.map_points):
        if mp is not None and not mp.is_bad and not cf.outliers[idx_kp]:
            uv = cf.kp_pts[idx_kp]
            p_w = mp.get_world_pos()
            p_c = cf.T_cw.R @ p_w + cf.T_cw.t
            kp_uv_list.append(uv)
            kp_depth_list.append(float(max(0.2, p_c[2])))
            obs_mp_xyz.append(p_w)
            curr_mp_to_uv[mp.id] = uv

    prev_uv_list = []
    curr_uv_list = []
    if pf is not None:
        for idx_p, mp_p in enumerate(pf.map_points):
            if mp_p is not None and mp_p.id in curr_mp_to_uv:
                prev_uv_list.append(pf.kp_pts[idx_p])
                curr_uv_list.append(curr_mp_to_uv[mp_p.id])

    return FrameVisRecord(
        frame_idx=frame_idx,
        timestamp=timestamp,
        img_rect_l=cf.img_left.copy(),
        R_wc_raw=T_wc.R.copy(),
        t_wc_raw=T_wc.t.copy(),
        gt_pos=gt_pos.copy() if gt_pos is not None else None,
        keypoints_uv=np.asarray(kp_uv_list, dtype=np.float32) if kp_uv_list else np.empty((0, 2), dtype=np.float32),
        keypoints_depth=np.asarray(kp_depth_list, dtype=np.float32) if kp_depth_list else np.empty((0,), dtype=np.float32),
        prev_uv=np.asarray(prev_uv_list, dtype=np.float32) if prev_uv_list else np.empty((0, 2), dtype=np.float32),
        curr_uv=np.asarray(curr_uv_list, dtype=np.float32) if curr_uv_list else np.empty((0, 2), dtype=np.float32),
        observed_mp_xyz_raw=np.asarray(obs_mp_xyz, dtype=np.float64) if obs_mp_xyz else np.empty((0, 3), dtype=np.float64),
        all_mp_xyz_raw=slam.get_map_points_xyz(),
        is_keyframe=bool(getattr(slam.tracker, "last_is_keyframe", False)),
        used_optical_flow=bool(getattr(slam.tracker, "last_used_optical_flow", False)),
        inlier_count=int(getattr(slam.tracker, "last_inliers", len(kp_uv_list))),
        num_keyframes=slam.slam_map.num_keyframes(),
        num_mappoints=slam.slam_map.num_mappoints(),
        frontend_ms=float(slam.tracker.timing_stats["frontend_ms"][-1]),
        backend_ms=float(slam.tracker.timing_stats["backend_ms"][-1]),
    )


def _sync_final_poses_to_records(slam: SLAMSystem, records: list[FrameVisRecord]) -> None:
    """将后端 Local BA / 回环优化后的最终平滑相机位姿同步回视频记录帧"""
    _, poses_wc = slam.get_camera_trajectory()
    for rec, T_wc_final in zip(records, poses_wc):
        rec.R_wc_raw = T_wc_final.R.copy()
        rec.t_wc_raw = T_wc_final.t.copy()


def run_synthetic_benchmark(num_frames: int = 80, output_dir: str = "outputs") -> None:
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 72)
    print(f"[Step 2] 运行 3D 闭环双目+IMU SLAM 端到端评测 (共 {num_frames} 帧)")
    print("=" * 72)

    cam = StereoCamera(
        fx=435.0, fy=435.0, cx=320.0, cy=240.0,
        baseline=0.12, width=640, height=480, th_depth_factor=40.0, fps=20.0
    )
    dataset = SyntheticStereoInertialDataset(camera=cam, num_frames=num_frames, num_landmarks=480)

    slam = SLAMSystem(
        camera=cam,
        n_features=550,
        max_local_kfs=8,
        use_hybrid_optical_flow=True
    )

    gt_positions = []
    vis_records: list[FrameVisRecord] = []

    for k in range(num_frames):
        T_wc_gt = dataset.gt_poses_wc[k]
        gt_pos = T_wc_gt.t.copy()
        gt_positions.append(gt_pos)
        img_l, img_r = dataset.render_stereo_pair(T_wc_gt)
        ts = float(dataset.timestamps[k])
        imus = dataset.imu_batches[k]

        slam.track_stereo_inertial(img_l, img_r, ts, imus)
        vis_records.append(_collect_frame_vis_record(slam, k, ts, gt_pos))

        if (k + 1) % 20 == 0 or k == 0 or k == num_frames - 1:
            est_pos, _ = slam.get_camera_trajectory()
            cur_err = float(np.linalg.norm(est_pos[-1] - gt_positions[-1]))
            print(
                f"  Frame [{k+1:3d}/{num_frames:3d}] | "
                f"KeyFrames: {slam.slam_map.num_keyframes():2d} | "
                f"MapPoints: {slam.slam_map.num_mappoints():4d} | "
                f"Loops: {slam.loop_closer.loop_count} | "
                f"Pos Err: {cur_err*100:.2f} cm"
            )

    _sync_final_poses_to_records(slam, vis_records)
    est_positions, _ = slam.get_camera_trajectory()
    gt_positions_arr = np.asarray(gt_positions, dtype=np.float64)
    metrics = SLAMSystem.evaluate_ate(est_positions, gt_positions_arr)

    fe_times = np.asarray(slam.tracker.timing_stats["frontend_ms"])
    be_times = np.asarray(slam.tracker.timing_stats["backend_ms"])
    tot_times = np.asarray(slam.tracker.timing_stats["total_ms"])

    print("\n" + "-" * 72)
    print("  [阶段 1 评测结果汇总 (Windows Python 原型)]")
    print(f"  - 最终活跃关键帧数 (KeyFrames)   : {slam.slam_map.num_keyframes()}")
    print(f"  - 最终活跃 3D 地图点数 (MapPoints): {slam.slam_map.num_mappoints()}")
    print(f"  - 绝对轨迹误差 ATE (Raw RMSE)    : {metrics['raw_rmse_m']*100:.2f} cm ({metrics['raw_rmse_m']:.4f} m)")
    print(f"  - 绝对轨迹误差 ATE (Aligned RMSE): {metrics['aligned_rmse_m']*100:.2f} cm ({metrics['aligned_rmse_m']:.4f} m)")
    print(f"  - 绝对轨迹误差中位数 (Median)    : {metrics['aligned_median_m']*100:.2f} cm")
    print(f"  - 前端跟踪平均耗时 (Frontend)    : {fe_times.mean():.2f} ms (中位数: {np.median(fe_times):.2f} ms)")
    print(f"  - 后端建图分摊耗时 (Backend)     : {be_times.mean():.2f} ms (仅关键帧触发)")
    print(f"  - 系统平均帧率 (Average FPS)     : {1000.0 / max(1e-3, tot_times.mean()):.1f} FPS")
    print("-" * 72)

    est_txt = os.path.join(output_dir, "trajectory_est.txt")
    gt_txt = os.path.join(output_dir, "trajectory_gt.txt")
    eval_png = os.path.join(output_dir, "phase1_slam_evaluation.png")

    np.savetxt(est_txt, est_positions, fmt="%.6f", header="tx ty tz (meters)")
    np.savetxt(gt_txt, gt_positions_arr, fmt="%.6f", header="tx ty tz (meters)")
    _save_evaluation_plot(
        est_positions, gt_positions_arr,
        slam.get_map_points_xyz(),
        fe_times, be_times,
        metrics,
        out_path=eval_png
    )

    visualizer = SLAMVideoFlowVisualizer()
    visualizer.render_all_outputs(
        records=vis_records,
        R_align=metrics["R_align"],
        t_align=metrics["t_align"],
        ate_metrics=metrics,
        output_dir=output_dir,
        prefix="synthetic",
        fps=15,
    )


def run_euroc_sequence(mav0_dir: str, max_frames: int = 150, output_dir: str = "outputs") -> None:
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 72)
    print(f"[EuRoC 真实数据集评测] 加载目录: {mav0_dir}")
    print("=" * 72)

    cam = StereoCamera.create_euroc_camera()
    dataset = EuRoCDataset(mav0_dir)
    n_run = min(len(dataset), max_frames)
    print(f"  已加载双目帧数: {len(dataset)} (本次评测前 {n_run} 帧) | IMU 采样数: {len(dataset.imu_data)}")

    slam = SLAMSystem(
        camera=cam,
        n_features=600,
        max_local_kfs=8,
        use_hybrid_optical_flow=True
    )

    gt_positions = []
    valid_frame_indices = []
    vis_records: list[FrameVisRecord] = []

    for k in range(n_run):
        img_l, img_r, ts, imus, gt_pos = dataset.get_item(k)
        slam.track_stereo_inertial(img_l, img_r, ts, imus)
        vis_records.append(_collect_frame_vis_record(slam, k, ts, gt_pos))

        if gt_pos is not None:
            gt_positions.append(gt_pos)
            valid_frame_indices.append(k)

        if (k + 1) % 20 == 0 or k == 0 or k == n_run - 1:
            print(
                f"  EuRoC Frame [{k+1:4d}/{n_run:4d}] | "
                f"KeyFrames: {slam.slam_map.num_keyframes():3d} | "
                f"MapPoints: {slam.slam_map.num_mappoints():4d} | "
                f"Loops: {slam.loop_closer.loop_count}"
            )

    _sync_final_poses_to_records(slam, vis_records)
    est_all, _ = slam.get_camera_trajectory()
    est_valid = est_all[valid_frame_indices]
    gt_valid = np.asarray(gt_positions, dtype=np.float64)

    metrics = SLAMSystem.evaluate_ate(est_valid, gt_valid)
    est_aligned = metrics["est_aligned"]

    fe_times = np.asarray(slam.tracker.timing_stats["frontend_ms"])
    be_times = np.asarray(slam.tracker.timing_stats["backend_ms"])
    tot_times = np.asarray(slam.tracker.timing_stats["total_ms"])

    print("\n" + "-" * 72)
    print("  [EuRoC 真实无人机数据集评测结果汇总]")
    print(f"  - 评测总帧数 (Frames)            : {n_run}")
    print(f"  - 最终活跃关键帧数 (KeyFrames)   : {slam.slam_map.num_keyframes()}")
    print(f"  - 最终活跃 3D 地图点数 (MapPoints): {slam.slam_map.num_mappoints()}")
    print(f"  - 绝对轨迹误差 ATE (SE3 RMSE)    : {metrics['aligned_rmse_m']*100:.2f} cm ({metrics['aligned_rmse_m']:.4f} m)")
    print(f"  - 绝对轨迹误差平均值 (Mean)      : {metrics['aligned_mean_m']*100:.2f} cm")
    print(f"  - 绝对轨迹误差中位数 (Median)    : {metrics['aligned_median_m']*100:.2f} cm")
    print(f"  - 前端跟踪平均耗时 (Frontend)    : {fe_times.mean():.2f} ms (中位数: {np.median(fe_times):.2f} ms)")
    print(f"  - 后端建图分摊耗时 (Backend)     : {be_times.mean():.2f} ms (仅关键帧触发)")
    print(f"  - 系统平均帧率 (Average FPS)     : {1000.0 / max(1e-3, tot_times.mean()):.1f} FPS")
    print("-" * 72)

    raw_mps = slam.get_map_points_xyz()
    if len(raw_mps) > 0:
        aligned_mps = (raw_mps @ metrics["R_align"].T) + metrics["t_align"]
    else:
        aligned_mps = raw_mps

    est_txt = os.path.join(output_dir, "euroc_trajectory_est.txt")
    gt_txt = os.path.join(output_dir, "euroc_trajectory_gt.txt")
    eval_png = os.path.join(output_dir, "euroc_slam_evaluation.png")

    np.savetxt(est_txt, est_aligned, fmt="%.6f", header="tx ty tz (meters, SE3 aligned)")
    np.savetxt(gt_txt, gt_valid, fmt="%.6f", header="tx ty tz (meters)")
    _save_evaluation_plot(
        est_aligned, gt_valid,
        aligned_mps,
        fe_times[valid_frame_indices], be_times[valid_frame_indices],
        metrics,
        out_path=eval_png
    )

    visualizer = SLAMVideoFlowVisualizer()
    visualizer.render_all_outputs(
        records=vis_records,
        R_align=metrics["R_align"],
        t_align=metrics["t_align"],
        ate_metrics=metrics,
        output_dir=output_dir,
        prefix="euroc",
        fps=15,
    )


def run_euroc_monocular_sequence(mav0_dir: str, max_frames: int = 150, output_dir: str = "outputs") -> None:
    """
    纯单目无 IMU 模式评测:
    仅输入左目 cam0 单张灰度图像与时间戳 (完全屏蔽右目 cam1 与 imu0)，
    验证单目双视图对极几何初始化、无绝对物理尺度 (Scale Ambiguity)、纯旋转掉头尺度漂移与 Sim(3) 对齐表现。
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 72)
    print(f"[EuRoC 纯单目 (Monocular, 无右目 / 无 IMU) 对比评测] 加载目录: {mav0_dir}")
    print("=" * 72)

    cam = StereoCamera.create_euroc_camera()
    dataset = EuRoCDataset(mav0_dir)
    n_run = min(len(dataset), max_frames)
    print(f"  仅加载左目 cam0 单目帧数: {n_run} | 右目 cam1: 已禁用 | IMU: 已禁用")

    slam = SLAMSystem(
        camera=cam,
        n_features=650,
        max_local_kfs=8,
        use_hybrid_optical_flow=True
    )

    gt_positions = []
    valid_frame_indices = []
    vis_records: list[FrameVisRecord] = []

    for k in range(n_run):
        img_l, _, ts, _, gt_pos = dataset.get_item(k)
        # 仅输入单目左图 img_l 与时间戳 ts (无右目，无 IMU)
        slam.track_monocular(img_l, ts)
        vis_records.append(_collect_frame_vis_record(slam, k, ts, gt_pos))

        if gt_pos is not None:
            gt_positions.append(gt_pos)
            valid_frame_indices.append(k)

        if (k + 1) % 20 == 0 or k == 0 or k == n_run - 1:
            print(
                f"  Mono Frame  [{k+1:4d}/{n_run:4d}] | "
                f"State: {slam.tracker.state.name:15s} | "
                f"KeyFrames: {slam.slam_map.num_keyframes():3d} | "
                f"MapPoints: {slam.slam_map.num_mappoints():4d}"
            )

    _sync_final_poses_to_records(slam, vis_records)
    est_all, _ = slam.get_camera_trajectory()
    est_valid = est_all[valid_frame_indices]
    gt_valid = np.asarray(gt_positions, dtype=np.float64)

    metrics = SLAMSystem.evaluate_ate(est_valid, gt_valid)
    fe_times = np.asarray(slam.tracker.timing_stats["frontend_ms"])
    be_times = np.asarray(slam.tracker.timing_stats["backend_ms"])
    tot_times = np.asarray(slam.tracker.timing_stats["total_ms"])

    print("\n" + "-" * 72)
    print("  [EuRoC 纯单目 (无 IMU) 评测结果汇总]")
    print(f"  - 评测总帧数 (Frames)                  : {n_run}")
    print(f"  - 最终活跃关键帧数 (KeyFrames)         : {slam.slam_map.num_keyframes()}")
    print(f"  - 最终活跃 3D 地图点数 (MapPoints)      : {slam.slam_map.num_mappoints()}")
    print(f"  - 真实物理尺度误差 ATE (SE3, s=1.0)    : {metrics['aligned_rmse_m']*100:.2f} cm (因单目无法观测绝对尺度)")
    print(f"  - 最优尺度修正因子 (Sim3 Scale s*)     : {metrics['sim3_scale']:.4f}x")
    print(f"  - 尺度对齐后误差 ATE (Sim3 RMSE, s=s*) : {metrics['sim3_rmse_m']*100:.2f} cm ({metrics['sim3_rmse_m']:.4f} m)")
    print(f"  - 尺度对齐后中位数误差 (Sim3 Median)   : {metrics['sim3_median_m']*100:.2f} cm")
    print(f"  - 前端跟踪平均耗时 (Frontend)          : {fe_times.mean():.2f} ms (中位数: {np.median(fe_times):.2f} ms)")
    print(f"  - 后端建图分摊耗时 (Backend)           : {be_times.mean():.2f} ms (仅关键帧触发)")
    print(f"  - 系统平均帧率 (Average FPS)           : {1000.0 / max(1e-3, tot_times.mean()):.1f} FPS")
    print("-" * 72)

    # 保存纯单目轨迹与评测图 (使用 Sim3 对齐展示单目轨迹几何形态与尺度漂移)
    s_sim3 = metrics["sim3_scale"]
    R_sim3 = metrics["R_align"]
    t_sim3 = metrics["t_sim3"]
    est_sim3 = metrics["est_sim3_aligned"]

    raw_mps = slam.get_map_points_xyz()
    if len(raw_mps) > 0:
        aligned_mps = s_sim3 * (raw_mps @ R_sim3.T) + t_sim3
    else:
        aligned_mps = raw_mps

    np.savetxt(os.path.join(output_dir, "euroc_mono_trajectory_est.txt"), est_sim3, fmt="%.6f", header="tx ty tz (meters, Sim3 aligned)")
    eval_mono_png = os.path.join(output_dir, "euroc_mono_slam_evaluation.png")

    mono_plot_metrics = dict(metrics)
    mono_plot_metrics["aligned_rmse_m"] = metrics["sim3_rmse_m"]
    mono_plot_metrics["aligned_median_m"] = metrics["sim3_median_m"]

    _save_evaluation_plot(
        est_sim3, gt_valid,
        aligned_mps,
        fe_times[valid_frame_indices], be_times[valid_frame_indices],
        mono_plot_metrics,
        out_path=eval_mono_png
    )

    # 若存在双目+IMU轨迹结果，自动生成【双目+IMU vs 纯单目(无IMU)】直观对比大图
    stereo_est_file = os.path.join(output_dir, "euroc_trajectory_est.txt")
    if os.path.exists(stereo_est_file):
        stereo_est = np.loadtxt(stereo_est_file)
        _save_mono_vs_stereo_comparison_plot(
            gt_valid=gt_valid,
            stereo_est=stereo_est,
            mono_se3_est=metrics["est_aligned"],
            mono_sim3_est=est_sim3,
            mono_metrics=metrics,
            out_path=os.path.join(output_dir, "euroc_mono_vs_stereo_imu_comparison.png")
        )

    # 将视频记录中的原始单目坐标乘上 s_sim3 后送入可视化器，生成单目摄像头运动视频流程图
    for rec in vis_records:
        rec.t_wc_raw = rec.t_wc_raw * s_sim3
        if len(rec.observed_mp_xyz_raw) > 0:
            rec.observed_mp_xyz_raw = rec.observed_mp_xyz_raw * s_sim3
        if len(rec.all_mp_xyz_raw) > 0:
            rec.all_mp_xyz_raw = rec.all_mp_xyz_raw * s_sim3

    visualizer = SLAMVideoFlowVisualizer()
    visualizer.render_all_outputs(
        records=vis_records,
        R_align=R_sim3,
        t_align=t_sim3,
        ate_metrics=mono_plot_metrics,
        output_dir=output_dir,
        prefix="euroc_mono",
        fps=15,
    )


def _save_evaluation_plot(
    est_pos: np.ndarray,
    gt_pos: np.ndarray,
    map_pts: np.ndarray,
    fe_times: np.ndarray,
    be_times: np.ndarray,
    metrics: dict[str, float],
    out_path: str
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  [提示] Matplotlib 绘图跳过: {exc}")
        return

    fig = plt.figure(figsize=(15, 10), dpi=140)

    # 1. 3D 轨迹与稀疏地图点云视图
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    if len(map_pts) > 0:
        mask = np.linalg.norm(map_pts, axis=1) < 12.0
        pts_vis = map_pts[mask]
        ax1.scatter(pts_vis[:, 0], pts_vis[:, 2], pts_vis[:, 1], c="#90a4ae", s=3, alpha=0.45, label="3D MapPoints")
    ax1.plot(gt_pos[:, 0], gt_pos[:, 2], gt_pos[:, 1], "k--", linewidth=2.0, label="Ground Truth")
    ax1.plot(est_pos[:, 0], est_pos[:, 2], est_pos[:, 1], color="#d32f2f", linewidth=2.0, label="SLAM Estimated")
    ax1.set_title(f"3D Trajectory & MapPoints (ATE RMSE = {metrics['aligned_rmse_m']*100:.2f} cm)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Z (m)")
    ax1.set_zlabel("Y (m)")
    ax1.legend(loc="upper right")

    # 2. X-Z 俯视平面轨迹对比
    ax2 = fig.add_subplot(2, 2, 2)
    if len(map_pts) > 0:
        mask = np.linalg.norm(map_pts, axis=1) < 10.0
        pts_vis = map_pts[mask]
        ax2.scatter(pts_vis[:, 0], pts_vis[:, 2], c="#cfd8dc", s=5, alpha=0.5, label="MapPoints")
    ax2.plot(gt_pos[:, 0], gt_pos[:, 2], "k--", linewidth=2.2, label="Ground Truth")
    ax2.plot(est_pos[:, 0], est_pos[:, 2], color="#1976d2", linewidth=2.0, label="SLAM Estimated")
    ax2.scatter([gt_pos[0, 0]], [gt_pos[0, 2]], c="green", s=60, zorder=5, label="Start")
    ax2.set_title("Top-Down View (X-Z Plane)")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Z (m)")
    ax2.axis("equal")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    # 3. X / Y / Z 三轴位移随帧号跟踪曲线
    ax3 = fig.add_subplot(2, 2, 3)
    frames_x = np.arange(len(gt_pos))
    ax3.plot(frames_x, gt_pos[:, 0], "k--", alpha=0.7, label="GT X")
    ax3.plot(frames_x, est_pos[:, 0], color="#e53935", label="Est X")
    ax3.plot(frames_x, gt_pos[:, 1], "g--", alpha=0.7, label="GT Y")
    ax3.plot(frames_x, est_pos[:, 1], color="#43a047", label="Est Y")
    ax3.plot(frames_x, gt_pos[:, 2], "b--", alpha=0.7, label="GT Z")
    ax3.plot(frames_x, est_pos[:, 2], color="#1e88e5", label="Est Z")
    ax3.set_title("3-Axis Position Tracking (X, Y, Z vs Frame)")
    ax3.set_xlabel("Frame Index")
    ax3.set_ylabel("Position (m)")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(ncol=3, fontsize=8)

    # 4. 每帧前后端耗时统计 (评估 RK3566 实时性裕度)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(frames_x, fe_times, color="#1976d2", linewidth=1.6, label=f"Frontend Track (mean={fe_times.mean():.1f}ms)")
    ax4.bar(frames_x, be_times, color="#ffa000", alpha=0.6, label="Backend Local BA (KeyFrame triggered)")
    ax4.axhline(33.3, color="#d32f2f", linestyle="--", linewidth=1.2, label="30 FPS Real-Time Budget (33.3ms)")
    ax4.set_title("Per-Frame Runtime Breakdown (Frontend OpticalFlow+ORB vs Backend BA)")
    ax4.set_xlabel("Frame Index")
    ax4.set_ylabel("Time (ms)")
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f"  [图表已保存] 评测报告图表已生成至    : {os.path.abspath(out_path)}")


def _save_mono_vs_stereo_comparison_plot(
    gt_valid: np.ndarray,
    stereo_est: np.ndarray,
    mono_se3_est: np.ndarray,
    mono_sim3_est: np.ndarray,
    mono_metrics: dict[str, float],
    out_path: str,
) -> None:
    """绘制【双目+IMU vs 纯单目(无IMU)】四合一直观对比分析图"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig = plt.figure(figsize=(15, 10), dpi=140)
    stereo_rmse_cm = float(np.sqrt(np.mean(np.sum((stereo_est - gt_valid) ** 2, axis=1)))) * 100.0
    mono_se3_cm = mono_metrics["aligned_rmse_m"] * 100.0
    mono_sim3_cm = mono_metrics["sim3_rmse_m"] * 100.0
    s_sim3 = mono_metrics["sim3_scale"]

    # 1. 3D 轨迹对比 (GT vs Stereo+IMU vs Mono Sim3)
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.plot(gt_valid[:, 0], gt_valid[:, 2], gt_valid[:, 1], "k--", linewidth=2.2, label="Vicon Ground Truth")
    ax1.plot(stereo_est[:, 0], stereo_est[:, 2], stereo_est[:, 1], color="#2e7d32", linewidth=2.2, label=f"Stereo + IMU (RMSE={stereo_rmse_cm:.2f}cm)")
    ax1.plot(mono_sim3_est[:, 0], mono_sim3_est[:, 2], mono_sim3_est[:, 1], color="#e65100", linewidth=2.0, label=f"Pure Mono Sim3 (RMSE={mono_sim3_cm:.2f}cm)")
    ax1.set_title("3D Trajectory: Stereo+IMU vs Pure Monocular (No IMU)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Z (m)")
    ax1.set_zlabel("Y (m)")
    ax1.legend(loc="upper right", fontsize=8)

    # 2. 俯视平面 (X-Y) 尺度不可观对比 (Unscaled SE3 vs Scaled Sim3 vs Stereo+IMU)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(gt_valid[:, 0], gt_valid[:, 1], "k--", linewidth=2.2, label="Vicon Ground Truth")
    ax2.plot(stereo_est[:, 0], stereo_est[:, 1], color="#2e7d32", linewidth=2.2, label=f"Stereo+IMU (True Metric Scale, {stereo_rmse_cm:.2f}cm)")
    ax2.plot(mono_se3_est[:, 0], mono_se3_est[:, 1], color="#9c27b0", linestyle=":", linewidth=1.8, label=f"Mono Unscaled SE3 (s=1.0, {mono_se3_cm:.2f}cm)")
    ax2.plot(mono_sim3_est[:, 0], mono_sim3_est[:, 1], color="#e65100", linewidth=2.0, label=f"Mono Sim3 Scaled (s*={s_sim3:.2f}x, {mono_sim3_cm:.2f}cm)")
    ax2.set_title("Top-Down X-Y View: Scale Ambiguity & Drift in Pure Monocular")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.axis("equal")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="best", fontsize=8)

    # 3. X 轴主运动方向随帧号对比
    ax3 = fig.add_subplot(2, 2, 3)
    frames_x = np.arange(len(gt_valid))
    ax3.plot(frames_x, gt_valid[:, 0], "k--", linewidth=2.0, label="GT X")
    ax3.plot(frames_x, stereo_est[:, 0], color="#2e7d32", linewidth=2.0, label="Stereo+IMU X")
    ax3.plot(frames_x, mono_se3_est[:, 0], color="#9c27b0", linestyle=":", linewidth=1.6, label="Mono Unscaled X (s=1.0)")
    ax3.plot(frames_x, mono_sim3_est[:, 0], color="#e65100", linewidth=1.8, label=f"Mono Sim3 X (s*={s_sim3:.2f}x)")
    ax3.set_title("Primary Motion Axis X(t): Initialization Delay & Turnaround Scale Drift")
    ax3.set_xlabel("Frame Index")
    ax3.set_ylabel("X Position (m)")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="best", fontsize=8)

    # 4. 逐帧绝对位置误差曲线对比 (cm)
    ax4 = fig.add_subplot(2, 2, 4)
    err_stereo_cm = np.linalg.norm(stereo_est - gt_valid, axis=1) * 100.0
    err_mono_se3_cm = np.linalg.norm(mono_se3_est - gt_valid, axis=1) * 100.0
    err_mono_sim3_cm = np.linalg.norm(mono_sim3_est - gt_valid, axis=1) * 100.0
    ax4.plot(frames_x, err_stereo_cm, color="#2e7d32", linewidth=2.0, label=f"Stereo+IMU Error (mean={err_stereo_cm.mean():.1f}cm)")
    ax4.plot(frames_x, err_mono_sim3_cm, color="#e65100", linewidth=2.0, label=f"Pure Mono Sim3 Error (mean={err_mono_sim3_cm.mean():.1f}cm)")
    ax4.plot(frames_x, err_mono_se3_cm, color="#9c27b0", linestyle=":", linewidth=1.5, label=f"Pure Mono Unscaled Error (mean={err_mono_se3_cm.mean():.1f}cm)")
    ax4.set_title("Per-Frame Position Error (cm): Stereo+IMU vs Pure Monocular")
    ax4.set_xlabel("Frame Index")
    ax4.set_ylabel("Position Error (cm)")
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f"  [对比图已保存] 双目+IMU vs 纯单目对比图: {os.path.abspath(out_path)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 Stereo/Stereo-Inertial/Monocular SLAM Validation")
    parser.add_argument("--euroc", type=str, default="", help="Path to EuRoC mav0 directory (optional)")
    parser.add_argument("--mono", action="store_true", help="Run pure Monocular mode (single camera only, no IMU)")
    parser.add_argument("--max_frames", type=int, default=80, help="Number of frames to process")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Unified directory for all output files")
    args = parser.parse_args()

    verify_core_operators()
    if args.euroc and os.path.isdir(args.euroc):
        if args.mono:
            run_euroc_monocular_sequence(args.euroc, max_frames=args.max_frames, output_dir=args.output_dir)
        else:
            run_euroc_sequence(args.euroc, max_frames=args.max_frames, output_dir=args.output_dir)
    else:
        run_synthetic_benchmark(num_frames=args.max_frames, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
