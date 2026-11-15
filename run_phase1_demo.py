# -*- coding: utf-8 -*-
"""
run_phase1_demo.py
==================
阶段 1：Windows 纯 Python 算法原型全面验证与端到端评测脚本
所有评测报告、轨迹文件与摄像头运动视频流程图统一输出至 `outputs/` 文件夹。

支持三种运行模式:
1. 默认模式 (零外部数据集依赖):
   `python run_phase1_demo.py`
2. EuRoC 真实数据集模式:
   `python run_phase1_demo.py --euroc data/EuRoC_MH_01_easy/mav0 --max_frames 150`
3. 实机 USB 双目摄像头 + 串口 IMU 实时模式:
   `python run_phase1_demo.py --live --max_frames 60`
"""

import argparse
import os
import sys

# 自动配置 Conda 环境 (如 C:\Users\14163\miniconda3\envs\hfnet) 下的 Library\bin DLL 路径
_conda_lib_bin = os.path.join(sys.prefix, "Library", "bin")
if os.path.isdir(_conda_lib_bin) and _conda_lib_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _conda_lib_bin + os.pathsep + os.environ.get("PATH", "")

import cv2
import numpy as np


def _imwrite_safe(path, img) :
    """兼容 OpenCV 3.4.2 在含中文路径 (视觉slam) 下的图像保存"""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
        return True
    return False

from src.core.lie_algebra import SE3, log_so3
from src.core.camera import StereoCamera
from src.frontend.imu_preintegration import ImuPreintegrator
from src.system import SLAMSystem
from src.dataset import SyntheticStereoInertialDataset, EuRoCDataset, LiveStereoInertialStream
from src.visualizer import FrameVisRecord, SLAMVideoFlowVisualizer, Live3DTrajectorySketch
from src.blackbox_logger import BlackboxLogger, print_blackbox_history


def verify_core_operators() :
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
    slam,
    frame_idx,
    timestamp,
    gt_pos,
) :
    """从当前跟踪帧提取用于视频流程图的特征、光流、共视 3D 点与算法状态"""
    cf = slam.tracker.curr_frame
    pf = getattr(slam.tracker, "prev_frame", None)
    T_wc = cf.get_pose_inverse()

    kp_uv_list = []
    kp_depth_list = []
    obs_mp_xyz = []
    curr_mp_to_uv = {}

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


def _sync_final_poses_to_records(slam, records) :
    """将后端 Local BA / 回环优化后的最终平滑相机位姿同步回视频记录帧"""
    _, poses_wc = slam.get_camera_trajectory()
    for rec, T_wc_final in zip(records, poses_wc):
        rec.R_wc_raw = T_wc_final.R.copy()
        rec.t_wc_raw = T_wc_final.t.copy()


def run_synthetic_benchmark(num_frames = 80, output_dir = "outputs") :
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
    blackbox = BlackboxLogger(
        output_dir=output_dir,
        mode="synthetic_stereo_imu",
        metadata={"num_frames": num_frames, "fx": cam.fx, "baseline_m": cam.baseline},
    )

    gt_positions = []
    vis_records = []

    for k in range(num_frames):
        T_wc_gt = dataset.gt_poses_wc[k]
        gt_pos = T_wc_gt.t.copy()
        gt_positions.append(gt_pos)
        img_l, img_r = dataset.render_stereo_pair(T_wc_gt)
        ts = float(dataset.timestamps[k])
        imus = dataset.imu_batches[k]

        slam.track_stereo_inertial(img_l, img_r, ts, imus)
        blackbox.log_frame(k, ts, slam, imu_batch=imus)
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
    bb_dir = blackbox.finalize(artifact_paths=[eval_png, est_txt, gt_txt])
    print(f"  [黑盒日志归档] 本次运行历史黑盒日志已保存至: {os.path.abspath(bb_dir)}")


def run_euroc_sequence(mav0_dir, max_frames = 150, output_dir = "outputs") :
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
    blackbox = BlackboxLogger(
        output_dir=output_dir,
        mode="euroc_stereo_imu",
        metadata={"mav0_dir": mav0_dir, "max_frames": n_run},
    )

    gt_positions = []
    valid_frame_indices = []
    vis_records = []

    for k in range(n_run):
        img_l, img_r, ts, imus, gt_pos = dataset.get_item(k)
        slam.track_stereo_inertial(img_l, img_r, ts, imus)
        blackbox.log_frame(k, ts, slam, imu_batch=imus)
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
    extractor_name = type(slam.tracker.orb_extractor).__name__
    hfnet_infer_cnt = getattr(slam.tracker.orb_extractor, "inference_count", 0)

    print("\n" + "-" * 72)
    print("  [EuRoC 真实无人机数据集评测结果汇总]")
    print(f"  - Python 运行环境                : {sys.executable}")
    print(f"  - 前端特征提取模型 (Extractor)   : {extractor_name} (HF-Net 推理次数: {hfnet_infer_cnt})")
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
    bb_dir = blackbox.finalize(artifact_paths=[eval_png, est_txt, gt_txt])
    print(f"  [黑盒日志归档] 本次运行历史黑盒日志已保存至: {os.path.abspath(bb_dir)}")

    sketch_euroc = Live3DTrajectorySketch(width=1280, height=500)
    sketch_panel = None
    for idx_r, rec in enumerate(vis_records):
        R_al = metrics["R_align"] @ rec.R_wc_raw
        t_al = (metrics["R_align"] @ rec.t_wc_raw) + metrics["t_align"]
        sketch_panel = sketch_euroc.update_and_render(
            frame_idx=idx_r,
            timestamp=rec.timestamp,
            R_wc=R_al,
            t_wc=t_al,
            map_points_w=aligned_mps,
            is_keyframe=rec.is_keyframe,
            tracking_state_name="OK",
            num_kfs=rec.num_keyframes,
            num_mps=rec.num_mappoints,
            inliers=rec.inlier_count,
            imu_acc=np.array([0.0, 0.0, -9.81]),
            imu_gyro=np.zeros(3),
            imu_euler_deg=np.rad2deg(log_so3(R_al)),
            fps=1000.0 / max(1.0, rec.frontend_ms),
        )
    if sketch_panel is not None:
        sketch_out = os.path.join(output_dir, "euroc_3d_trajectory_sketch.png")
        _imwrite_safe(sketch_out, sketch_panel)
        print(f"  [3D简图已保存] 实时 3D 轨迹简图面板  : {os.path.abspath(sketch_out)}")


def run_euroc_monocular_sequence(mav0_dir, max_frames = 150, output_dir = "outputs") :
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
    blackbox = BlackboxLogger(
        output_dir=output_dir,
        mode="euroc_mono",
        metadata={"mav0_dir": mav0_dir, "max_frames": n_run},
    )

    gt_positions = []
    valid_frame_indices = []
    vis_records = []

    for k in range(n_run):
        img_l, _, ts, _, gt_pos = dataset.get_item(k)
        # 仅输入单目左图 img_l 与时间戳 ts (无右目，无 IMU)
        slam.track_monocular(img_l, ts)
        blackbox.log_frame(k, ts, slam)
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
    bb_dir = blackbox.finalize(artifact_paths=[eval_mono_png])
    print(f"  [黑盒日志归档] 本次运行历史黑盒日志已保存至: {os.path.abspath(bb_dir)}")


def run_euroc_monocular_inertial_sequence(mav0_dir, max_frames = 150, output_dir = "outputs") :
    """
    单目 + IMU 紧耦合模式评测 (Monocular-Inertial VIO):
    仅输入左目 cam0 单张灰度图像 + 200Hz IMU (屏蔽右目 cam1)，
    验证 IMU 陀螺仪旋转流形先验对单目掉头尺度漂移的抑制效果以及惯性尺度恢复精度。
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 72)
    print(f"[EuRoC 单目 + IMU (Monocular-Inertial VIO) 评测] 加载目录: {mav0_dir}")
    print("=" * 72)

    cam = StereoCamera.create_euroc_camera()
    dataset = EuRoCDataset(mav0_dir)
    n_run = min(len(dataset), max_frames)
    print(f"  加载左目 cam0 单目帧数: {n_run} | 右目 cam1: 已禁用 | IMU (200Hz ADIS16448): 已启用")

    slam = SLAMSystem(
        camera=cam,
        n_features=650,
        max_local_kfs=8,
        use_hybrid_optical_flow=True
    )
    blackbox = BlackboxLogger(
        output_dir=output_dir,
        mode="euroc_mono_imu",
        metadata={"mav0_dir": mav0_dir, "max_frames": n_run},
    )

    gt_positions = []
    valid_frame_indices = []
    vis_records = []

    for k in range(n_run):
        img_l, _, ts, imus, gt_pos = dataset.get_item(k)
        slam.track_monocular_inertial(img_l, ts, imus)
        blackbox.log_frame(k, ts, slam, imu_batch=imus)
        vis_records.append(_collect_frame_vis_record(slam, k, ts, gt_pos))

        if gt_pos is not None:
            gt_positions.append(gt_pos)
            valid_frame_indices.append(k)

        if (k + 1) % 20 == 0 or k == 0 or k == n_run - 1:
            print(
                f"  Mono+IMU    [{k+1:4d}/{n_run:4d}] | "
                f"State: {slam.tracker.state.name:15s} | "
                f"KeyFrames: {slam.slam_map.num_keyframes():3d} | "
                f"MapPoints: {slam.slam_map.num_mappoints():4d} | "
                f"IMUs: {len(imus):2d}"
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
    print("  [EuRoC 单目 + IMU (Monocular-Inertial VIO) 评测结果汇总]")
    print(f"  - 评测总帧数 (Frames)                  : {n_run}")
    print(f"  - 最终活跃关键帧数 (KeyFrames)         : {slam.slam_map.num_keyframes()}")
    print(f"  - 最终活跃 3D 地图点数 (MapPoints)      : {slam.slam_map.num_mappoints()}")
    print(f"  - 真实物理尺度误差 ATE (SE3, s=1.0)    : {metrics['aligned_rmse_m']*100:.2f} cm (中位数: {metrics['aligned_median_m']*100:.2f} cm)")
    print(f"  - 惯性尺度因子偏差 (Sim3 Scale s*)     : {metrics['sim3_scale']:.4f}x (理想值 1.0000x)")
    print(f"  - 尺度精修后误差 ATE (Sim3 RMSE, s=s*) : {metrics['sim3_rmse_m']*100:.2f} cm (中位数: {metrics['sim3_median_m']*100:.2f} cm)")
    print(f"  - 前端跟踪平均耗时 (Frontend)          : {fe_times.mean():.2f} ms (中位数: {np.median(fe_times):.2f} ms)")
    print(f"  - 后端建图分摊耗时 (Backend)           : {be_times.mean():.2f} ms (仅关键帧触发)")
    print(f"  - 系统平均帧率 (Average FPS)           : {1000.0 / max(1e-3, tot_times.mean()):.1f} FPS")
    print("-" * 72)

    est_aligned = metrics["est_aligned"]
    raw_mps = slam.get_map_points_xyz()
    if len(raw_mps) > 0:
        aligned_mps = (raw_mps @ metrics["R_align"].T) + metrics["t_align"]
    else:
        aligned_mps = raw_mps

    np.savetxt(
        os.path.join(output_dir, "euroc_mono_imu_trajectory_est.txt"),
        est_aligned,
        fmt="%.6f",
        header="tx ty tz (meters, SE3 aligned)",
    )
    eval_mono_imu_png = os.path.join(output_dir, "euroc_mono_imu_slam_evaluation.png")
    _save_evaluation_plot(
        est_aligned,
        gt_valid,
        aligned_mps,
        fe_times[valid_frame_indices],
        be_times[valid_frame_indices],
        metrics,
        out_path=eval_mono_imu_png,
    )

    sketch_mono_imu = Live3DTrajectorySketch(width=1280, height=500)
    sketch_panel = None
    for idx_r, rec in enumerate(vis_records):
        R_al = metrics["R_align"] @ rec.R_wc_raw
        t_al = (metrics["R_align"] @ rec.t_wc_raw) + metrics["t_align"]
        sketch_panel = sketch_mono_imu.update_and_render(
            frame_idx=idx_r,
            timestamp=rec.timestamp,
            R_wc=R_al,
            t_wc=t_al,
            map_points_w=aligned_mps,
            is_keyframe=rec.is_keyframe,
            tracking_state_name="OK",
            num_kfs=rec.num_keyframes,
            num_mps=rec.num_mappoints,
            inliers=rec.inlier_count,
            imu_acc=np.array([0.0, 0.0, 9.81]),
            imu_gyro=np.zeros(3),
            imu_euler_deg=np.rad2deg(log_so3(R_al)),
            fps=1000.0 / max(1.0, rec.frontend_ms),
        )
    if sketch_panel is not None:
        sketch_out = os.path.join(output_dir, "euroc_mono_imu_3d_trajectory_sketch.png")
        _imwrite_safe(sketch_out, sketch_panel)
        print(f"  [3D简图已保存] 单目+IMU 3D 轨迹简图面板: {os.path.abspath(sketch_out)}")
    bb_dir = blackbox.finalize(artifact_paths=[eval_mono_imu_png])
    print(f"  [黑盒日志归档] 本次运行历史黑盒日志已保存至: {os.path.abspath(bb_dir)}")

    visualizer = SLAMVideoFlowVisualizer()
    visualizer.render_all_outputs(
        records=vis_records,
        R_align=metrics["R_align"],
        t_align=metrics["t_align"],
        ate_metrics=metrics,
        output_dir=output_dir,
        prefix="euroc_mono_imu",
        fps=15,
    )


def _save_evaluation_plot(
    est_pos,
    gt_pos,
    map_pts,
    fe_times,
    be_times,
    metrics,
    out_path) :
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
    gt_valid,
    stereo_est,
    mono_se3_est,
    mono_sim3_est,
    mono_metrics,
    out_path,
) :
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


def _save_live_hardware_plot(
    timestamps,
    est_pos,
    map_pts,
    acc_history,
    gyro_history,
    euler_history,
    fe_times,
    be_times,
    out_path,
    is_monocular = False,
) :
    """绘制实机 USB 双目/单目摄像头 + 串口 IMU 实时采集与 SLAM 跟踪四合一评测报告图"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  [提示] Matplotlib 绘图跳过: {exc}")
        return

    fig = plt.figure(figsize=(15, 10), dpi=140)
    mode_tag = "Monocular + IMU" if is_monocular else "Stereo + IMU"

    # 1. 3D 实时相机轨迹与稀疏地图点云
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    if len(map_pts) > 0:
        mask = np.linalg.norm(map_pts, axis=1) < 15.0
        pts_vis = map_pts[mask]
        ax1.scatter(pts_vis[:, 0], pts_vis[:, 2], pts_vis[:, 1], c="#90a4ae", s=4, alpha=0.5, label="3D MapPoints")
    ax1.plot(est_pos[:, 0], est_pos[:, 2], est_pos[:, 1], color="#d32f2f", linewidth=2.2, label="Live Camera 6DoF Trajectory")
    ax1.scatter([est_pos[0, 0]], [est_pos[0, 2]], [est_pos[0, 1]], c="green", s=60, label="Start")
    ax1.set_title(f"Live USB {mode_tag} 3D Trajectory ({len(est_pos)} Frames, {len(map_pts)} MapPoints)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Z (m)")
    ax1.set_zlabel("Y (m)")
    ax1.legend(loc="upper right", fontsize=8)

    # 2. 串口 IMU 三轴加速度与重力模长实时波形 (m/s^2)
    ax2 = fig.add_subplot(2, 2, 2)
    g_norm = np.linalg.norm(acc_history, axis=1)
    ax2.plot(timestamps, acc_history[:, 0], color="#e53935", linewidth=1.5, label="Acc X (m/s²)")
    ax2.plot(timestamps, acc_history[:, 1], color="#43a047", linewidth=1.5, label="Acc Y (m/s²)")
    ax2.plot(timestamps, acc_history[:, 2], color="#1e88e5", linewidth=1.5, label="Acc Z (m/s²)")
    ax2.plot(timestamps, g_norm, "k--", linewidth=1.8, label=f"|g| Norm (mean={g_norm.mean():.2f} m/s²)")
    ax2.set_title("Live Serial IMU 3-Axis Acceleration & Gravity Norm")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Acceleration (m/s²)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="best", fontsize=8)

    # 3. 串口 IMU 三轴角速度 (rad/s) 与欧拉姿态角 (deg)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(timestamps, gyro_history[:, 0], color="#d81b60", linewidth=1.4, label="Gyro Wx (rad/s)")
    ax3.plot(timestamps, gyro_history[:, 1], color="#00897b", linewidth=1.4, label="Gyro Wy (rad/s)")
    ax3.plot(timestamps, gyro_history[:, 2], color="#3949ab", linewidth=1.4, label="Gyro Wz (rad/s)")
    ax3_twin = ax3.twinx()
    ax3_twin.plot(timestamps, euler_history[:, 0], color="#ff8f00", linestyle="--", alpha=0.75, label="Roll (deg)")
    ax3_twin.plot(timestamps, euler_history[:, 1], color="#6d4c41", linestyle=":", alpha=0.75, label="Pitch (deg)")
    ax3_twin.plot(timestamps, euler_history[:, 2], color="#5e35b1", linestyle="-.", alpha=0.75, label="Yaw (deg)")
    ax3.set_title("Live Serial IMU Angular Velocity (rad/s) & Attitude (deg)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Gyro (rad/s)")
    ax3_twin.set_ylabel("Euler Angle (deg)")
    ax3.grid(True, linestyle=":", alpha=0.6)
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="best", ncol=2, fontsize=8)

    # 4. 逐帧前后端实时耗时统计
    ax4 = fig.add_subplot(2, 2, 4)
    frames_x = np.arange(len(fe_times))
    ax4.plot(frames_x, fe_times, color="#1976d2", linewidth=1.6, label=f"Frontend Track (median={np.median(fe_times):.1f}ms)")
    ax4.bar(frames_x, be_times, color="#ffa000", alpha=0.6, label="Backend Local BA")
    ax4.axhline(33.3, color="#d32f2f", linestyle="--", linewidth=1.2, label="30 FPS Budget (33.3ms)")
    ax4.set_title("Live Hardware Per-Frame Runtime Breakdown")
    ax4.set_xlabel("Frame Index")
    ax4.set_ylabel("Time (ms)")
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f"  [图表已保存] 实机双目+IMU评测图已生成至: {os.path.abspath(out_path)}")


def run_live_hardware_slam(
    cam_id = -1,
    concat_width = 1280,
    height = 480,
    imu_port = "COM3",
    imu_baud = 115200,
    calib_json = "config/custom_stereo_imu.json",
    max_frames = 0,
    output_dir = "outputs",
    show_gui = True,
    force_monocular = False,
) :
    """
    实机 USB 双目 (1280x480) 或 单目 (640x480) 摄像头 + 串口 IMU 实时 SLAM 跟踪与可视化评测主流程
    (持续运行，仅当用户在窗口按下 'q' 键时退出)
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 72)
    print("[实机硬件 SLAM 模式] 正在初始化 USB 摄像头与串口 IMU...")
    print("=" * 72)

    stream = LiveStereoInertialStream(
        cam_id=cam_id,
        concat_width=concat_width,
        height=height,
        imu_port=imu_port,
        imu_baud=imu_baud,
        calib_json=calib_json,
        force_monocular=force_monocular,
    )
    mode_tag = "单目 + IMU (Mono-Inertial)" if stream.is_monocular else "双目 + IMU (Stereo-Inertial)"
    imu_snap = stream.imu.get_status_snapshot()
    if stream.is_monocular:
        print(f"  [Camera] 已连接单目摄像头 ID={stream.cam_id} | 分辨率={stream.half_w}x{height} | 运行模式: {mode_tag}")
    else:
        print(f"  [Camera] 已连接双目摄像头 ID={stream.cam_id} | 拼接分辨率={concat_width}x{height} (单目={stream.half_w}x{height}) | 运行模式: {mode_tag}")
    print(f"  [Calib ] 相机参数来源: {stream.calib_source}")
    if imu_snap["connected"]:
        acc0 = imu_snap["acc_mps2"]
        gyro0 = imu_snap["gyro_rads"]
        eul0 = imu_snap["euler_deg"]
        print(
            f"  [IMU   ] 已连接串口 {imu_snap['port']} @ {imu_snap['baud']} bps ({imu_snap.get('rate_hz', 200)}Hz) | 已收包: {imu_snap['total_packets']} | "
            f"温度: {imu_snap['temp_c']:.2f} °C"
        )
        print(
            f"           初始加速度 Acc=[{acc0[0]:+.3f}, {acc0[1]:+.3f}, {acc0[2]:+.3f}] m/s² (|g|={np.linalg.norm(acc0):.3f}) | "
            f"姿态角 Roll={eul0[0]:+.1f}°, Pitch={eul0[1]:+.1f}°, Yaw={eul0[2]:+.1f}°"
        )
    else:
        print(f"  [IMU   ] 警告: 未能打开串口 {imu_port}，将退化为纯视觉模式运行。")

    slam = SLAMSystem(
        camera=stream.camera,
        n_features=600,
        max_local_kfs=8,
        use_hybrid_optical_flow=True,
    )
    run_mode_str = "live_mono_imu" if stream.is_monocular else "live_stereo_imu"
    blackbox = BlackboxLogger(
        output_dir=output_dir,
        mode=run_mode_str,
        metadata={
            "cam_id": int(stream.cam_id),
            "resolution": f"{stream.half_w}x{height}",
            "is_monocular": bool(stream.is_monocular),
            "calib_source": str(stream.calib_source),
            "fx": float(stream.camera.fx),
            "fy": float(stream.camera.fy),
            "cx": float(stream.camera.cx),
            "cy": float(stream.camera.cy),
            "imu_port": str(imu_snap.get("port", imu_port)),
            "imu_baud": int(imu_snap.get("baud", imu_baud)),
            "imu_rate_hz": int(imu_snap.get("rate_hz", 200)),
            "imu_connected": bool(imu_snap.get("connected", False)),
        },
    )
    print(f"  [Blackbox] 实时黑盒记录已启动 -> {os.path.abspath(blackbox.run_dir)}")

    vis_records = []
    ts_list = []
    acc_list = []
    gyro_list = []
    euler_list = []
    sample_pair_saved = False

    sketch_3d = Live3DTrajectorySketch(width=1280, height=500)
    last_sketch_panel = None
    last_combined_dashboard = None
    R_imu_ref = None

    def _euler_to_R(eul_deg) :
        r, p, y = np.deg2rad(eul_deg[0]), np.deg2rad(eul_deg[1]), np.deg2rad(eul_deg[2])
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64)
        Ry = np.array([[cr, 0.0, sr], [0.0, 1.0, 0.0], [-sr, 0.0, cr]], dtype=np.float64)
        Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        return Rz @ Ry @ Rx

    print("  [提示   ] 实机窗口已开启，将持续运行，在画面窗口按 'q' 键退出。")
    k = 0
    try:
        while True:
            img_l, img_r, ts, imus, imu_status = stream.read_frame()
            if stream.is_monocular:
                slam.track_monocular_inertial(img_l, ts, imus)
            else:
                slam.track_stereo_inertial(img_l, img_r, ts, imus)

            blackbox.log_frame(k, ts, slam, imu_batch=imus, imu_status=imu_status)
            rec = _collect_frame_vis_record(slam, k, ts, gt_pos=None)
            eul_cur = imu_status["euler_deg"]
            R_wb_cur = _euler_to_R(eul_cur)
            if R_imu_ref is None:
                R_imu_ref = R_wb_cur.copy()

            # 若视觉尚未完成三角化初始化，使用实测 IMU 相对旋转实时驱动 3D 视锥体姿态
            if slam.tracker.state.name != "OK" and R_imu_ref is not None:
                R_bc = stream.camera.T_b_c1[:3, :3]
                rec.R_wc_raw = R_bc.T @ (R_imu_ref.T @ R_wb_cur) @ R_bc

            rec.gt_pos = rec.t_wc_raw.copy()
            vis_records.append(rec)

            ts_list.append(ts)
            acc_list.append(imu_status["acc_mps2"])
            gyro_list.append(imu_status["gyro_rads"])
            euler_list.append(eul_cur)

            if stream.is_monocular:
                rect_l = stream.camera.rectify_mono(img_l)
                left_vis = cv2.cvtColor(rect_l, cv2.COLOR_GRAY2BGR)
                for p0_uv, p1_uv in zip(rec.prev_uv, rec.curr_uv):
                    cv2.line(
                        left_vis,
                        (int(p0_uv[0]), int(p0_uv[1])),
                        (int(p1_uv[0]), int(p1_uv[1])),
                        (0, 255, 120),
                        2,
                        cv2.LINE_AA,
                    )
                for pt in slam.tracker.curr_frame.kp_pts[:280]:
                    cv2.circle(left_vis, (int(pt[0]), int(pt[1])), 3, (0, 220, 255), -1, cv2.LINE_AA)

                right_info = np.full_like(left_vis, (24, 28, 36))
                cv2.rectangle(right_info, (12, 48), (628, 468), (58, 68, 82), 1)
                acc_c = imu_status["acc_mps2"]
                gyr_c = imu_status["gyro_rads"]
                raw_e = imu_status.get("raw_euler_deg", eul_cur)
                lines_right = [
                    "MONOCULAR + IMU TIGHTLY-COUPLED VIO",
                    f"Camera Device : USB Camera ID={stream.cam_id} ({stream.half_w}x{height})",
                    f"Serial IMU    : {imu_status['port']} @ {imu_status['baud']} bps / {imu_status.get('rate_hz', 200)}Hz (Pkts={imu_status['total_packets']})",
                    f"Tracker State : {slam.tracker.state.name}  |  Inliers: {rec.inlier_count}",
                    f"Active Map    : KeyFrames = {rec.num_keyframes}  |  3D MapPoints = {rec.num_mappoints}",
                    f"Camera Pos (m): X={rec.t_wc_raw[0]:+.3f}, Y={rec.t_wc_raw[1]:+.3f}, Z={rec.t_wc_raw[2]:+.3f}",
                    f"IMU Acc (m/s2): Ax={acc_c[0]:+.2f}, Ay={acc_c[1]:+.2f}, Az={acc_c[2]:+.2f} (|g|={np.linalg.norm(acc_c):.2f})",
                    f"IMU Gyro(rad/s): Wx={gyr_c[0]:+.3f}, Wy={gyr_c[1]:+.3f}, Wz={gyr_c[2]:+.3f}",
                    f"Attitude (deg): Roll={eul_cur[0]:+.1f}, Pitch={eul_cur[1]:+.1f}, Yaw={eul_cur[2]:+.1f}",
                    f"Raw IMU (deg) : Roll={raw_e[0]:+.1f}, Pitch={raw_e[1]:+.1f}, Yaw={raw_e[2]:+.1f}",
                ]
                for li, ltxt in enumerate(lines_right):
                    col_t = (0, 230, 255) if li == 0 else ((120, 255, 160) if li in (3, 4, 5) else (215, 225, 235))
                    cv2.putText(right_info, ltxt, (28, 82 + li * 36), cv2.FONT_HERSHEY_SIMPLEX, 0.50, col_t, 1, cv2.LINE_AA)
                hud = np.hstack([left_vis, right_info])
            else:
                rect_l, rect_r = stream.camera.rectify_stereo(img_l, img_r)
                hud = cv2.cvtColor(np.hstack([rect_l, rect_r]), cv2.COLOR_GRAY2BGR)
                for p0_uv, p1_uv in zip(rec.prev_uv, rec.curr_uv):
                    cv2.line(
                        hud,
                        (int(p0_uv[0]), int(p0_uv[1])),
                        (int(p1_uv[0]), int(p1_uv[1])),
                        (0, 255, 120),
                        1,
                        cv2.LINE_AA,
                    )
                for pt in slam.tracker.curr_frame.kp_pts[:250]:
                    cv2.circle(hud, (int(pt[0]), int(pt[1])), 3, (0, 220, 255), -1, cv2.LINE_AA)

            if not sample_pair_saved and (k == 5 or len(rec.keypoints_uv) > 10):
                pair_vis = hud.copy()
                for y in range(24, pair_vis.shape[0], 32):
                    cv2.line(pair_vis, (0, y), (pair_vis.shape[1], y), (0, 200, 0), 1, cv2.LINE_AA)
                sample_path = os.path.join(output_dir, "live_stereo_pair_sample.png")
                _imwrite_safe(sample_path, pair_vis)
                sample_pair_saved = True

            tot_ms_last = max(1.0, float(slam.tracker.timing_stats["total_ms"][-1]))
            inst_fps = 1000.0 / tot_ms_last
            title_mode = "Live Mono+IMU VIO" if stream.is_monocular else "Live Stereo+IMU SLAM"
            info_txt = (
                f"{title_mode} | Frame {k+1} [Press 'q' to quit] | State: {slam.tracker.state.name} | "
                f"KFs: {slam.slam_map.num_keyframes()} | MPs: {slam.slam_map.num_mappoints()} | "
                f"IMU RPY: ({eul_cur[0]:+.1f}, {eul_cur[1]:+.1f}, {eul_cur[2]:+.1f}) deg"
            )
            cv2.rectangle(hud, (0, 0), (hud.shape[1], 36), (20, 24, 30), -1)
            cv2.putText(hud, info_txt, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 235, 255), 2, cv2.LINE_AA)

            # 实时渲染 3D 轨迹简图面板与上下拼接大屏
            last_sketch_panel = sketch_3d.update_and_render(
                frame_idx=k,
                timestamp=ts,
                R_wc=rec.R_wc_raw,
                t_wc=rec.t_wc_raw,
                map_points_w=rec.all_mp_xyz_raw,
                is_keyframe=rec.is_keyframe,
                tracking_state_name=slam.tracker.state.name,
                num_kfs=rec.num_keyframes,
                num_mps=rec.num_mappoints,
                inliers=rec.inlier_count,
                imu_acc=imu_status["acc_mps2"],
                imu_gyro=imu_status["gyro_rads"],
                imu_euler_deg=eul_cur,
                fps=inst_fps,
                stereo_top_hud=None,
            )
            last_combined_dashboard = np.vstack([hud, last_sketch_panel])

            if (k + 1) % 30 == 0 or k == 0:
                acc_cur = imu_status["acc_mps2"]
                raw_eul = imu_status.get("raw_euler_deg", eul_cur)
                print(
                    f"  Live Frame [{k+1:4d}] | "
                    f"State: {slam.tracker.state.name:15s} | "
                    f"KFs: {slam.slam_map.num_keyframes():2d} | "
                    f"MPs: {slam.slam_map.num_mappoints():4d} | "
                    f"IMUs: {len(imus):2d} | "
                    f"|g|: {np.linalg.norm(acc_cur):.2f}m/s² | "
                    f"RPY: ({eul_cur[0]:+.1f}°, {eul_cur[1]:+.1f}°, {eul_cur[2]:+.1f}°) "
                    f"[Raw: ({raw_eul[0]:+.1f}°, {raw_eul[1]:+.1f}°, {raw_eul[2]:+.1f}°)]"
                )
                if (k + 1) % 60 == 0 or k == 0:
                    if last_sketch_panel is not None:
                        _imwrite_safe(os.path.join(output_dir, "live_3d_trajectory_sketch.png"), last_sketch_panel)
                    if last_combined_dashboard is not None:
                        _imwrite_safe(os.path.join(output_dir, "live_slam_dashboard_combined.png"), last_combined_dashboard)

            cv2.imshow("Live USB Stereo + IMU SLAM & 3D Trajectory Sketch", last_combined_dashboard)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                print("\n  [用户退出] 检测到按键 'q'，正在退出并保存评测数据...")
                break
            sketch_3d.handle_key(key)
            k += 1
    except KeyboardInterrupt:
        print("\n  [用户中断] 收到 Ctrl+C，正在保存当前已采集的 3D 轨迹简图与位姿数据...")
    finally:
        stream.close()
        cv2.destroyAllWindows()

    sketch_path = os.path.join(output_dir, "live_3d_trajectory_sketch.png")
    dash_path = os.path.join(output_dir, "live_slam_dashboard_combined.png")
    if last_sketch_panel is not None:
        _imwrite_safe(sketch_path, last_sketch_panel)
        print(f"  [3D简图已保存] 实时 3D 轨迹简图面板  : {os.path.abspath(sketch_path)}")
    if last_combined_dashboard is not None:
        _imwrite_safe(dash_path, last_combined_dashboard)
        print(f"  [监控大屏保存] 双目画面+3D轨迹简图大屏: {os.path.abspath(dash_path)}")

    if len(vis_records) == 0:
        bb_dir = blackbox.finalize()
        print(f"  [黑盒日志已归档] 本次运行历史黑盒目录: {os.path.abspath(bb_dir)}")
        return

    _sync_final_poses_to_records(slam, vis_records)
    for rec in vis_records:
        rec.gt_pos = rec.t_wc_raw.copy()

    est_positions, _ = slam.get_camera_trajectory()
    fe_times = np.asarray(slam.tracker.timing_stats["frontend_ms"])
    be_times = np.asarray(slam.tracker.timing_stats["backend_ms"])
    tot_times = np.asarray(slam.tracker.timing_stats["total_ms"])
    ts_arr = np.asarray(ts_list, dtype=np.float64)
    acc_arr = np.asarray(acc_list, dtype=np.float64)
    gyro_arr = np.asarray(gyro_list, dtype=np.float64)
    euler_arr = np.asarray(euler_list, dtype=np.float64)

    mode_cn = "单目摄像头 + 串口 IMU" if stream.is_monocular else "双目摄像头 + 串口 IMU"
    print("\n" + "-" * 72)
    print(f"  [实机 USB {mode_cn} 实时运行汇总]")
    print(f"  - 实际采集处理帧数 (Frames)      : {len(vis_records)}")
    print(f"  - 最终跟踪状态 (Tracker State)   : {slam.tracker.state.name}")
    print(f"  - 最终活跃关键帧数 (KeyFrames)   : {slam.slam_map.num_keyframes()}")
    print(f"  - 最终活跃 3D 地图点数 (MapPoints): {slam.slam_map.num_mappoints()}")
    print(f"  - IMU 累计接收有效帧数           : {stream.imu.total_packets}")
    print(f"  - IMU 平均重力模长 (|g| Norm)    : {np.linalg.norm(acc_arr, axis=1).mean():.4f} m/s²")
    print(f"  - 前端跟踪平均耗时 (Frontend)    : {fe_times.mean():.2f} ms (中位数: {np.median(fe_times):.2f} ms)")
    print(f"  - 系统平均帧率 (Average FPS)     : {1000.0 / max(1e-3, tot_times.mean()):.1f} FPS")
    print("-" * 72)

    est_txt = os.path.join(output_dir, "live_trajectory_est.txt")
    imu_txt = os.path.join(output_dir, "live_imu_telemetry.txt")
    eval_png = os.path.join(output_dir, "live_slam_evaluation.png")

    np.savetxt(est_txt, est_positions, fmt="%.6f", header="tx ty tz (meters)")
    imu_table = np.column_stack([ts_arr, acc_arr, gyro_arr, euler_arr])
    np.savetxt(
        imu_txt,
        imu_table,
        fmt="%.6f",
        header="timestamp_s ax_mps2 ay_mps2 az_mps2 wx_rads wy_rads wz_rads roll_deg pitch_deg yaw_deg",
    )

    _save_live_hardware_plot(
        timestamps=ts_arr,
        est_pos=est_positions,
        map_pts=slam.get_map_points_xyz(),
        acc_history=acc_arr,
        gyro_history=gyro_arr,
        euler_history=euler_arr,
        fe_times=fe_times,
        be_times=be_times,
        out_path=eval_png,
        is_monocular=stream.is_monocular,
    )

    bb_dir = blackbox.finalize(
        artifact_paths=[dash_path, sketch_path, eval_png, est_txt, imu_txt]
    )
    print(f"  [黑盒日志归档] 本次运行历史黑盒日志已完整保存至: {os.path.abspath(bb_dir)}")
    print(f"                 - 逐帧全状态表 : {os.path.abspath(blackbox.csv_path)}")
    print(f"                 - 复盘诊断报告 : {os.path.abspath(blackbox.report_path)}")

    dummy_metrics = {
        "aligned_rmse_m": 0.0,
        "aligned_median_m": 0.0,
    }
    visualizer = SLAMVideoFlowVisualizer()
    vis_records_out = vis_records[::max(1, len(vis_records) // 180)] if len(vis_records) > 240 else vis_records
    visualizer.render_all_outputs(
        records=vis_records_out,
        R_align=np.eye(3, dtype=np.float64),
        t_align=np.zeros(3, dtype=np.float64),
        ate_metrics=dummy_metrics,
        output_dir=output_dir,
        prefix="live",
        fps=15,
    )


def main() :
    parser = argparse.ArgumentParser(description="Phase 1 Stereo/Stereo-Inertial/Monocular SLAM Validation")
    parser.add_argument("--euroc", type=str, default="", help="Path to EuRoC mav0 directory (optional)")
    parser.add_argument("--mono", action="store_true", help="Run pure Monocular mode (single camera only, no IMU)")
    parser.add_argument("--mono_imu", "--mono-imu", dest="mono_imu", action="store_true", help="Run Monocular + IMU tightly-coupled VIO mode")
    parser.add_argument("--live", action="store_true", help="Run live USB Camera (Stereo or Monocular) + Serial IMU mode")
    parser.add_argument("--mode", type=str, default="", help="Run mode alias (e.g. 'live', 'mono', 'mono_imu', 'history')")
    parser.add_argument("--history", action="store_true", help="Review historical blackbox run logs and exit")
    parser.add_argument("--cam_id", "--cam-id", dest="cam_id", type=int, default=-1, help="USB camera device ID (-1 for auto-detect)")
    parser.add_argument("--imu_port", "--imu-port", dest="imu_port", type=str, default="COM3", help="Serial port for IMU (auto-scans COM4/COM3)")
    parser.add_argument("--imu_baud", "--imu-baud", dest="imu_baud", type=int, default=115200, help="Baud rate for serial IMU (default 115200, auto-negotiates 9600->115200 @ 200Hz)")
    parser.add_argument("--calib", type=str, default="config/custom_stereo_imu.json", help="Calibration JSON path")
    parser.add_argument("--show", action="store_true", help="Show live OpenCV HUD window during --live mode")
    parser.add_argument("--max_frames", "--max-frames", dest="max_frames", type=int, default=80, help="Number of frames to process (for dataset/synthetic modes)")
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=str, default="outputs", help="Unified directory for all output files")
    args = parser.parse_args()

    mode_str = args.mode.strip().lower()
    if mode_str == "live":
        args.live = True
    elif mode_str == "history":
        args.history = True
    elif mode_str in ("mono_imu", "mono-imu"):
        args.mono_imu = True
    elif mode_str == "mono":
        args.mono = True

    if args.history:
        print_blackbox_history(output_dir=args.output_dir)
        return

    verify_core_operators()
    if args.live:
        run_live_hardware_slam(
            cam_id=args.cam_id,
            imu_port=args.imu_port,
            imu_baud=args.imu_baud,
            calib_json=args.calib,
            max_frames=args.max_frames,
            output_dir=args.output_dir,
            show_gui=True,
            force_monocular=bool(args.mono or args.mono_imu),
        )
    elif args.euroc and os.path.isdir(args.euroc):
        if args.mono_imu:
            run_euroc_monocular_inertial_sequence(args.euroc, max_frames=args.max_frames, output_dir=args.output_dir)
        elif args.mono:
            run_euroc_monocular_sequence(args.euroc, max_frames=args.max_frames, output_dir=args.output_dir)
        else:
            run_euroc_sequence(args.euroc, max_frames=args.max_frames, output_dir=args.output_dir)
    else:
        run_synthetic_benchmark(num_frames=args.max_frames, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
