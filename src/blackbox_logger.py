# -*- coding: utf-8 -*-
"""
src/blackbox_logger.py
======================
视觉/惯性 SLAM 运行时黑盒飞行记录仪 (Blackbox Flight Recorder)

每次运行自动在 `outputs/blackbox_logs/run_YYYYMMDD_HHMMSS_<mode>/` 下生成完整黑盒历史档案：
  1. `frames.csv`   : 逐帧实时流式写入 (每帧 flush) 的 35 维全状态遥测表 (位姿、速度、光流、旋转门控、IMU、耗时)
  2. `events.jsonl` : 关键事件流 (系统启动参数、初始化完成、关键帧插入、纯旋转门控触发、跟踪状态切换、运行结束汇总)
  3. `summary.json` : 结构化运行统计摘要 (每 30 帧自动增量更新 + 退出时终态写入，防断电/强杀丢失)
  4. `report.md`    : 可直接阅读的 Markdown 黑盒复盘报告 (含运动阶段分解：静止/平移/纯旋转门控期间的位移统计)
  5. `history_index.jsonl` : 全局历史运行索引表，汇总历次运行核心指标，便于横向对比回顾
"""

# Python 3.6 compatible
import csv
import datetime
import json
import os
import shutil
from typing import Any
import numpy as np


class BlackboxLogger:
    """SLAM 全状态实时黑盒记录器"""

    CSV_HEADERS = [
        "frame_idx",
        "timestamp_s",
        "dt_s",
        "state",
        "is_keyframe",
        "used_optical_flow",
        "inliers",
        "num_kps",
        "num_kfs",
        "num_mps",
        "pos_x_m",
        "pos_y_m",
        "pos_z_m",
        "step_dist_m",
        "total_travel_m",
        "cam_roll_deg",
        "cam_pitch_deg",
        "cam_yaw_deg",
        "vel_x_mps",
        "vel_y_mps",
        "vel_z_mps",
        "speed_mps",
        "trans_gate",
        "eff_gyro_deg_s",
        "raw_flow_px",
        "derot_res_px",
        "radial_med_px",
        "imu_packets_frame",
        "imu_total_packets",
        "acc_x_mps2",
        "acc_y_mps2",
        "acc_z_mps2",
        "acc_norm_mps2",
        "gyro_x_rads",
        "gyro_y_rads",
        "gyro_z_rads",
        "gyro_norm_degs",
        "imu_roll_deg",
        "imu_pitch_deg",
        "imu_yaw_deg",
        "raw_imu_roll_deg",
        "raw_imu_pitch_deg",
        "raw_imu_yaw_deg",
        "frontend_ms",
        "backend_ms",
        "total_ms",
        "inst_fps",
    ]

    def __init__(
        self,
        output_dir = "outputs",
        mode = "live_mono_imu",
        metadata = None,
    ) :
        self.base_dir = os.path.join(output_dir, "blackbox_logs")
        os.makedirs(self.base_dir, exist_ok=True)

        now = datetime.datetime.now()
        self.start_time_iso = now.isoformat(timespec="seconds")
        self.run_id = f"run_{now.strftime('%Y%m%d_%H%M%S')}_{mode}"
        self.run_dir = os.path.join(self.base_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        self.mode = mode
        self.metadata = metadata or {}

        self.csv_path = os.path.join(self.run_dir, "frames.csv")
        self.events_path = os.path.join(self.run_dir, "events.jsonl")
        self.summary_path = os.path.join(self.run_dir, "summary.json")
        self.report_path = os.path.join(self.run_dir, "report.md")
        self.index_path = os.path.join(self.base_dir, "history_index.jsonl")

        self._csv_file = open(self.csv_path, mode="w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(self.CSV_HEADERS)
        self._csv_file.flush()

        self._events_file = open(self.events_path, mode="w", encoding="utf-8")

        # 实时累积统计量
        self.frame_count = 0
        self.first_ts = None
        self.prev_ts = None
        self.prev_pos = None
        self.prev_state = "NO_IMAGES_YET"
        self.total_travel_m = 0.0

        self.pos_history = []
        self.step_history = []
        self.gate_history = []
        self.gyro_degs_history = []
        self.flow_px_history = []
        self.derot_px_history = []
        self.fe_ms_history = []
        self.be_ms_history = []
        self.tot_ms_history = []
        self.inliers_history = []
        self.acc_norm_history = []

        self.rot_gated_frames = 0
        self.rot_gated_disp_m = 0.0
        self.translation_frames = 0
        self.translation_disp_m = 0.0
        self.stationary_frames = 0
        self.last_num_kfs = 0
        self.last_num_mps = 0
        self.last_imu_packets = 0
        self._finalized = False

        self.log_event("SESSION_START", {
            "run_id": self.run_id,
            "mode": self.mode,
            "start_time": self.start_time_iso,
            "metadata": self.metadata,
        })

    @staticmethod
    def _rot_to_euler_deg(R_wc) :
        """将相机光学坐标系旋转矩阵 R_wc (X右, Y下, Z前) 转换为水平机身系 (X右, Y前, Z上) 姿态角 (Roll, Pitch, Yaw, 与 IMU 严格对齐):
        - Roll : 绕 +Y(前/光轴) 旋转 (横滚角)
        - Pitch: 绕 +X(右)      旋转 (俯仰角，上抬抬头为正)
        - Yaw  : 绕 +Z(上)      旋转 (偏航角)
        """
        R_bc = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64)
        R = R_bc @ R_wc @ R_bc.T
        sy = float(np.hypot(R[0, 0], R[1, 0]))
        if sy > 1e-6:
            roll = float(np.rad2deg(np.arctan2(-R[2, 0], sy)))
            pitch = float(np.rad2deg(np.arctan2(R[2, 1], R[2, 2])))
            yaw = float(np.rad2deg(np.arctan2(R[1, 0], R[0, 0])))
        else:
            roll = float(np.rad2deg(np.arctan2(-R[2, 0], sy)))
            pitch = float(np.rad2deg(np.arctan2(-R[1, 2], R[1, 1])))
            yaw = 0.0
        return roll, pitch, yaw

    def log_event(self, event_type, payload) :
        if self._events_file.closed:
            return
        record = {
            "wall_time": datetime.datetime.now().isoformat(timespec="milliseconds"),
            "frame_idx": self.frame_count,
            "event": event_type,
            **payload,
        }
        self._events_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._events_file.flush()

    def log_frame(
        self,
        frame_idx,
        timestamp,
        slam,
        imu_batch = None,
        imu_status = None,
    ) :
        """记录单帧全量状态并立即刷盘"""
        tracker = slam.tracker
        cf = tracker.curr_frame
        state_name = tracker.state.name

        if self.first_ts is None:
            self.first_ts = float(timestamp)
        dt_s = (timestamp - self.prev_ts) if (self.prev_ts is not None and timestamp > self.prev_ts) else 0.033
        self.prev_ts = float(timestamp)

        if cf is not None and cf.has_pose:
            T_wc = cf.get_pose_inverse()
            pos = T_wc.t.copy()
            R_wc = T_wc.R
            cam_roll, cam_pitch, cam_yaw = self._rot_to_euler_deg(R_wc)
            vel = getattr(cf, "v_w", np.zeros(3, dtype=np.float64))
            num_kps = int(cf.num_kps)
        else:
            pos = np.zeros(3, dtype=np.float64)
            cam_roll, cam_pitch, cam_yaw = 0.0, 0.0, 0.0
            vel = np.zeros(3, dtype=np.float64)
            num_kps = 0

        if self.prev_pos is not None and state_name == "OK":
            step_dist = float(np.linalg.norm(pos - self.prev_pos))
        else:
            step_dist = 0.0
        if state_name == "OK":
            self.prev_pos = pos.copy()
            self.total_travel_m += step_dist

        speed_mps = float(np.linalg.norm(vel))
        is_kf = bool(getattr(tracker, "last_is_keyframe", False))
        used_flow = bool(getattr(tracker, "last_used_optical_flow", False))
        inliers = int(getattr(tracker, "last_inliers", 0))
        num_kfs = int(slam.slam_map.num_keyframes())
        num_mps = int(slam.slam_map.num_mappoints())

        diag = getattr(tracker, "last_diag", {}) or {}
        trans_gate = float(diag.get("trans_gate", 1.0))
        eff_gyro_deg_s = float(diag.get("eff_gyro_deg_s", 0.0))
        raw_flow_px = float(diag.get("raw_flow_px", 0.0))
        derot_res_px = float(diag.get("derot_res_px", 0.0))
        radial_med_px = float(diag.get("radial_med_px", 0.0))

        imu_n = len(imu_batch) if imu_batch is not None else 0
        if imu_status is not None:
            imu_tot = int(imu_status.get("total_packets", 0))
            acc = np.asarray(imu_status.get("acc_mps2", [0.0, 0.0, 0.0]), dtype=np.float64)
            gyro = np.asarray(imu_status.get("gyro_rads", [0.0, 0.0, 0.0]), dtype=np.float64)
            eul = np.asarray(imu_status.get("euler_deg", [0.0, 0.0, 0.0]), dtype=np.float64)
            raw_eul = np.asarray(imu_status.get("raw_euler_deg", eul), dtype=np.float64)
        elif imu_batch is not None and len(imu_batch) > 0:
            imu_tot = self.last_imu_packets + len(imu_batch)
            acc = np.mean([m.acc for m in imu_batch], axis=0)
            gyro = np.mean([m.gyro for m in imu_batch], axis=0)
            eul = np.array([cam_roll, cam_pitch, cam_yaw], dtype=np.float64)
            raw_eul = eul.copy()
        else:
            imu_tot = 0
            acc = np.zeros(3, dtype=np.float64)
            gyro = np.zeros(3, dtype=np.float64)
            eul = np.array([cam_roll, cam_pitch, cam_yaw], dtype=np.float64)
            raw_eul = eul.copy()

        acc_norm = float(np.linalg.norm(acc))
        gyro_norm_degs = float(np.rad2deg(np.linalg.norm(gyro)))

        fe_ms = float(tracker.timing_stats["frontend_ms"][-1]) if tracker.timing_stats["frontend_ms"] else 0.0
        be_ms = float(tracker.timing_stats["backend_ms"][-1]) if tracker.timing_stats["backend_ms"] else 0.0
        tot_ms = float(tracker.timing_stats["total_ms"][-1]) if tracker.timing_stats["total_ms"] else max(1.0, fe_ms + be_ms)
        inst_fps = 1000.0 / max(1e-3, tot_ms)

        row = [
            frame_idx,
            f"{timestamp:.4f}",
            f"{dt_s:.4f}",
            state_name,
            int(is_kf),
            int(used_flow),
            inliers,
            num_kps,
            num_kfs,
            num_mps,
            f"{pos[0]:.5f}",
            f"{pos[1]:.5f}",
            f"{pos[2]:.5f}",
            f"{step_dist:.5f}",
            f"{self.total_travel_m:.5f}",
            f"{cam_roll:.2f}",
            f"{cam_pitch:.2f}",
            f"{cam_yaw:.2f}",
            f"{vel[0]:.4f}",
            f"{vel[1]:.4f}",
            f"{vel[2]:.4f}",
            f"{speed_mps:.4f}",
            f"{trans_gate:.3f}",
            f"{eff_gyro_deg_s:.2f}",
            f"{raw_flow_px:.2f}",
            f"{derot_res_px:.2f}",
            f"{radial_med_px:.2f}",
            imu_n,
            imu_tot,
            f"{acc[0]:.3f}",
            f"{acc[1]:.3f}",
            f"{acc[2]:.3f}",
            f"{acc_norm:.3f}",
            f"{gyro[0]:.4f}",
            f"{gyro[1]:.4f}",
            f"{gyro[2]:.4f}",
            f"{gyro_norm_degs:.2f}",
            f"{eul[0]:.2f}",
            f"{eul[1]:.2f}",
            f"{eul[2]:.2f}",
            f"{raw_eul[0]:.2f}",
            f"{raw_eul[1]:.2f}",
            f"{raw_eul[2]:.2f}",
            f"{fe_ms:.2f}",
            f"{be_ms:.2f}",
            f"{tot_ms:.2f}",
            f"{inst_fps:.1f}",
        ]
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        # 更新内存统计与事件检测
        self.frame_count += 1
        self.pos_history.append(pos.copy())
        self.step_history.append(step_dist)
        self.gate_history.append(trans_gate)
        self.gyro_degs_history.append(max(eff_gyro_deg_s, gyro_norm_degs))
        self.flow_px_history.append(raw_flow_px)
        self.derot_px_history.append(derot_res_px)
        self.fe_ms_history.append(fe_ms)
        self.be_ms_history.append(be_ms)
        self.tot_ms_history.append(tot_ms)
        self.inliers_history.append(inliers)
        if acc_norm > 0.1:
            self.acc_norm_history.append(acc_norm)
        self.last_num_kfs = num_kfs
        self.last_num_mps = num_mps
        self.last_imu_packets = imu_tot

        if trans_gate < 0.50:
            self.rot_gated_frames += 1
            self.rot_gated_disp_m += step_dist
        elif step_dist > 0.0015 or raw_flow_px > 1.5:
            self.translation_frames += 1
            self.translation_disp_m += step_dist
        else:
            self.stationary_frames += 1

        if state_name != self.prev_state:
            self.log_event("STATE_CHANGE", {
                "from_state": self.prev_state,
                "to_state": state_name,
                "timestamp_s": round(timestamp, 4),
                "pos_m": [round(float(x), 4) for x in pos],
                "num_kfs": num_kfs,
                "num_mps": num_mps,
            })
            self.prev_state = state_name

        if is_kf:
            self.log_event("KEYFRAME_CREATED", {
                "timestamp_s": round(timestamp, 4),
                "num_kfs": num_kfs,
                "num_mps": num_mps,
                "inliers": inliers,
                "pos_m": [round(float(x), 4) for x in pos],
                "cam_rpy_deg": [round(cam_roll, 1), round(cam_pitch, 1), round(cam_yaw, 1)],
                "backend_ms": round(be_ms, 2),
            })

        if trans_gate < 0.35 and (frame_idx % 5 == 0):
            self.log_event("PURE_ROTATION_GATED", {
                "timestamp_s": round(timestamp, 4),
                "trans_gate": round(trans_gate, 3),
                "eff_gyro_deg_s": round(eff_gyro_deg_s, 2),
                "raw_flow_px": round(raw_flow_px, 2),
                "derot_res_px": round(derot_res_px, 2),
                "step_dist_m": round(step_dist, 5),
                "pos_m": [round(float(x), 4) for x in pos],
            })

        # 每 30 帧增量更新一次 summary.json 与 report.md，确保随时可查阅
        if self.frame_count % 30 == 0 or self.frame_count == 1:
            self._write_summary_and_report(is_final=False)

    def _build_summary_dict(self, is_final = True) :
        pos_arr = np.asarray(self.pos_history, dtype=np.float64) if self.pos_history else np.zeros((1, 3))
        pos_min = pos_arr.min(axis=0).tolist()
        pos_max = pos_arr.max(axis=0).tolist()
        pos_span = (pos_arr.max(axis=0) - pos_arr.min(axis=0)).tolist()
        final_pos = pos_arr[-1].tolist()

        fe_arr = np.asarray(self.fe_ms_history, dtype=np.float64) if self.fe_ms_history else np.zeros(1)
        tot_arr = np.asarray(self.tot_ms_history, dtype=np.float64) if self.tot_ms_history else np.ones(1)
        inl_arr = np.asarray(self.inliers_history, dtype=np.float64) if self.inliers_history else np.zeros(1)
        gyro_arr = np.asarray(self.gyro_degs_history, dtype=np.float64) if self.gyro_degs_history else np.zeros(1)
        acc_arr = np.asarray(self.acc_norm_history, dtype=np.float64) if self.acc_norm_history else np.zeros(1)

        avg_fps = float(1000.0 / max(1e-3, float(tot_arr.mean())))
        rot_drift_per_frame_mm = (
            (self.rot_gated_disp_m * 1000.0 / self.rot_gated_frames) if self.rot_gated_frames > 0 else 0.0
        )

        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "start_time": self.start_time_iso,
            "last_updated": datetime.datetime.now().isoformat(timespec="seconds"),
            "is_final": is_final,
            "run_dir": os.path.abspath(self.run_dir),
            "files": {
                "frames_csv": os.path.abspath(self.csv_path),
                "events_jsonl": os.path.abspath(self.events_path),
                "summary_json": os.path.abspath(self.summary_path),
                "report_md": os.path.abspath(self.report_path),
            },
            "metadata": self.metadata,
            "stats": {
                "total_frames": int(self.frame_count),
                "duration_s": round(
                    float((self.prev_ts - self.first_ts) if (self.prev_ts is not None and self.first_ts is not None) else 0.0),
                    3,
                ),
                "final_state": self.prev_state,
                "active_keyframes": int(self.last_num_kfs),
                "active_mappoints": int(self.last_num_mps),
                "imu_total_packets": int(self.last_imu_packets),
                "mean_gravity_mps2": round(float(acc_arr.mean()), 4),
                "mean_inliers": round(float(inl_arr.mean()), 1),
                "min_inliers": int(inl_arr.min()),
                "frontend_mean_ms": round(float(fe_arr.mean()), 2),
                "frontend_median_ms": round(float(np.median(fe_arr)), 2),
                "average_fps": round(avg_fps, 1),
            },
            "trajectory": {
                "total_travel_m": round(float(self.total_travel_m), 4),
                "final_pos_m": [round(float(x), 4) for x in final_pos],
                "min_pos_m": [round(float(x), 4) for x in pos_min],
                "max_pos_m": [round(float(x), 4) for x in pos_max],
                "span_xyz_m": [round(float(x), 4) for x in pos_span],
            },
            "motion_breakdown": {
                "stationary_frames": int(self.stationary_frames),
                "translation_frames": int(self.translation_frames),
                "translation_distance_m": round(float(self.translation_disp_m), 4),
                "rotation_gated_frames": int(self.rot_gated_frames),
                "rotation_gated_total_drift_m": round(float(self.rot_gated_disp_m), 4),
                "rotation_gated_drift_per_frame_mm": round(float(rot_drift_per_frame_mm), 3),
                "max_gyro_deg_s": round(float(gyro_arr.max()), 2),
            },
        }

    def _write_summary_and_report(self, is_final = False) :
        summary = self._build_summary_dict(is_final=is_final)
        with open(self.summary_path, mode="w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        s = summary["stats"]
        t = summary["trajectory"]
        m = summary["motion_breakdown"]

        md_lines = [
            f"# SLAM 运行黑盒复盘报告 (`{self.run_id}`)",
            "",
            f"- **启动时间**: `{summary['start_time']}`",
            f"- **更新时间**: `{summary['last_updated']}` ({'已完成归档' if is_final else '运行中实时快照'})",
            f"- **运行模式**: `{self.mode}`",
            f"- **黑盒目录**: `{summary['run_dir']}`",
            "",
            "## 1. 核心运行指标",
            "| 指标 | 数值 |",
            "| :--- | :--- |",
            f"| 总处理帧数 (Frames) | **{s['total_frames']}** (时长 `{s['duration_s']} s`) |",
            f"| 最终跟踪状态 (State) | `{s['final_state']}` |",
            f"| 活跃关键帧 / 地图点数 | **{s['active_keyframes']} KFs** / **{s['active_mappoints']} MPs** |",
            f"| 平均跟踪内点数 (Inliers) | **{s['mean_inliers']}** (最低 `{s['min_inliers']}`) |",
            f"| 前端耗时 (均值 / 中位数) | **{s['frontend_mean_ms']} ms** / **{s['frontend_median_ms']} ms** |",
            f"| 系统平均帧率 (FPS) | **{s['average_fps']} FPS** |",
            f"| IMU 累计收包 / 重力模长 | `{s['imu_total_packets']}` 包 / `{s['mean_gravity_mps2']} m/s²` |",
            "",
            "## 2. 轨迹与旋转-平移解耦诊断",
            "| 统计项 | 数值 |",
            "| :--- | :--- |",
            f"| 累计运动总行程 (Total Travel) | **{t['total_travel_m']*100:.2f} cm** (`{t['total_travel_m']:.4f} m`) |",
            f"| 最终空间坐标 (X, Y, Z) | `({t['final_pos_m'][0]:+.4f}, {t['final_pos_m'][1]:+.4f}, {t['final_pos_m'][2]:+.4f}) m` |",
            f"| 轨迹三轴包络跨度 (ΔX, ΔY, ΔZ) | `({t['span_xyz_m'][0]:.4f}, {t['span_xyz_m'][1]:.4f}, {t['span_xyz_m'][2]:.4f}) m` |",
            f"| 静止帧数 | `{m['stationary_frames']}` 帧 |",
            f"| 有效平移帧数 / 累计平移量 | **{m['translation_frames']} 帧** / **{m['translation_distance_m']*100:.2f} cm** |",
            f"| 纯旋转门控帧数 / 期间残余位移 | **{m['rotation_gated_frames']} 帧** / **{m['rotation_gated_total_drift_m']*100:.2f} cm** (`{m['rotation_gated_drift_per_frame_mm']:.3f} mm/帧`) |",
            f"| 峰值角速度 (Max Gyro) | `{m['max_gyro_deg_s']} °/s` |",
            "",
            "## 3. 黑盒数据文件清单",
            f"- 逐帧遥测全表 (CSV): `{summary['files']['frames_csv']}`",
            f"- 关键事件时间线 (JSONL): `{summary['files']['events_jsonl']}`",
            f"- 结构化汇总 (JSON): `{summary['files']['summary_json']}`",
        ]
        with open(self.report_path, mode="w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")

        # 同步最新报告到 outputs/blackbox_logs/ 根目录方便快速打开
        shutil.copyfile(self.summary_path, os.path.join(self.base_dir, "latest_summary.json"))
        shutil.copyfile(self.report_path, os.path.join(self.base_dir, "latest_report.md"))
        return summary

    def finalize(self, artifact_paths = None) :
        """运行结束时归档图表快照、写入最终黑盒报告并追加至全局历史索引"""
        if self._finalized:
            return self.run_dir
        self._finalized = True

        archived_files = []
        if artifact_paths:
            for p in artifact_paths:
                if p and os.path.exists(p):
                    dst = os.path.join(self.run_dir, os.path.basename(p))
                    try:
                        shutil.copyfile(p, dst)
                        archived_files.append(dst)
                    except OSError:
                        pass

        summary = self._write_summary_and_report(is_final=True)
        summary["archived_artifacts"] = archived_files
        with open(self.summary_path, mode="w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        shutil.copyfile(self.summary_path, os.path.join(self.base_dir, "latest_summary.json"))

        self.log_event("SESSION_END", {
            "stats": summary["stats"],
            "trajectory": summary["trajectory"],
            "motion_breakdown": summary["motion_breakdown"],
            "archived_artifacts": archived_files,
        })

        # 追加到全局历史运行索引表
        index_entry = {
            "run_id": self.run_id,
            "start_time": self.start_time_iso,
            "mode": self.mode,
            "frames": summary["stats"]["total_frames"],
            "duration_s": summary["stats"]["duration_s"],
            "fps": summary["stats"]["average_fps"],
            "kfs": summary["stats"]["active_keyframes"],
            "mps": summary["stats"]["active_mappoints"],
            "travel_m": summary["trajectory"]["total_travel_m"],
            "span_xyz_m": summary["trajectory"]["span_xyz_m"],
            "trans_frames": summary["motion_breakdown"]["translation_frames"],
            "rot_gated_frames": summary["motion_breakdown"]["rotation_gated_frames"],
            "rot_drift_m": summary["motion_breakdown"]["rotation_gated_total_drift_m"],
            "run_dir": os.path.abspath(self.run_dir),
        }
        with open(self.index_path, mode="a", encoding="utf-8") as f:
            f.write(json.dumps(index_entry, ensure_ascii=False) + "\n")

        if not self._csv_file.closed:
            self._csv_file.close()
        if not self._events_file.closed:
            self._events_file.close()

        return self.run_dir


def print_blackbox_history(output_dir = "outputs", last_n = 10) :
    """在终端打印历史黑盒日志列表与最新一次运行的详细复盘摘要"""
    base_dir = os.path.join(output_dir, "blackbox_logs")
    index_path = os.path.join(base_dir, "history_index.jsonl")
    print("\n" + "=" * 88)
    print("[SLAM 黑盒历史日志回顾] 目录:", os.path.abspath(base_dir))
    print("=" * 88)
    if not os.path.exists(index_path):
        print("  暂无已归档的历史黑盒记录。")
        return

    entries = []
    with open(index_path, mode="r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    for e in entries[-last_n:]:
        print(
            f"  - [{e['run_id']}] | 帧数: {e['frames']:4d} ({e['duration_s']:5.1f}s, {e['fps']:4.1f} FPS) | "
            f"KFs: {e['kfs']:2d}, MPs: {e['mps']:4d} | 行程: {e['travel_m']*100:6.1f}cm | "
            f"平移帧: {e['trans_frames']:3d}, 旋转门控帧: {e['rot_gated_frames']:3d} (旋转漂移: {e['rot_drift_m']*100:4.2f}cm)"
        )

    latest_report = os.path.join(base_dir, "latest_report.md")
    if os.path.exists(latest_report):
        print("-" * 88)
        with open(latest_report, mode="r", encoding="utf-8") as f:
            print(f.read())


if __name__ == "__main__":
    print_blackbox_history()
