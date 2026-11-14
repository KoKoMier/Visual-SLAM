# -*- coding: utf-8 -*-
"""
dataset.py
==========
数据集加载器、高保真双目+IMU 闭环仿真生成器与实机 USB 双目+串口 IMU 实时采集流
1. EuRoCDataset: 标准 EuRoC MAV 数据集加载器 (支持 mav0/cam0, cam1, imu0, state_groundtruth_estimate0)。
2. SyntheticStereoInertialDataset: 生成带丰富多尺度纹理的 3D 室内外场景双目图像流、200Hz IMU 测量与精确真值轨迹，
   用于零外部依赖下一键验证四叉树 ORB、极线立体匹配、光流跟踪、舒尔补 Local BA 与回环闭合全流程。
3. WitMotionSerialIMU & LiveStereoInertialStream: 实机 USB 左右拼接双目摄像头 (1280x480) + CH340 串口 IMU (0x55 协议) 实时同步数据流。
"""

# Python 3.6 compatible
import os
import csv
import struct
import time
import threading
import ctypes
from ctypes import wintypes
import numpy as np
import cv2

from .core.lie_algebra import SE3, exp_so3, log_so3
from .core.camera import StereoCamera
from .frontend.imu_preintegration import ImuMeasurement


class EuRoCDataset:
    """
    标准 EuRoC MAV 数据集读取器 (如 MH_01_easy/mav0)
    """
    def __init__(self, mav0_dir):
        self.mav0_dir = mav0_dir
        self.cam0_csv = os.path.join(mav0_dir, "cam0", "data.csv")
        self.cam0_img_dir = os.path.join(mav0_dir, "cam0", "data")
        self.cam1_img_dir = os.path.join(mav0_dir, "cam1", "data")
        self.imu_csv = os.path.join(mav0_dir, "imu0", "data.csv")
        self.gt_csv = os.path.join(mav0_dir, "state_groundtruth_estimate0", "data.csv")

        self.frames = []
        self.imu_data = []
        self.gt_timestamps = np.zeros(0, dtype=np.float64)
        self.gt_positions = np.zeros((0, 3), dtype=np.float64)

        self._load_metadata()

    def _load_metadata(self) :
        if not os.path.exists(self.cam0_csv):
            raise FileNotFoundError(f"未找到 EuRoC cam0 数据文件: {self.cam0_csv}")

        with open(self.cam0_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2 or row[0].startswith("#"):
                    continue
                ts_ns = int(row[0].strip())
                fname = row[1].strip()
                path_l = os.path.join(self.cam0_img_dir, fname)
                path_r = os.path.join(self.cam1_img_dir, fname)
                if os.path.exists(path_l) and os.path.exists(path_r):
                    self.frames.append((ts_ns * 1e-9, path_l, path_r))

        if os.path.exists(self.imu_csv):
            with open(self.imu_csv, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 7 or row[0].startswith("#"):
                        continue
                    ts = int(row[0].strip()) * 1e-9
                    gyro = np.array([float(row[1]), float(row[2]), float(row[3])], dtype=np.float64)
                    acc = np.array([float(row[4]), float(row[5]), float(row[6])], dtype=np.float64)
                    self.imu_data.append(ImuMeasurement(timestamp=ts, acc=acc, gyro=gyro))

        if os.path.exists(self.gt_csv):
            gt_ts = []
            gt_pos = []
            t_b_c0 = np.array([-0.0216401454975, -0.064676986768, 0.00981073058949], dtype=np.float64)
            with open(self.gt_csv, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 8 or row[0].startswith("#"):
                        continue
                    gt_ts.append(int(row[0].strip()) * 1e-9)
                    p_wb = np.array([float(row[1]), float(row[2]), float(row[3])], dtype=np.float64)
                    qw, qx, qy, qz = float(row[4]), float(row[5]), float(row[6]), float(row[7])
                    # 四元数转旋转矩阵 R_wb，计算左目相机光心真实世界坐标 p_wc = p_wb + R_wb * t_b_c0
                    R_wb = np.array([
                        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
                        [2*(qx*qy + qw*qz),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
                        [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy)]
                    ], dtype=np.float64)
                    p_wc = p_wb + (R_wb @ t_b_c0)
                    gt_pos.append(p_wc)
            self.gt_timestamps = np.asarray(gt_ts, dtype=np.float64)
            self.gt_positions = np.asarray(gt_pos, dtype=np.float64)

    def __len__(self) :
        return len(self.frames)

    def get_item(self, idx) :
        ts, path_l, path_r = self.frames[idx]
        # 兼容 OpenCV 3.4.2 在含中文路径 (视觉slam) 下的图像读取
        img_l = cv2.imdecode(np.fromfile(path_l, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imdecode(np.fromfile(path_r, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        prev_ts = self.frames[idx - 1][0] if idx > 0 else ts - 0.05
        imus = [m for m in self.imu_data if prev_ts < m.timestamp <= ts]

        gt_pos = None
        if len(self.gt_timestamps) > 0:
            nearest_i = int(np.argmin(np.abs(self.gt_timestamps - ts)))
            if abs(self.gt_timestamps[nearest_i] - ts) < 0.05:
                gt_pos = self.gt_positions[nearest_i].copy()

        return img_l, img_r, ts, imus, gt_pos


class SyntheticStereoInertialDataset:
    """
    高真实感 3D 纹理场景双目 + IMU 仿真序列生成器
    生成具有丰富 FAST/ORB 纹理斑块的 3D 空间地标，渲染无畸变极线对齐的左右目灰度图像与 200Hz IMU 数据。
    """
    def __init__(
        self,
        camera,
        num_frames = 80,
        num_landmarks = 450,
        seed = 2026
    ):
        self.camera = camera
        self.num_frames = int(num_frames)
        self.rng = np.random.default_rng(seed)

        # 1. 生成围绕相机轨迹分布的 3D 纹理特征簇与随机结构斑块
        self.landmarks_w, self.landmark_patterns = self._generate_3d_textured_world(num_landmarks)

        # 2. 生成平滑的 3D 闭环/曲线相机运动真值轨迹 T_wc(t)
        self.timestamps, self.gt_poses_wc, self.imu_batches = self._generate_trajectory_and_imu()

    def _generate_3d_textured_world(self, n_landmarks) :
        """在环形走廊空间生成 3D 结构点及每个点的唯一局部像素微纹理模板 (9x9)"""
        pts = []
        # 分布在相机前方及环线四周 (深度 1.5m ~ 7.5m)
        for _ in range(n_landmarks):
            angle = self.rng.uniform(-np.pi, np.pi)
            radius = self.rng.uniform(2.2, 6.2)
            x = radius * np.sin(angle)
            z = 3.0 + radius * np.cos(angle)
            y = self.rng.uniform(-1.6, 1.6)
            pts.append([x, y, z])
        landmarks_w = np.asarray(pts, dtype=np.float64)

        # 为每个 3D 点生成独特的 9x9 高对比度棋盘/角点微纹理，使 FAST 与 ORB 描述子具有稳定可区分性
        patterns = self.rng.integers(40, 235, size=(n_landmarks, 9, 9), dtype=np.uint8)
        for i in range(n_landmarks):
            # 增强中心角点梯度以触发 FAST 检测器
            if i % 2 == 0:
                patterns[i, 3:6, 3:6] = 250
                patterns[i, 0:2, :] = 30
                patterns[i, :, 0:2] = 30
            else:
                patterns[i, 3:6, 3:6] = 15
                patterns[i, 0:2, :] = 225
                patterns[i, :, 0:2] = 225
        return landmarks_w, patterns

    def _generate_trajectory_and_imu(self) :
        """生成 20 FPS 相机平滑 6DoF 轨迹与帧间 200Hz IMU 测量"""
        dt_cam = 1.0 / self.camera.fps
        timestamps = np.arange(self.num_frames, dtype=np.float64) * dt_cam
        gt_poses_wc = []

        # 设计平滑的椭圆 + 波浪 3D 轨迹，首尾相接形成闭环
        for k, t in enumerate(timestamps):
            phase = 2.0 * np.pi * (float(k) / float(self.num_frames))
            tx = 1.2 * np.sin(phase)
            ty = 0.25 * np.sin(2.0 * phase)
            tz = 1.0 * (1.0 - np.cos(phase))
            pos = np.array([tx, ty, tz], dtype=np.float64)

            yaw = -0.35 * np.sin(phase)
            pitch = 0.08 * np.cos(phase)
            roll = 0.05 * np.sin(2.0 * phase)
            R_wc = exp_so3(np.array([pitch, yaw, roll], dtype=np.float64))
            gt_poses_wc.append(SE3(R_wc, pos))

        # 根据解析位姿差分生成帧间 200Hz IMU 测量 (每个相机间隔 10 条 IMU)
        g_w = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        imu_batches = [[]]
        n_sub = 10

        for k in range(1, self.num_frames):
            T_prev = gt_poses_wc[k - 1]
            T_curr = gt_poses_wc[k]
            t0 = timestamps[k - 1]
            dt_sub = dt_cam / n_sub

            # 帧间平均角速度与加速度
            omega_body = log_so3(T_prev.R.T @ T_curr.R) / dt_cam
            v_prev = (T_curr.t - T_prev.t) / dt_cam if k == 1 else (gt_poses_wc[k - 1].t - gt_poses_wc[k - 2].t) / dt_cam
            v_curr = (T_curr.t - T_prev.t) / dt_cam
            a_world = (v_curr - v_prev) / dt_cam - g_w

            batch = []
            for s in range(1, n_sub + 1):
                ts_imu = t0 + s * dt_sub
                alpha = float(s) / float(n_sub)
                R_interp = T_prev.R @ exp_so3(omega_body * (alpha * dt_cam))
                acc_body = R_interp.T @ a_world
                batch.append(ImuMeasurement(timestamp=ts_imu, acc=acc_body, gyro=omega_body))
            imu_batches.append(batch)

        return timestamps, gt_poses_wc, imu_batches

    def render_stereo_pair(self, T_wc) :
        """根据给定相机位姿 T_wc 渲染左目与右目灰度图像"""
        h, w = self.camera.height, self.camera.width
        img_l = np.full((h, w), 115, dtype=np.uint8)
        img_r = np.full((h, w), 115, dtype=np.uint8)

        T_cw = T_wc.inverse()
        pts_c = T_cw.transform(self.landmarks_w)

        # 按深度从远到近排序绘制 (画家算法处理遮挡)
        valid_mask = (pts_c[:, 2] > 0.6) & (pts_c[:, 2] < 12.0)
        valid_indices = np.nonzero(valid_mask)[0]
        order = valid_indices[np.argsort(-pts_c[valid_indices, 2])]

        fx, fy, cx, cy, bf = (
            self.camera.fx, self.camera.fy,
            self.camera.cx, self.camera.cy, self.camera.bf
        )

        for idx in order:
            X, Y, Z = pts_c[idx]
            inv_z = 1.0 / Z
            ul = int(np.round(fx * X * inv_z + cx))
            vl = int(np.round(fy * Y * inv_z + cy))
            ur = int(np.round(ul - bf * inv_z))

            if 6 <= vl < h - 6:
                pat = self.landmark_patterns[idx]
                if 6 <= ul < w - 6:
                    img_l[vl - 4 : vl + 5, ul - 4 : ul + 5] = pat
                if 6 <= ur < w - 6:
                    img_r[vl - 4 : vl + 5, ur - 4 : ur + 5] = pat

        return img_l, img_r


class _Win32DCB(ctypes.Structure):
    _fields_ = [
        ("DCBlength", wintypes.DWORD),
        ("BaudRate", wintypes.DWORD),
        ("fBitFields", wintypes.DWORD),
        ("wReserved", wintypes.WORD),
        ("XonLim", wintypes.WORD),
        ("XoffLim", wintypes.WORD),
        ("ByteSize", wintypes.BYTE),
        ("Parity", wintypes.BYTE),
        ("StopBits", wintypes.BYTE),
        ("XonChar", ctypes.c_char),
        ("XoffChar", ctypes.c_char),
        ("ErrorChar", ctypes.c_char),
        ("EofChar", ctypes.c_char),
        ("EvtChar", ctypes.c_char),
        ("wReserved1", wintypes.WORD),
    ]


class _Win32CommTimeouts(ctypes.Structure):
    _fields_ = [
        ("ReadIntervalTimeout", wintypes.DWORD),
        ("ReadTotalTimeoutMultiplier", wintypes.DWORD),
        ("ReadTotalTimeoutConstant", wintypes.DWORD),
        ("WriteTotalTimeoutMultiplier", wintypes.DWORD),
        ("WriteTotalTimeoutConstant", wintypes.DWORD),
    ]


class WitMotionSerialIMU:
    """
    维特智能 (WitMotion JY901 / WT901 / JY61, CH340 串口) 实时后台 IMU 接收与解析器
    协议帧格式 (11 字节, 0x55 帧头):
      - 0x55 0x51: 三轴加速度 Ax, Ay, Az (±16g -> m/s^2) + 温度 T (°C)
      - 0x55 0x52: 三轴角速度 Wx, Wy, Wz (±2000 deg/s -> rad/s)
      - 0x55 0x53: 三轴欧拉角 Roll, Pitch, Yaw (±180 deg)
    支持 Windows 原生 Win32 API (零第三方串口库依赖) 与 Linux/RK3566 pyserial 双后端。
    """

    def __init__(
        self,
        port = "COM3",
        baud = 9600,
        R_mount = None,
    ):
        self.port = str(port)
        self.baud = int(baud)
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._win_handle = None
        self._pyserial_obj = None

        # 默认采用水平安装单位阵 R_mount = I_3；启动后将根据重力向量与静态欧拉角自动精化水平归零矩阵
        if R_mount is not None:
            self.R_mount = np.asarray(R_mount, dtype=np.float64).reshape(3, 3)
        else:
            self.R_mount = np.eye(3, dtype=np.float64)

        self._measurements = []
        self.latest_raw_acc = np.array([0.0, 0.0, 9.80665], dtype=np.float64)
        self.latest_raw_gyro = np.zeros(3, dtype=np.float64)
        self.latest_raw_euler_deg = np.zeros(3, dtype=np.float64)
        self.latest_acc = np.array([0.0, 0.0, 9.80665], dtype=np.float64)
        self.latest_gyro = np.zeros(3, dtype=np.float64)
        self.latest_euler_deg = np.zeros(3, dtype=np.float64)
        self.latest_temp_c = 25.0
        self.total_packets = 0
        self.is_connected = False
        self._mount_auto_checked = False
        self._mount_calib_count = 0
        self._ref_tilt_deg = np.zeros(2, dtype=np.float64)

    @staticmethod
    def euler_deg_to_R(eul_deg) :
        """ZYX 欧拉角 (Roll, Pitch, Yaw, 单位: deg) 转 3x3 旋转矩阵 R = Rz(yaw) @ Ry(pitch) @ Rx(roll)"""
        r, p, y = np.deg2rad(eul_deg[0]), np.deg2rad(eul_deg[1]), np.deg2rad(eul_deg[2])
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
        Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
        Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        return Rz @ Ry @ Rx

    @staticmethod
    def R_to_euler_deg(R) :
        """3x3 旋转矩阵反解 ZYX 欧拉角 (Roll, Pitch, Yaw, 单位: deg)"""
        sy = float(np.clip(-R[2, 0], -1.0, 1.0))
        pitch = np.arcsin(sy)
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        return np.rad2deg(np.array([roll, pitch, yaw], dtype=np.float64))

    def start(self) :
        """打开串口并启动后台高频接收线程"""
        if self._running:
            return self.is_connected

        if os.name == "nt":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            candidate_ports = [self.port] + [
                p for p in ("COM4", "COM3", "COM5", "COM6", "COM7", "COM8", "COM1", "COM2") if p != self.port
            ]
            h = -1
            first_opened_h = -1
            first_opened_port = self.port
            for cand in candidate_ports:
                h_try = kernel32.CreateFileW(
                    f"\\\\.\\{cand}",
                    0x80000000 | 0x40000000,  # GENERIC_READ | GENERIC_WRITE
                    0,
                    None,
                    3,  # OPEN_EXISTING
                    0,
                    None,
                )
                if h_try in (-1, ctypes.c_void_p(-1).value, None):
                    continue

                dcb = _Win32DCB()
                dcb.DCBlength = ctypes.sizeof(_Win32DCB)
                if not kernel32.GetCommState(h_try, ctypes.byref(dcb)):
                    kernel32.CloseHandle(h_try)
                    continue
                dcb.BaudRate = self.baud
                dcb.ByteSize = 8
                dcb.Parity = 0
                dcb.StopBits = 0
                dcb.fBitFields = 1
                if not kernel32.SetCommState(h_try, ctypes.byref(dcb)):
                    kernel32.CloseHandle(h_try)
                    continue

                timeouts = _Win32CommTimeouts(5, 1, 15, 1, 15)
                kernel32.SetCommTimeouts(h_try, ctypes.byref(timeouts))
                kernel32.PurgeComm(h_try, 0x0008 | 0x0004)

                # 探测该串口是否有真实 0x55 维特智能 IMU 数据流到达 (避免误连无数据的虚拟串口)
                probe_chunk = ctypes.create_string_buffer(128)
                n_probe = wintypes.DWORD(0)
                got_wit_pkt = False
                t_probe = time.perf_counter()
                while (time.perf_counter() - t_probe) < 0.15:
                    ok_r = kernel32.ReadFile(h_try, probe_chunk, 128, ctypes.byref(n_probe), None)
                    if ok_r and n_probe.value > 0:
                        raw_bytes = probe_chunk.raw[: n_probe.value]
                        if 0x55 in raw_bytes:
                            got_wit_pkt = True
                            break
                    time.sleep(0.01)

                if got_wit_pkt:
                    if first_opened_h not in (-1, ctypes.c_void_p(-1).value, None):
                        kernel32.CloseHandle(first_opened_h)
                    h = h_try
                    self.port = cand
                    break
                else:
                    if first_opened_h in (-1, ctypes.c_void_p(-1).value, None):
                        first_opened_h = h_try
                        first_opened_port = cand
                    else:
                        kernel32.CloseHandle(h_try)

            if h in (-1, ctypes.c_void_p(-1).value, None):
                if first_opened_h not in (-1, ctypes.c_void_p(-1).value, None):
                    h = first_opened_h
                    self.port = first_opened_port
                else:
                    self.is_connected = False
                    return False

            self._win_handle = h
            self.is_connected = True
            # 发送 WitMotion 临时解锁指令 + 20Hz 回传速率配置 (不写入掉电 EEPROM，仅本次会话提升帧间 IMU 采样密度至 20Hz)
            try:
                n_written = wintypes.DWORD(0)
                unlock_cmd = b"\xFF\xAA\x69\x88\xB5"
                rate_20hz_cmd = b"\xFF\xAA\x03\x07\x00"
                kernel32.WriteFile(self._win_handle, unlock_cmd, len(unlock_cmd), ctypes.byref(n_written), None)
                time.sleep(0.015)
                kernel32.WriteFile(self._win_handle, rate_20hz_cmd, len(rate_20hz_cmd), ctypes.byref(n_written), None)
            except Exception:
                pass
        else:
            try:
                import glob as _glob
                import serial
                linux_cands = []
                if self.port and not str(self.port).upper().startswith("COM"):
                    linux_cands.append(self.port)
                for p_cand in sorted(_glob.glob("/dev/ttyUSB*")) + sorted(_glob.glob("/dev/ttyACM*")) + ["/dev/ttyS3", "/dev/ttyS4", "/dev/ttyS2"]:
                    if p_cand not in linux_cands:
                        linux_cands.append(p_cand)
                first_ser = None
                for p_cand in linux_cands:
                    try:
                        ser = serial.Serial(p_cand, self.baud, timeout=0.18)
                        probe = ser.read(64)
                        if probe and (0x55 in probe):
                            self._pyserial_obj = ser
                            self.port = p_cand
                            if first_ser is not None and first_ser is not ser:
                                first_ser.close()
                            break
                        if first_ser is None:
                            first_ser = ser
                            self.port = p_cand
                        else:
                            ser.close()
                    except Exception:
                        continue
                if self._pyserial_obj is None and first_ser is not None:
                    self._pyserial_obj = first_ser
                if self._pyserial_obj is None:
                    self.is_connected = False
                    return False
                self._pyserial_obj.timeout = 0.02
                self.is_connected = True
                try:
                    self._pyserial_obj.write(b"\xFF\xAA\x69\x88\xB5")
                    time.sleep(0.015)
                    self._pyserial_obj.write(b"\xFF\xAA\x03\x07\x00")
                except Exception:
                    pass
            except Exception:
                self.is_connected = False
                return False

        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        return True

    def _read_loop(self) :
        buf = bytearray()
        has_acc = False

        if os.name == "nt":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            chunk = ctypes.create_string_buffer(256)
            n_read = wintypes.DWORD(0)

            while self._running and self._win_handle is not None:
                ok = kernel32.ReadFile(self._win_handle, chunk, 256, ctypes.byref(n_read), None)
                if ok and n_read.value > 0:
                    buf.extend(chunk.raw[: n_read.value])
                    has_acc = self._parse_buffer(buf, has_acc)
                else:
                    time.sleep(0.002)
        else:
            while self._running and self._pyserial_obj is not None:
                data = self._pyserial_obj.read(256)
                if data:
                    buf.extend(data)
                    has_acc = self._parse_buffer(buf, has_acc)
                else:
                    time.sleep(0.002)

    def _parse_buffer(self, buf, has_acc) :
        while len(buf) >= 11:
            if buf[0] != 0x55:
                del buf[0]
                continue
            pkt = bytes(buf[:11])
            if (sum(pkt[:10]) & 0xFF) != pkt[10]:
                del buf[0]
                continue
            del buf[:11]

            ptype = pkt[1]
            v0, v1, v2, v3 = struct.unpack("<hhhh", pkt[2:10])
            ts_now = time.perf_counter()

            with self._lock:
                self.total_packets += 1
                if ptype == 0x51:  # 加速度帧 (±16g -> m/s^2)
                    ax = (v0 / 32768.0) * 16.0 * 9.80665
                    ay = (v1 / 32768.0) * 16.0 * 9.80665
                    az = (v2 / 32768.0) * 16.0 * 9.80665
                    raw_acc = np.array([ax, ay, az], dtype=np.float64)
                    acc_norm = float(np.linalg.norm(raw_acc))
                    self.latest_raw_acc = raw_acc
                    if self._mount_calib_count == 0 and 8.2 < acc_norm < 11.5 and az > 6.0:
                        r_acc = float(np.rad2deg(np.arctan2(ay, az)))
                        p_acc = float(np.rad2deg(np.arctan2(-ax, np.hypot(ay, az))))
                        self._ref_tilt_deg = np.array([r_acc, p_acc], dtype=np.float64)
                        self.R_mount = self.euler_deg_to_R(np.array([r_acc, p_acc, 0.0], dtype=np.float64))
                    acc_leveled = self.R_mount @ raw_acc
                    tilt_gyro = float(np.hypot(self.latest_gyro[0], self.latest_gyro[1]))
                    if (
                        abs(self.latest_euler_deg[0]) < 6.0
                        and abs(self.latest_euler_deg[1]) < 6.0
                        and tilt_gyro < 0.30
                        and 8.2 < acc_norm < 11.5
                    ):
                        acc_leveled[0] *= 0.12
                        acc_leveled[1] *= 0.12
                    self.latest_acc = acc_leveled
                    self.latest_temp_c = float(v3) / 100.0
                    has_acc = True
                elif ptype == 0x52:  # 角速度帧 (±2000 deg/s -> rad/s)
                    wx = np.deg2rad((v0 / 32768.0) * 2000.0)
                    wy = np.deg2rad((v1 / 32768.0) * 2000.0)
                    wz = np.deg2rad((v2 / 32768.0) * 2000.0)
                    self.latest_raw_gyro = np.array([wx, wy, wz], dtype=np.float64)
                    gyro_leveled = self.R_mount @ self.latest_raw_gyro
                    # 水平桌面重力约束：当加速度计与姿态角确认设备处于水平面 (|Roll|<6°, |Pitch|<6°) 且无显著翻滚/俯仰角速度
                    # (或处于快速水平偏航旋转 |Wz| 主导状态) 时，抑制贴装倾角在偏航旋转时向 Roll/Pitch 轴产生的正交耦合分量
                    tilt_gyro_lvl = float(np.hypot(gyro_leveled[0], gyro_leveled[1]))
                    yaw_dominant = abs(gyro_leveled[2]) > 1.8 * tilt_gyro_lvl
                    if (
                        abs(self.latest_euler_deg[0]) < 6.0
                        and abs(self.latest_euler_deg[1]) < 6.0
                        and (tilt_gyro_lvl < 0.35 or yaw_dominant)
                    ):
                        gyro_leveled[0] = 0.0 if abs(gyro_leveled[0]) < 0.04 else gyro_leveled[0] * 0.04
                        gyro_leveled[1] = 0.0 if abs(gyro_leveled[1]) < 0.04 else gyro_leveled[1] * 0.04
                    self.latest_gyro = gyro_leveled
                    if has_acc or self.total_packets > 2:
                        self._measurements.append(
                            ImuMeasurement(
                                timestamp=ts_now,
                                acc=self.latest_acc.copy(),
                                gyro=self.latest_gyro.copy(),
                            )
                        )
                        if len(self._measurements) > 2000:
                            del self._measurements[: len(self._measurements) - 1000]
                elif ptype == 0x53:  # 欧拉角帧 (±180 deg)
                    roll = (v0 / 32768.0) * 180.0
                    pitch = (v1 / 32768.0) * 180.0
                    yaw = (v2 / 32768.0) * 180.0
                    # 校验串口包物理合理性，拒绝断包错位导致的异常跳变角 (如 -75° 伪包)
                    if abs(roll) > 65.0 or abs(pitch) > 65.0:
                        continue
                    self.latest_raw_euler_deg = np.array([roll, pitch, yaw], dtype=np.float64)
                    tilt_gyro = float(np.hypot(self.latest_gyro[0], self.latest_gyro[1]))
                    acc_norm = float(np.linalg.norm(self.latest_raw_acc))
                    if self._mount_calib_count == 0 and abs(roll) < 40.0 and abs(pitch) < 40.0:
                        self._ref_tilt_deg = np.array([roll, pitch], dtype=np.float64)
                        self.R_mount = self.euler_deg_to_R(np.array([roll, pitch, 0.0], dtype=np.float64))
                        self._mount_calib_count = 1
                        self._mount_auto_checked = True
                    elif self._mount_calib_count < 15 and tilt_gyro < 0.30 and abs(roll) < 40.0 and abs(pitch) < 40.0:
                        alpha = 0.35
                        self._ref_tilt_deg = (1.0 - alpha) * self._ref_tilt_deg + alpha * np.array([roll, pitch], dtype=np.float64)
                        self.R_mount = self.euler_deg_to_R(
                            np.array([self._ref_tilt_deg[0], self._ref_tilt_deg[1], 0.0], dtype=np.float64)
                        )
                        self._mount_calib_count += 1
                    elif (
                        tilt_gyro < 0.28
                        and 8.5 < acc_norm < 11.2
                        and abs(roll - self._ref_tilt_deg[0]) < 12.0
                        and abs(pitch - self._ref_tilt_deg[1]) < 12.0
                    ):
                        # 在水平面平移或绕 Z 轴偏航旋转时，自适应跟踪桌面/贴装基准倾角，消除偏航旋转 180° 时的斜面倾角反转耦合
                        alpha = 0.20
                        self._ref_tilt_deg = (1.0 - alpha) * self._ref_tilt_deg + alpha * np.array([roll, pitch], dtype=np.float64)
                        self.R_mount = self.euler_deg_to_R(
                            np.array([self._ref_tilt_deg[0], self._ref_tilt_deg[1], 0.0], dtype=np.float64)
                        )
                    rel_roll = float(roll - self._ref_tilt_deg[0])
                    rel_pitch = float(pitch - self._ref_tilt_deg[1])
                    if abs(rel_roll) < 4.5 and abs(rel_pitch) < 4.5 and tilt_gyro < 0.30:
                        rel_roll *= 0.12
                        rel_pitch *= 0.12
                    self.latest_euler_deg = np.array([rel_roll, rel_pitch, yaw], dtype=np.float64)
        return has_acc

    def get_measurements_between(self, t_prev, t_curr) :
        """获取 (t_prev, t_curr] 区间内的所有 IMU 采样；若间隔较短尚未到达新串口包，则用最新状态零阶保持补齐"""
        with self._lock:
            batch = [m for m in self._measurements if t_prev < m.timestamp <= t_curr]
            # 清理过旧的历史缓存
            self._measurements = [m for m in self._measurements if m.timestamp > t_prev - 1.0]
            if len(batch) == 0 and self.total_packets > 0:
                batch = [
                    ImuMeasurement(
                        timestamp=t_curr,
                        acc=self.latest_acc.copy(),
                        gyro=self.latest_gyro.copy(),
                    )
                ]
            return batch

    def get_status_snapshot(self) :
        with self._lock:
            return {
                "connected": self.is_connected,
                "port": self.port,
                "baud": self.baud,
                "total_packets": int(self.total_packets),
                "acc_mps2": self.latest_acc.copy(),
                "gyro_rads": self.latest_gyro.copy(),
                "euler_deg": self.latest_euler_deg.copy(),
                "raw_acc_mps2": self.latest_raw_acc.copy(),
                "raw_gyro_rads": self.latest_raw_gyro.copy(),
                "raw_euler_deg": self.latest_raw_euler_deg.copy(),
                "temp_c": float(self.latest_temp_c),
            }

    def close(self) :
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        if os.name == "nt" and self._win_handle is not None:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CloseHandle(self._win_handle)
            self._win_handle = None
        elif self._pyserial_obj is not None:
            try:
                self._pyserial_obj.close()
            except Exception:
                pass
            self._pyserial_obj = None


def open_camera_capture(cam_id):
    """兼容 OpenCV 3.4.2 (单参数 VideoCapture(id))、OpenCV 4.x 与 RK3566 Linux V4L2 的摄像头打开函数"""
    cid = int(cam_id)
    if os.name == "nt":
        try:
            cap = cv2.VideoCapture(cid, cv2.CAP_DSHOW)
            if cap is not None and cap.isOpened():
                return cap
            if cap is not None:
                cap.release()
        except TypeError:
            cap = cv2.VideoCapture(cid)
            if cap is not None and cap.isOpened():
                return cap
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(cid + int(cv2.CAP_DSHOW))
            if cap is not None and cap.isOpened():
                return cap
            if cap is not None:
                cap.release()
    else:
        try:
            v4l2_flag = getattr(cv2, "CAP_V4L2", 200)
            cap = cv2.VideoCapture(cid, v4l2_flag)
            if cap is not None and cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                except Exception:
                    pass
                return cap
            if cap is not None:
                cap.release()
        except TypeError:
            pass
    cap = cv2.VideoCapture(cid)
    if os.name != "nt" and cap is not None and cap.isOpened():
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
    return cap


def find_stereo_camera_id(
    preferred_id = -1,
    concat_width = 1280,
    height = 480,
) :
    """
    自动探测外部 USB 双目摄像头 (1280x480) 或外置 USB 单目摄像头 (640x480):
    若 preferred_id >= 0 则优先验证该设备；若 preferred_id == -1 则优先扫描外置摄像头索引 [1, 2, 3]，
    仅当所有外置摄像头均未接入时才回退至笔记本内置摄像头 (索引 0)。
    """
    if preferred_id >= 0:
        return int(preferred_id)

    best_id = None
    for cid in [1, 2, 3, 0]:
        cap = open_camera_capture(cid)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, concat_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            cap2 = open_camera_capture(cid)
            if cap2.isOpened():
                ret, frame = cap2.read()
            cap2.release()
        if ret and frame is not None:
            h_actual, w_actual = frame.shape[:2]
            if w_actual == concat_width and h_actual == height and cid != 0:
                return cid
            if best_id is None and float(frame.std()) > 3.0 and float(frame.mean()) > 8.0:
                best_id = cid
    return best_id if best_id is not None else 1


class LiveStereoInertialStream:
    """
    实机 USB 双目摄像头 (1280x480) / 单目摄像头 (640x480) + 串口 IMU 实时采集器
    自动识别当前接入的是左右拼接双目摄像头还是单目摄像头 (self.is_monocular)。
    """

    def __init__(
        self,
        cam_id = -1,
        concat_width = 1280,
        height = 480,
        imu_port = "COM3",
        imu_baud = 9600,
        calib_json = "config/custom_stereo_imu.json",
        force_monocular = False,
    ):
        self.concat_width = int(concat_width)
        self.height = int(height)
        self.half_w = self.concat_width // 2
        self.cam_id = find_stereo_camera_id(cam_id, concat_width=self.concat_width, height=self.height)
        self.calib_json = calib_json
        self.is_monocular = bool(force_monocular)

        self.imu_bg = np.zeros(3, dtype=np.float64)
        self.imu_ba = np.zeros(3, dtype=np.float64)
        self.R_imu_mount = None

        # 1. 构建双目相机模型 (若存在标定文件则加载真实标定参数，否则采用 640x480 标准初始内参)
        if calib_json and os.path.exists(calib_json):
            import json
            self.camera = StereoCamera.from_calib_file(calib_json)
            with open(calib_json, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if "imu_bg" in cfg:
                self.imu_bg = np.asarray(cfg["imu_bg"], dtype=np.float64).reshape(3)
            if "imu_ba" in cfg:
                self.imu_ba = np.asarray(cfg["imu_ba"], dtype=np.float64).reshape(3)
            if "R_imu_mount" in cfg:
                self.R_imu_mount = np.asarray(cfg["R_imu_mount"], dtype=np.float64).reshape(3, 3)
            elif "imu_mount_rpy_deg" in cfg:
                self.R_imu_mount = WitMotionSerialIMU.euler_deg_to_R(
                    np.asarray(cfg["imu_mount_rpy_deg"], dtype=np.float64)
                )
            self.calib_source = f"Loaded calibration JSON ({calib_json})"
        else:
            self.camera = StereoCamera(
                fx=450.0,
                fy=450.0,
                cx=self.half_w * 0.5,
                cy=self.height * 0.5,
                baseline=0.060,
                width=self.half_w,
                height=self.height,
                th_depth_factor=40.0,
                fps=30.0,
            )
            self.calib_source = "Default 640x480 Stereo Intrinsics (b=6.0cm, uncalibrated)"

        # 2. 启动串口 IMU (带竖立安装旋转矩阵补偿)
        self.imu = WitMotionSerialIMU(port=imu_port, baud=imu_baud, R_mount=self.R_imu_mount)
        self.imu_ok = self.imu.start()

        # 3. 打开 USB 摄像头并自动检测双目 (1280x480) 或单目 (640x480)
        self.cap = open_camera_capture(self.cam_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 (cam_id={self.cam_id})")

        self._last_valid_frame = None
        if not self.is_monocular:
            # 仅在首次 read() 之前尝试请求 1280x480；绝不在 read() 成功后再次调用 cap.set()，以免在 OpenCV 3.4.2 下中断视频流
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.concat_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            ret_probe, frame_probe = self.cap.read()
            if not ret_probe or frame_probe is None:
                # 若摄像头不支持设置 1280x480，重新以默认分辨率打开
                self.cap.release()
                self.cap = open_camera_capture(self.cam_id)
                ret_probe, frame_probe = self.cap.read()
            if ret_probe and frame_probe is not None:
                self._last_valid_frame = frame_probe.copy()
                hp, wp = frame_probe.shape[:2]
                if wp != self.concat_width or hp != self.height:
                    # 该设备不支持 1280x480 左右拼接双目，自动切换为 640x480 单目+IMU模式 (不再调用 cap.set 中断已开启的流)
                    self.is_monocular = True
        else:
            ret_probe, frame_probe = self.cap.read()
            if ret_probe and frame_probe is not None:
                self._last_valid_frame = frame_probe.copy()

        # 预热读取几帧并缓存最新有效帧
        for _ in range(3):
            ret_w, frm_w = self.cap.read()
            if ret_w and frm_w is not None:
                self._last_valid_frame = frm_w.copy()
        if self.imu_ok:
            t_wait = time.perf_counter()
            while self.imu.total_packets < 2 and (time.perf_counter() - t_wait) < 0.35:
                time.sleep(0.01)

        self._t0 = time.perf_counter()
        self._last_ts = 0.0

        # 4. 启动后台异步摄像头抓帧线程 (消除关键帧 HF-Net 推理期间 UVC 缓冲区积压导致的延迟抖动与大光流跳变)
        self._cam_lock = threading.Lock()
        self._latest_frame = self._last_valid_frame.copy() if self._last_valid_frame is not None else None
        self._latest_frame_ts = time.perf_counter()
        self._latest_frame_seq = 0
        self._last_consumed_seq = -1
        self._cam_running = True
        self._cam_thread = threading.Thread(target=self._cam_read_loop, daemon=True)
        self._cam_thread.start()

    def _cam_read_loop(self):
        """后台高频排空摄像头硬件缓冲区，始终保存最新鲜的一帧图像及其真实曝光完成时间戳"""
        fail_streak = 0
        while self._cam_running and self.cap is not None:
            ret, frm = self.cap.read()
            if ret and frm is not None:
                ts_now = time.perf_counter()
                fail_streak = 0
                with self._cam_lock:
                    self._latest_frame = frm
                    self._latest_frame_ts = ts_now
                    self._latest_frame_seq += 1
                    self._last_valid_frame = frm
            else:
                fail_streak += 1
                if fail_streak >= 15:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = open_camera_capture(self.cam_id)
                    fail_streak = 0
                time.sleep(0.005)

    def read_frame(self) :
        """
        读取一帧同步双目或单目灰度图像与帧间 IMU 数据包:
        返回 (img_l_gray, img_r_gray, timestamp_sec, imu_measurements, imu_status_dict)
        """
        frame = None
        ts_perf = None
        # 等待后台抓帧线程产出新帧 (最长等待 45ms，若超时则立即取最新可用帧)
        t_wait0 = time.perf_counter()
        while (time.perf_counter() - t_wait0) < 0.045:
            with self._cam_lock:
                if self._latest_frame is not None and self._latest_frame_seq > self._last_consumed_seq:
                    frame = self._latest_frame.copy()
                    ts_perf = self._latest_frame_ts
                    self._last_consumed_seq = self._latest_frame_seq
                    break
            time.sleep(0.002)

        if frame is None:
            with self._cam_lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame.copy()
                    ts_perf = time.perf_counter()
                elif self._last_valid_frame is not None:
                    frame = self._last_valid_frame.copy()
                    ts_perf = time.perf_counter()

        if frame is None:
            raise RuntimeError(f"从摄像头 (cam_id={self.cam_id}) 读取图像帧失败")
        if ts_perf is None:
            ts_perf = time.perf_counter()

        if self.is_monocular:
            if frame.shape[1] != self.half_w or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.half_w, self.height))
            img_l = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
            img_r = img_l.copy()
        else:
            if frame.shape[1] != self.concat_width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.concat_width, self.height))
            bgr_l = frame[:, : self.half_w]
            bgr_r = frame[:, self.half_w :]
            img_l = cv2.cvtColor(bgr_l, cv2.COLOR_BGR2GRAY) if bgr_l.ndim == 3 else bgr_l.copy()
            img_r = cv2.cvtColor(bgr_r, cv2.COLOR_BGR2GRAY) if bgr_r.ndim == 3 else bgr_r.copy()

        rel_ts = max(self._last_ts + 1e-3, ts_perf - self._t0)
        prev_perf = self._t0 + self._last_ts if self._last_ts > 0.0 else (ts_perf - 0.033)
        imu_raw = self.imu.get_measurements_between(prev_perf, ts_perf) if self.imu_ok else []
        imu_batch = [
            ImuMeasurement(
                timestamp=max(0.0, m.timestamp - self._t0),
                acc=m.acc - self.imu_ba,
                gyro=m.gyro - self.imu_bg,
            )
            for m in imu_raw
        ]
        self._last_ts = rel_ts
        return img_l, img_r, rel_ts, imu_batch, self.imu.get_status_snapshot()

    def close(self) :
        self._cam_running = False
        if hasattr(self, "_cam_thread") and self._cam_thread is not None and self._cam_thread.is_alive():
            self._cam_thread.join(timeout=0.5)
        if self.cap is not None:
            self.cap.release()
        self.imu.close()

