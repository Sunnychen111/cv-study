"""双云台对等协同智能检测系统（UDP事件去重版，单文件集成）。

本文件已集成海康 ISAPI 云台控制、RTSP 最新帧读取、YOLO 检测、
低延迟流畅跟踪、无目标自动搜索、双机联动、火源归档、操作日志
以及 save 图片/TXT 记录查看功能，不依赖项目内的其他 Python 脚本。
两端运行同一份代码，事件发起方与响应方由事件通信动态决定；响应方
持续搜寻并周期回传实时图像，双方均周期保存本机检测画面。
"""
import base64
import csv
import hashlib
import json
import math
import os
import socket
import sys
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from collections import OrderedDict, deque
from datetime import datetime
from urllib.parse import quote

if sys.platform.startswith("linux"):
    os.environ["QT_QPA_PLATFORM"] = "xcb"

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|max_delay;0",
)

import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth
from PySide6.QtCore import QEvent, QRect, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import (
    QColor,
    QDoubleValidator,
    QFont,
    QImage,
    QPainter,
    QPalette,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QBoxLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QFileDialog,
    QProgressBar,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# ================= 海康摄像机配置 =================
# 与 ptz_controller.py 保持一致；部署时按实际设备修改。
IPC_IP = "192.168.1.2"
NVR_IP = "192.168.1.157"
USERNAME = "admin"
PASSWORD = "abcd1234"
CHANNEL = 1
RTSP_PORT = 554
RTSP_CHANNEL = 101

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PERSON_MODEL_PATH = os.path.join(SCRIPT_DIR, "yolo11n.pt")
FIRE_OFFLINE_MODEL_PATH = os.path.join(SCRIPT_DIR, "fire_weight.pt")
FIRE_TRACK_MODEL_PATH = FIRE_OFFLINE_MODEL_PATH
MODEL_PATH = PERSON_MODEL_PATH
CSV_FILENAME = "gimbal_relative_position_data_network.csv"
IMG_SAVE_DIR = "captured_images_network"

# 同一份脚本部署到两台电脑时自动使用主机名作为设备标识；如需固定名称，
# 可通过 PTZ_DEVICE_ID 环境变量覆盖，不需要维护“主机版/友机版”两套代码。
_raw_device_id = os.environ.get("PTZ_DEVICE_ID", socket.gethostname())
DEVICE_ID = "".join(
    character if character.isalnum() or character in "-_" else "_"
    for character in _raw_device_id
).strip("_") or "DEVICE"
UDP_DEFAULT_RECEIVE_PORT = 56227
UDP_RETRY_INTERVAL_SECONDS = 3.0
UDP_MAX_RETRIES = 3
UDP_EVENT_MISSING_TIMEOUT_SECONDS = 45.0
UDP_DEDUP_TTL_SECONDS = 600.0
UDP_DEDUP_MAX_RECORDS = 512
LOCAL_DETECTION_SAVE_INTERVAL_SECONDS = 10.0
PEER_LIVE_IMAGE_INTERVAL_SECONDS = 5.0

os.makedirs(IMG_SAVE_DIR, exist_ok=True)


def set_solid_background(widget, color="#071A33"):
    """强制控件使用指定底色，避免 Windows 原生主题露出白色画布。"""
    palette = widget.palette()
    palette.setColor(QPalette.Window, QColor(color))
    widget.setPalette(palette)
    widget.setAutoFillBackground(True)


class SolidBackgroundWidget(QWidget):
    """不依赖系统主题，始终自行绘制纯色背景的页面容器。"""

    def __init__(self, color="#071A33", parent=None):
        super().__init__(parent)
        self._solid_background_color = QColor(color)
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self._solid_background_color)
        painter.end()
        super().paintEvent(event)


class AspectRatioVideoLabel(QLabel):
    """在可用区域中居中绘制固定 16:9 的视频画布。

    控件外层可以随布局自由伸缩，但黑色画面、占位文字和图像
    始终被限制在 16:9 矩形内，不会因窗口比例而拉伸。
    """

    ASPECT_WIDTH = 16
    ASPECT_HEIGHT = 9

    def video_rect(self):
        available = self.rect().adjusted(1, 1, -1, -1)
        if available.width() <= 0 or available.height() <= 0:
            return QRect()
        target_width = available.width()
        target_height = round(
            target_width * self.ASPECT_HEIGHT / self.ASPECT_WIDTH
        )
        if target_height > available.height():
            target_height = available.height()
            target_width = round(
                target_height * self.ASPECT_WIDTH / self.ASPECT_HEIGHT
            )
        left = available.x() + (available.width() - target_width) // 2
        top = available.y() + (available.height() - target_height) // 2
        return QRect(left, top, target_width, target_height)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#17375E"))
        canvas = self.video_rect()
        if canvas.isEmpty():
            painter.end()
            return

        painter.fillRect(canvas, QColor("#020A13"))
        painter.setPen(QColor("#2B5785"))
        painter.drawRect(canvas.adjusted(0, 0, -1, -1))

        pixmap = self.pixmap()
        if pixmap is not None and not pixmap.isNull():
            scaled = pixmap.scaled(
                canvas.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            image_x = canvas.x() + (canvas.width() - scaled.width()) // 2
            image_y = canvas.y() + (canvas.height() - scaled.height()) // 2
            painter.drawPixmap(image_x, image_y, scaled)
        elif self.text():
            painter.setPen(QColor("#6F8BAA"))
            painter.drawText(
                canvas.adjusted(12, 8, -12, -8),
                Qt.AlignCenter | Qt.TextWordWrap,
                self.text(),
            )
        painter.end()


class GeoCalculator:
    R_EARTH = 6_371_000.0

    @staticmethod
    def latlon_to_xy_offset(lat_ref, lon_ref, lat_target, lon_target):
        rad_lat = math.radians(lat_ref)
        dx = (
            math.radians(lon_target - lon_ref)
            * GeoCalculator.R_EARTH
            * math.cos(rad_lat)
        )
        dy = math.radians(lat_target - lat_ref) * GeoCalculator.R_EARTH
        return dx, dy

    @staticmethod
    def xy_offset_to_latlon(lat_ref, lon_ref, x, y):
        rad_lat = math.radians(lat_ref)
        dlat = y / GeoCalculator.R_EARTH
        dlon = x / (GeoCalculator.R_EARTH * math.cos(rad_lat))
        return lat_ref + math.degrees(dlat), lon_ref + math.degrees(dlon)


class RelativeBearingCalculator:
    """正东为 0 度、顺时针增加时的双站方位交会计算。"""

    @staticmethod
    def _direction(azimuth_degrees):
        radians = math.radians(float(azimuth_degrees) % 360.0)
        # 局部坐标：X 向东为正，Y 向南为正。
        return math.cos(radians), math.sin(radians)

    @staticmethod
    def solve(peer_bearing_degrees, baseline_metres, local_azimuth, peer_azimuth):
        baseline = float(baseline_metres)
        if not math.isfinite(baseline) or baseline <= 0.0:
            raise ValueError("两设备间距离必须大于 0")

        peer_dx, peer_dy = RelativeBearingCalculator._direction(
            peer_bearing_degrees
        )
        peer_x = baseline * peer_dx
        peer_y = baseline * peer_dy
        local_dx, local_dy = RelativeBearingCalculator._direction(local_azimuth)
        remote_dx, remote_dy = RelativeBearingCalculator._direction(peer_azimuth)

        denominator = local_dx * remote_dy - local_dy * remote_dx
        if abs(denominator) <= 0.01:
            raise ValueError("两条观测方向接近平行，无法稳定交会")

        local_distance = (
            peer_x * remote_dy - peer_y * remote_dx
        ) / denominator
        peer_distance = (
            peer_x * local_dy - peer_y * local_dx
        ) / denominator
        if local_distance <= 0.0 or peer_distance <= 0.0:
            raise ValueError("两条观测射线未在设备前方相交")

        target_x = local_distance * local_dx
        target_y = local_distance * local_dy
        target_bearing = (
            math.degrees(math.atan2(target_y, target_x)) + 360.0
        ) % 360.0
        return {
            "bearing_deg": target_bearing,
            "east_m": target_x,
            "south_m": target_y,
            "distance_m": local_distance,
            "peer_distance_m": peer_distance,
        }


class SettingsWindow(QWidget):
    params_updated = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("双站相对位置设置")
        self.resize(390, 210)
        self.setStyleSheet(
            "QWidget {background:#071A33; color:#E6F0FF; font-size:12px;}"
            "QGroupBox {background:#17375E; border:1px solid #2C527E; "
            "border-radius:6px; margin-top:10px; padding-top:6px;}"
            "QGroupBox::title {subcontrol-origin:margin; left:10px; padding:0 5px;}"
            "QLineEdit {background:#061A34; color:#EEF6FF; "
            "border:1px solid #2A5688; border-radius:4px; padding:5px;}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        local_group = QGroupBox("对端相对本机的位置")
        local_layout = QGridLayout(local_group)
        local_layout.addWidget(QLabel("对端方位角(°):"), 0, 0)
        self.input_peer_bearing = QLineEdit("0.0")
        self.input_peer_bearing.setToolTip("正东为0°，向右（顺时针）增加")
        local_layout.addWidget(self.input_peer_bearing, 0, 1)
        local_layout.addWidget(QLabel("设备间距离(m):"), 1, 0)
        self.input_peer_distance = QLineEdit("100.0")
        local_layout.addWidget(self.input_peer_distance, 1, 1)
        convention_label = QLabel(
            "角度约定：正东=0°，正南=90°，正西=180°，正北=270°"
        )
        convention_label.setWordWrap(True)
        convention_label.setStyleSheet("color:#78C2FF; padding:3px;")
        local_layout.addWidget(convention_label, 2, 0, 1, 2)
        layout.addWidget(local_group)

        apply_button = QPushButton("应用参数")
        apply_button.setStyleSheet(
            "background:#4CAF50; color:white; font-weight:bold; padding:6px;"
        )
        apply_button.clicked.connect(self.emit_params)
        layout.addWidget(apply_button)

    def emit_params(self):
        try:
            peer_bearing = float(self.input_peer_bearing.text()) % 360.0
            peer_distance = float(self.input_peer_distance.text())
            if not math.isfinite(peer_bearing) or not math.isfinite(peer_distance):
                raise ValueError
            if peer_distance <= 0.0:
                raise ValueError
            self.input_peer_bearing.setText(f"{peer_bearing:.3f}")
            self.input_peer_distance.setText(f"{peer_distance:.3f}")
            self.params_updated.emit(peer_bearing, peer_distance)
            self.setWindowTitle("参数设置（已生效）")
        except ValueError:
            self.setWindowTitle("参数设置（输入错误）")


class UdpReceiverThread(QThread):
    save_trigger = Signal(str, float, float, str)
    ack_received = Signal(str)
    remote_event_received = Signal(str)
    peer_event_started = Signal(str, float, float, str, str, int)
    peer_event_ended = Signal(str)
    peer_image_received = Signal(str, str, float, float, str, str, bool)

    def __init__(self):
        super().__init__()
        self.udp_ip = "0.0.0.0"
        self.udp_port = UDP_DEFAULT_RECEIVE_PORT
        self.sock = None
        self.is_running = True
        self.processed_messages = OrderedDict()

    def set_port(self, port):
        self.udp_port = int(port)

    def _cleanup_processed_messages(self, now):
        while self.processed_messages:
            _, saved_at = next(iter(self.processed_messages.items()))
            if (
                now - saved_at <= UDP_DEDUP_TTL_SECONDS
                and len(self.processed_messages) <= UDP_DEDUP_MAX_RECORDS
            ):
                break
            self.processed_messages.popitem(last=False)

    def _is_duplicate_message(self, message_key):
        now = time.monotonic()
        self._cleanup_processed_messages(now)
        if message_key in self.processed_messages:
            self.processed_messages.move_to_end(message_key)
            self.processed_messages[message_key] = now
            return True
        self.processed_messages[message_key] = now
        self._cleanup_processed_messages(now)
        return False

    def _send_protocol_ack(self, sender_address, event_id, message_id):
        if self.sock is None:
            return
        ack_payload = {
            "type": "ack",
            "status": "received",
            "msg": "ack",
            "device_id": DEVICE_ID,
            "event_id": event_id,
            "message_id": message_id,
            "timestamp": datetime.now().isoformat(sep=" ", timespec="milliseconds"),
        }
        try:
            self.sock.sendto(
                json.dumps(ack_payload, ensure_ascii=False).encode("utf-8"),
                sender_address,
            )
        except OSError:
            pass

    def run(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
        except OSError:
            pass
        self.sock.settimeout(1.0)
        try:
            self.sock.bind((self.udp_ip, self.udp_port))
        except OSError:
            return

        while self.is_running:
            try:
                data, sender_address = self.sock.recvfrom(65535)
                message = json.loads(data.decode("utf-8"))
                message_type = str(message.get("type", ""))
                is_ack = message_type == "ack" or (
                    message.get("status") == "received"
                    and message.get("msg") == "ack"
                )
                if is_ack:
                    self.ack_received.emit(str(message.get("event_id", "")))
                    continue

                event_id = str(message.get("event_id", ""))
                message_id = str(message.get("message_id", ""))

                if message_type == "event_end":
                    message_key = message_id or f"event_end:{event_id}"
                    duplicate = self._is_duplicate_message(message_key)
                    try:
                        reply_port = int(
                            message.get("reply_port", sender_address[1])
                        )
                    except (TypeError, ValueError):
                        reply_port = sender_address[1]
                    self._send_protocol_ack(
                        (sender_address[0], reply_port),
                        event_id,
                        message_id or message_key,
                    )
                    if not duplicate and event_id:
                        self.peer_event_ended.emit(event_id)
                    continue

                if "az" in message and "el" in message:
                    if message_id:
                        message_key = message_id
                    elif event_id:
                        message_key = f"{message_type}:{event_id}"
                    else:
                        # 兼容旧消息：只对字节完全相同的重复数据包去重，
                        # 不进行任何图像相似度或目标身份匹配。
                        message_key = hashlib.sha256(data).hexdigest()

                    duplicate = self._is_duplicate_message(message_key)
                    try:
                        reply_port = int(
                            message.get("reply_port", sender_address[1])
                        )
                    except (TypeError, ValueError):
                        reply_port = sender_address[1]
                    ack_address = (sender_address[0], reply_port)
                    self._send_protocol_ack(
                        ack_address, event_id, message_id or message_key
                    )
                    if duplicate:
                        continue

                    target_name = str(message.get("target", "")).strip().lower()

                    if message_type in (
                        "image_refresh_prepare",
                        "image_refresh_request",
                    ):
                        continue

                    # locked_target 会使接收方在本事件内成为响应方并持续搜寻。
                    if message_type == "locked_target" and not message.get("image"):
                        self.remote_event_received.emit(event_id or message_key)
                        self.peer_event_started.emit(
                            event_id or message_key,
                            float(message.get("az", 0.0)),
                            float(message.get("el", 0.0)),
                            target_name,
                            sender_address[0],
                            reply_port,
                        )
                        continue

                    if message_type == "peer_live_image":
                        has_remote_pose = (
                            "az" in message
                            and message.get("angle_reference")
                            == "east_zero_clockwise"
                        )
                        self.remote_event_received.emit(event_id or message_key)
                        self.peer_image_received.emit(
                            event_id or message_key,
                            str(
                                message.get(
                                    "timestamp",
                                    datetime.now().isoformat(sep=" "),
                                )
                            ),
                            float(message.get("az", 0.0)),
                            float(message.get("el", 0.0)),
                            str(message.get("image", "")),
                            target_name,
                            has_remote_pose,
                        )
                        continue

                    if message_type == "geo_result":
                        # 兼容旧版本消息；当前版本由双方直接根据图像包
                        # 携带的设备坐标和方位角独立完成计算。
                        continue

                    self.remote_event_received.emit(event_id or message_key)
                    self.save_trigger.emit(
                        str(message.get("timestamp", datetime.now().isoformat(sep=" "))),
                        float(message.get("az", 0.0)),
                        float(message.get("el", 0.0)),
                        message.get("image", ""),
                    )
            except socket.timeout:
                continue
            except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
                if self.is_running:
                    continue

        if self.sock:
            self.sock.close()

    def stop(self):
        self.is_running = False
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
        self.wait(1500)


class NetworkPtzBackend:
    """后台执行 ISAPI 控制和状态查询，避免阻塞视频检测线程。"""

    def __init__(self, state_callback, status_callback):
        self.ipc_base_url = f"http://{IPC_IP}"
        self.nvr_base_url = f"http://{NVR_IP}"
        self.auth = HTTPDigestAuth(USERNAME, PASSWORD)
        self.channel = CHANNEL
        self.state_callback = state_callback
        self.status_callback = status_callback

        self._running = False
        self._command_event = threading.Event()
        self._command_lock = threading.Lock()
        self._desired_command = (0, 0)
        self._last_queued_command = None
        self._last_queue_time = 0.0
        self._threads = []

    def start(self):
        self._running = True
        self._threads = [
            threading.Thread(target=self._control_loop, daemon=True),
            threading.Thread(target=self._status_loop, daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def move(self, pan, tilt, force=False):
        pan = int(max(-100, min(100, pan)))
        tilt = int(max(-100, min(100, tilt)))
        command = (pan, tilt)
        now = time.monotonic()
        # 连续控制会保持运动。仅在速度变化或到达保活周期时发送，避免每帧发 HTTP。
        if not force and command == self._last_queued_command and now - self._last_queue_time < 0.5:
            return
        with self._command_lock:
            self._desired_command = command
            self._last_queued_command = command
            self._last_queue_time = now
        self._command_event.set()

    def _control_loop(self):
        while self._running:
            if not self._command_event.wait(0.2):
                continue
            self._command_event.clear()
            with self._command_lock:
                pan, tilt = self._desired_command
            result = set_continuous(
                self.nvr_base_url, self.channel, self.auth, pan, tilt
            )
            if result.get("status") != "success" and self._running:
                self.status_callback(
                    f"状态: PTZ 控制失败 - {result.get('error', '未知错误')}"
                )

    def _status_loop(self):
        last_error = None
        while self._running:
            result = get_absoluteEx(self.ipc_base_url, self.channel, self.auth)
            if result.get("status") == "success":
                state = result.get("state", {})
                self.state_callback(state)
                last_error = None
            else:
                error = str(result.get("error", "未知错误"))
                if error != last_error:
                    self.status_callback(f"状态: 云台状态读取失败 - {error}")
                    last_error = error
            # 与 ptz_controller.py 一致，每 2 秒刷新一次角度状态。
            for _ in range(20):
                if not self._running:
                    return
                time.sleep(0.1)

    def stop(self):
        if not self._running:
            return
        # 与 ptz_controller.py 的退出逻辑一致：关闭前同步发送停止命令。
        # 即使后台线程正忙，这个请求也不会被后续的退出标志吞掉。
        set_continuous(self.nvr_base_url, self.channel, self.auth, 0, 0)
        self._running = False
        self._command_event.set()
        for thread in self._threads:
            thread.join(timeout=1.0)


class CameraThread(QThread):
    frame_ready = Signal(QImage)
    remote_frame_ready = Signal(QImage)
    status_update = Signal(str)
    angle_update = Signal(float, float)
    ptz_state_update = Signal(dict)
    geo_result_signal = Signal(str, str, str)

    def __init__(self):
        super().__init__()
        self.is_running = True
        self.video_active = False
        self.tracking_active = False
        self.model = None
        self.target_label = "person"
        self.class_map = {"person": 0, "fire": 1}

        self.dead_zone = 30
        self.kp = 0.12
        self.max_speed = 30
        self.smooth_factor = 0.6
        self.prev_cx = 0.0
        self.prev_cy = 0.0

        self.locked_id = None
        self.is_acked = False
        self.last_send_time = 0.0
        self.memory_az = None
        self.current_az = 0.0
        self.current_el = 0.0

        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.temp_local_path = None

        self.local_lat = 30.0
        self.local_lon = 120.0
        self.remote_lat = 30.001
        self.remote_lon = 120.001

        self.target_ip = "127.0.0.1"
        self.target_port = UDP_DEFAULT_RECEIVE_PORT
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ptz = NetworkPtzBackend(self._on_ptz_state, self._on_ptz_status)
        self._init_csv_file()

    @staticmethod
    def _rtsp_url():
        username = quote(USERNAME, safe="")
        password = quote(PASSWORD, safe="")
        return (
            f"rtsp://{username}:{password}@{IPC_IP}:{RTSP_PORT}"
            f"/Streaming/Channels/{RTSP_CHANNEL}"
        )

    def _on_ptz_state(self, state):
        azimuth = float(state.get("azimuth", 0.0))
        elevation = float(state.get("elevation", 0.0))
        self.current_az = azimuth
        self.current_el = elevation
        self.angle_update.emit(azimuth, elevation)
        self.ptz_state_update.emit(dict(state))

    def _on_ptz_status(self, message):
        self.status_update.emit(message)

    def _init_csv_file(self):
        if os.path.exists(CSV_FILENAME):
            return
        with open(CSV_FILENAME, "w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(
                [
                    "Timestamp",
                    "Loc_Az",
                    "Rem_Az",
                    "Loc_Img",
                    "Rem_Img",
                    "Target_Bearing_East0_Clockwise",
                    "Target_East_m",
                    "Target_South_m",
                    "Local_Distance_m",
                ]
            )

    def set_network_config(self, ip, port):
        self.target_ip = ip
        self.target_port = int(port)

    @Slot(float, float)
    def set_gps_coords(self, local_lat, local_lon):
        self.local_lat = local_lat
        self.local_lon = local_lon

    def send_locked_data(self, target_id, azimuth, elevation):
        payload = {
            "type": "locked_target",
            "id": target_id,
            "az": round(azimuth, 2),
            "el": round(elevation, 2),
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
        }
        try:
            self.send_sock.sendto(
                json.dumps(payload).encode("utf-8"),
                (self.target_ip, self.target_port),
            )
        except OSError as error:
            self.status_update.emit(f"状态: UDP 发送失败 - {error}")

    @Slot(str)
    def on_ack_received(self, _ack_id):
        if self.locked_id is not None and not self.is_acked:
            self.is_acked = True
            self.status_update.emit("状态: 对方已确认 (ACK)")

    @Slot(str, float, float, str)
    def on_save_command(self, timestamp, az_remote, _el_remote, remote_img_b64):
        target_lat, target_lon, distance = "0.0", "0.0", "0.0"
        if self.memory_az is not None:
            try:
                rx, ry = GeoCalculator.latlon_to_xy_offset(
                    self.local_lat,
                    self.local_lon,
                    self.remote_lat,
                    self.remote_lon,
                )
                rad_a = math.radians(90 - self.memory_az)
                rad_b = math.radians(90 - az_remote)
                tan_a, tan_b = math.tan(rad_a), math.tan(rad_b)
                if abs(tan_a - tan_b) > 0.01:
                    tx = (ry - rx * tan_b) / (tan_a - tan_b)
                    ty = tx * tan_a
                    distance_value = math.hypot(tx, ty)
                    lat, lon = GeoCalculator.xy_offset_to_latlon(
                        self.local_lat, self.local_lon, tx, ty
                    )
                    target_lat = f"{lat:.6f}"
                    target_lon = f"{lon:.6f}"
                    distance = f"{distance_value:.1f}"
                    self.geo_result_signal.emit(target_lat, target_lon, distance)
                else:
                    self.geo_result_signal.emit("平行", "平行", "0.0")
            except (ArithmeticError, ValueError):
                self.geo_result_signal.emit("解算失败", "解算失败", "0.0")
        else:
            self.geo_result_signal.emit("等待锁定", "等待锁定", "0.0")

        safe_time = str(timestamp).replace(":", "-").replace(" ", "_").replace(".", "-")
        remote_filename = os.path.join(
            IMG_SAVE_DIR, f"img_{safe_time}_REMOTE.jpg"
        )
        saved_remote = "None"
        saved_local = "Failed"

        if self.temp_local_path and os.path.exists(self.temp_local_path):
            saved_local = self.temp_local_path
        else:
            with self.frame_lock:
                if self.current_frame is not None:
                    fallback = os.path.join(
                        IMG_SAVE_DIR, f"img_{safe_time}_LOCAL_fb.jpg"
                    )
                    if cv2.imwrite(fallback, self.current_frame):
                        saved_local = fallback

        if remote_img_b64:
            try:
                raw_image = base64.b64decode(remote_img_b64)
                remote_image = cv2.imdecode(
                    np.frombuffer(raw_image, np.uint8), cv2.IMREAD_COLOR
                )
                if remote_image is not None:
                    cv2.imwrite(remote_filename, remote_image)
                    saved_remote = remote_filename
                    self.remote_frame_ready.emit(self._to_qimage(remote_image))
            except (ValueError, TypeError):
                pass

        with open(CSV_FILENAME, "a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(
                [
                    timestamp,
                    self.memory_az if self.memory_az is not None else 0.0,
                    az_remote,
                    saved_local,
                    saved_remote,
                    target_lat,
                    target_lon,
                    distance,
                ]
            )
        self.status_update.emit("状态: 数据归档完成")

    def set_tracking_state(self, active):
        self.tracking_active = bool(active)
        self.locked_id = None
        self.is_acked = False
        self.temp_local_path = None
        self.memory_az = None
        self.prev_cx = 0.0
        self.prev_cy = 0.0
        if active:
            self.ptz.move(0, 0, force=True)
            self.status_update.emit("状态: 跟踪开启")
        else:
            self.ptz.move(0, 0, force=True)
            self.status_update.emit("状态: 待机")

    def set_video_state(self, active):
        """控制 RTSP 视频流，工作线程本身保持运行以便随时重新启动。"""
        self.video_active = bool(active)
        if active:
            self.status_update.emit("状态: 正在启动视频流...")
        else:
            # 关闭画面时禁止继续自动跟踪，并立即让云台停止。
            self.tracking_active = False
            self.ptz.move(0, 0, force=True)
            self.status_update.emit("状态: 视频流已停止")

    def manual_move(self, pan, tilt):
        """发送手动连续控制命令；自动跟踪期间不接受手动命令。"""
        if self.tracking_active:
            return
        self.ptz.move(pan, tilt, force=True)

    def set_target(self, label):
        self.target_label = label
        self.locked_id = None
        self.prev_cx = 0.0
        self.prev_cy = 0.0

    @staticmethod
    def _to_qimage(frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        return QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format_RGB888,
        ).copy()

    def _open_stream(self):
        url = self._rtsp_url()
        capture = cv2.VideoCapture()
        # 新版 OpenCV/FFmpeg 支持打开和读取超时；旧版不支持时 set 会安全返回 False。
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
        capture.open(url, cv2.CAP_FFMPEG)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if capture.isOpened():
            self.status_update.emit("状态: RTSP 视频流连接成功")
            return capture
        capture.release()
        return None

    def _select_target(self, results):
        if results.boxes is None or results.boxes.id is None:
            return None
        ids = results.boxes.id.int().cpu().tolist()
        boxes = results.boxes.xyxy.cpu().tolist()
        targets = []
        for target_id, coords in zip(ids, boxes):
            x1, y1, x2, y2 = map(int, coords)
            targets.append(
                {
                    "id": target_id,
                    "box": (x1, y1, x2, y2),
                    "area": max(0, x2 - x1) * max(0, y2 - y1),
                }
            )

        selected = None
        if self.locked_id is not None:
            selected = next(
                (target for target in targets if target["id"] == self.locked_id),
                None,
            )
            if selected is None:
                self.locked_id = None

        if self.locked_id is None and targets:
            selected = max(targets, key=lambda target: target["area"])
            self.locked_id = selected["id"]
            self.is_acked = False
            self.last_send_time = 0.0
            self.temp_local_path = None
            self.memory_az = None
            self.prev_cx = self.prev_cy = 0.0
        return selected

    def _track_frame(self, frame):
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        cv2.line(frame, (center_x, 0), (center_x, height), (180, 180, 180), 1)
        cv2.line(frame, (0, center_y), (width, center_y), (180, 180, 180), 1)
        cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

        if not self.tracking_active or self.model is None:
            return frame

        try:
            result = self.model.track(
                frame,
                persist=True,
                classes=[self.class_map.get(self.target_label, 0)],
                verbose=False,
                conf=0.25,
            )[0]
        except Exception as error:
            self.status_update.emit(f"状态: 目标检测失败 - {error}")
            self.ptz.move(0, 0)
            return frame

        target = self._select_target(result)
        if target is None:
            self.ptz.move(0, 0)
            self.prev_cx = self.prev_cy = 0.0
            return frame

        x1, y1, x2, y2 = target["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID:{target['id']}",
            (x1, max(20, y1 - 35)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        raw_x = (x1 + x2) / 2.0
        raw_y = (y1 + y2) / 2.0
        if self.prev_cx == 0.0 and self.prev_cy == 0.0:
            self.prev_cx, self.prev_cy = raw_x, raw_y
        alpha = self.smooth_factor
        smooth_x = alpha * raw_x + (1.0 - alpha) * self.prev_cx
        smooth_y = alpha * raw_y + (1.0 - alpha) * self.prev_cy
        self.prev_cx, self.prev_cy = smooth_x, smooth_y
        dx, dy = int(smooth_x - center_x), int(smooth_y - center_y)

        # 海康 continuous 接口：pan 正值向右，tilt 正值向上。
        pan = int(dx * self.kp) if abs(dx) >= self.dead_zone else 0
        tilt = int(-dy * self.kp) if abs(dy) >= self.dead_zone else 0
        pan = max(-self.max_speed, min(self.max_speed, pan))
        tilt = max(-self.max_speed, min(self.max_speed, tilt))
        if abs(pan) < 3:
            pan = 0
        if abs(tilt) < 3:
            tilt = 0
        self.ptz.move(pan, tilt)

        if abs(dx) < self.dead_zone and abs(dy) < self.dead_zone:
            cv2.putText(
                frame,
                "CENTERED",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            self.memory_az = self.current_az
            if self.temp_local_path is None:
                timestamp = datetime.now().strftime("%H-%M-%S-%f")[:-3]
                filename = os.path.join(
                    IMG_SAVE_DIR, f"img_{timestamp}_LOCAL.jpg"
                )
                if cv2.imwrite(filename, frame):
                    self.temp_local_path = filename

            now = time.time()
            if not self.is_acked and now - self.last_send_time > 3.0:
                self.send_locked_data(
                    self.locked_id, self.current_az, self.current_el
                )
                self.last_send_time = now
                cv2.putText(
                    frame,
                    "SENDING...",
                    (x1, max(20, y1 - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
            elif self.is_acked:
                cv2.putText(
                    frame,
                    "ACK OK",
                    (x1, max(20, y1 - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )
        return frame

    def run(self):
        self.ptz.start()
        if YOLO is None:
            self.status_update.emit("状态: 缺少 ultralytics，无法启动跟踪")
        else:
            try:
                self.model = YOLO(MODEL_PATH)
                if hasattr(self, "loaded_model_label"):
                    self.loaded_model_label = "person"
            except Exception as error:
                self.status_update.emit(f"状态: 模型加载失败 - {error}")

        capture = None
        while self.is_running:
            if not self.video_active:
                if capture is not None:
                    capture.release()
                    capture = None
                time.sleep(0.05)
                continue

            if capture is None:
                self.status_update.emit("状态: 正在连接 RTSP 视频流...")
                capture = self._open_stream()
                if capture is None:
                    self.status_update.emit("状态: RTSP 连接失败，2 秒后重试")
                    for _ in range(20):
                        if not self.is_running or not self.video_active:
                            break
                        time.sleep(0.1)
                    continue

            ok, frame = capture.read()
            if not ok:
                self.status_update.emit("状态: 视频帧读取失败，正在重连")
                capture.release()
                capture = None
                continue

            # capture.read() 返回期间用户可能已经点击了“停止视频”。
            if not self.video_active:
                continue

            frame = self._track_frame(frame)
            with self.frame_lock:
                self.current_frame = frame.copy()
            self.frame_ready.emit(self._to_qimage(frame))
            time.sleep(0.005)

        if capture is not None:
            capture.release()
        self.ptz.stop()
        self.send_sock.close()

    def stop(self):
        self.is_running = False
        self.video_active = False
        self.ptz.move(0, 0, force=True)
        # RTSP 打开最多 5 秒，ISAPI 停止请求最多 10 秒；必须等 run() 真正退出，
        # 否则窗口销毁时会出现 “QThread: Destroyed while thread is still running”。
        return self.wait(20_000)


class HostOfflineAnalysisThread(QThread):
    """使用 fire_weight.pt 对本地图片或视频执行离线火情分析。"""

    frame_ready = Signal(QImage)
    status_update = Signal(str)
    progress_update = Signal(int)
    analysis_finished = Signal(bool, str)

    def __init__(self, source_path, is_video, confidence=0.5):
        super().__init__()
        self.source_path = source_path
        self.is_video = is_video
        self.confidence = confidence
        self._run_flag = True

    @staticmethod
    def _to_qimage(frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        return QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format_RGB888,
        ).copy()

    @staticmethod
    def _draw_result(frame, result):
        annotated = frame.copy()
        fire_count = 0
        smoke_count = 0
        if result.boxes is None:
            return annotated, fire_count, smoke_count

        boxes = result.boxes.xyxy.cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        for box, class_id, score in zip(boxes, classes, confidences):
            if class_id not in (0, 1):
                continue
            x1, y1, x2, y2 = map(int, box)
            if class_id == 0:
                name = "fire"
                color = (0, 0, 255)
                fire_count += 1
            else:
                name = "smoke"
                color = (0, 165, 255)
                smoke_count += 1
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                annotated,
                f"{name} {score:.2f}",
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
                cv2.LINE_AA,
            )
        return annotated, fire_count, smoke_count

    def _analyze_frame(self, model, frame):
        result = model.predict(
            frame,
            classes=[0, 1],
            conf=self.confidence,
            verbose=False,
        )[0]
        return self._draw_result(frame, result)

    def _run_image(self, model):
        image_data = np.fromfile(self.source_path, dtype=np.uint8)
        frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("无法读取所选图片")
        annotated, fire_count, smoke_count = self._analyze_frame(model, frame)
        if not self._run_flag:
            return
        self.frame_ready.emit(self._to_qimage(annotated))
        self.progress_update.emit(100)
        self.status_update.emit(
            f"图片分析完成｜fire: {fire_count}｜smoke: {smoke_count}"
        )

    def _run_video(self, model):
        capture = cv2.VideoCapture(self.source_path)
        if not capture.isOpened():
            raise RuntimeError("无法打开所选视频")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_interval = 1.0 / fps if fps and fps > 0 else 0.0
        frame_index = 0
        try:
            while self._run_flag and not self.isInterruptionRequested():
                started_at = time.monotonic()
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index += 1
                annotated, fire_count, smoke_count = self._analyze_frame(
                    model, frame
                )
                if not self._run_flag:
                    break
                self.frame_ready.emit(self._to_qimage(annotated))
                progress = (
                    int(frame_index * 100 / total_frames)
                    if total_frames > 0 else 0
                )
                self.progress_update.emit(min(100, progress))
                self.status_update.emit(
                    f"正在分析第 {frame_index} 帧｜"
                    f"fire: {fire_count}｜smoke: {smoke_count}"
                )
                remaining = frame_interval - (time.monotonic() - started_at)
                if remaining > 0:
                    time.sleep(min(remaining, 0.05))
        finally:
            capture.release()

        if self._run_flag:
            self.progress_update.emit(100)
            self.status_update.emit(f"视频分析完成｜共处理 {frame_index} 帧")

    def run(self):
        if YOLO is None:
            self.analysis_finished.emit(False, "未安装Ultralytics，无法分析")
            return
        if not os.path.exists(FIRE_OFFLINE_MODEL_PATH):
            self.analysis_finished.emit(
                False, f"未找到模型文件: {FIRE_OFFLINE_MODEL_PATH}"
            )
            return
        try:
            self.status_update.emit("正在加载 fire_weight.pt...")
            model = YOLO(FIRE_OFFLINE_MODEL_PATH)
            if self.is_video:
                self._run_video(model)
            else:
                self._run_image(model)
            message = "分析已停止" if not self._run_flag else "分析完成"
            self.analysis_finished.emit(True, message)
        except Exception as exc:
            self.analysis_finished.emit(False, f"离线分析失败: {exc}")

    def stop(self):
        self._run_flag = False
        self.requestInterruption()
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能检测系统 - 海康网络云台版")
        self.resize(1380, 790)
        self.setMinimumSize(1160, 680)

        self.manual_pan_speed = 20.0
        self.manual_tilt_speed = 20.0
        self.manual_direction_state = {
            "up": False,
            "down": False,
            "left": False,
            "right": False,
        }

        self.settings_window = SettingsWindow()
        self.thread = CameraThread()
        self.thread.frame_ready.connect(self.update_image)
        self.thread.remote_frame_ready.connect(self.update_remote_image)
        self.thread.status_update.connect(self.update_status_label)
        self.thread.angle_update.connect(self.update_angle_display)
        self.thread.ptz_state_update.connect(self.update_ptz_state_display)
        self.thread.geo_result_signal.connect(self.update_geo_result)
        self.settings_window.params_updated.connect(self.thread.set_gps_coords)
        self.settings_window.emit_params()

        self.rx_thread = UdpReceiverThread()
        self.rx_thread.save_trigger.connect(self.thread.on_save_command)
        self.rx_thread.ack_received.connect(self.thread.on_ack_received)
        self.rx_thread.start()

        self.setup_ui()
        QApplication.instance().installEventFilter(self)
        self.thread.start()

    def setup_ui(self):
        central = SolidBackgroundWidget()
        central.setObjectName("mainPanel")
        set_solid_background(central)
        self.setCentralWidget(central)
        self.setStyleSheet(
            """
            QWidget#mainPanel {
                background: #F4F7FB;
                color: #263445;
            }
            QFrame#videoCard, QFrame#controlBar {
                background: #FFFFFF;
                border: 1px solid #D8E0EA;
                border-radius: 8px;
            }
            QLabel#videoTitle {
                color: #334155;
                font-size: 13px;
                font-weight: 600;
                border: none;
                background: transparent;
            }
            QLabel#videoDisplay {
                color: #64748B;
                background: #E9EFF6;
                border: 1px solid #D8E0EA;
                border-radius: 5px;
                font-size: 13px;
            }
            QGroupBox {
                color: #334155;
                background: #FFFFFF;
                border: 1px solid #D8E0EA;
                border-radius: 8px;
                margin-top: 12px;
                font-size: 13px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0 5px;
                color: #475569;
                background: #F4F7FB;
            }
            QLineEdit, QComboBox {
                color: #263445;
                background: #FFFFFF;
                border: 1px solid #C9D3DF;
                border-radius: 5px;
                padding: 5px 8px;
                min-height: 20px;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #3B82F6;
            }
            QPushButton {
                color: #334155;
                background: #EEF2F7;
                border: 1px solid #CBD5E1;
                border-radius: 5px;
                padding: 6px 14px;
                min-height: 20px;
                font-weight: 600;
            }
            QPushButton:hover { background: #E2E8F0; }
            QPushButton#trackButton {
                color: #FFFFFF;
                background: #2563EB;
                border-color: #2563EB;
            }
            QPushButton#trackButton:hover { background: #1D4ED8; }
            QPushButton#trackButton[tracking="true"] {
                background: #DC2626;
                border-color: #DC2626;
            }
            QPushButton#videoButton {
                color: #FFFFFF;
                background: #0F766E;
                border-color: #0F766E;
            }
            QPushButton#videoButton:hover { background: #0D9488; }
            QPushButton#videoButton[streaming="true"] {
                background: #EA580C;
                border-color: #EA580C;
            }
            QFrame#manualPanel {
                background: #F8FAFC;
                border: 1px solid #D8E0EA;
                border-radius: 8px;
            }
            QLabel#ptzStateValue {
                color: #0F4C81;
                font-weight: 600;
                background: #EEF6FF;
                border-radius: 4px;
                padding: 3px 6px;
            }
            QPushButton#directionButton {
                color: #1E3A5F;
                background: #FFFFFF;
                border: 1px solid #B8C7D9;
                font-size: 18px;
                padding: 2px;
            }
            QPushButton#directionButton:pressed {
                color: #FFFFFF;
                background: #2563EB;
                border-color: #2563EB;
            }
            QLabel#statusLabel {
                color: #334155;
                background: #EEF3F8;
                border: 1px solid #D8E0EA;
                border-radius: 5px;
                padding: 5px 10px;
            }
            """
        )
        self.setStyleSheet(
            """
            QWidget#mainPanel {
                background: #071A33;
                color: #E6F0FF;
            }
            QWidget {
                font-family: "Microsoft YaHei", "Segoe UI", Arial;
                font-size: 13px;
                color: #E6F0FF;
            }
            QFrame#topHeader {
                background: #091E39;
                border: 1px solid #173C66;
                border-radius: 5px;
            }
            QLabel#appTitle {
                color: #FFFFFF;
                font-size: 20px;
                font-weight: 700;
                background: transparent;
                border: none;
            }
            QLabel#subTitle {
                color: #7FA9D8;
                font-size: 12px;
                background: transparent;
                border: none;
            }
            QWidget#mainPanel QFrame#videoCard,
            QWidget#mainPanel QFrame#controlBar,
            QWidget#mainPanel QFrame#manualPanel {
                background: #17375E;
                border: 1px solid #2C527E;
                border-radius: 6px;
            }
            QWidget#mainPanel QLabel#videoTitle {
                color: #EAF4FF;
                font-size: 13px;
                font-weight: 600;
                border: none;
                background: transparent;
            }
            QWidget#mainPanel QLabel#videoDisplay {
                color: #6F8BAA;
                background: #020A13;
                border: 1px solid #2B5785;
                border-radius: 5px;
                font-size: 14px;
            }
            QWidget#mainPanel QGroupBox {
                color: #FFFFFF;
                background: #17375E;
                border: 1px solid #2C527E;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 6px;
                font-size: 13px;
                font-weight: 600;
            }
            QWidget#mainPanel QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 6px;
                color: #FFFFFF;
                background: #071A33;
            }
            QWidget#mainPanel QLineEdit,
            QWidget#mainPanel QComboBox,
            QWidget#mainPanel QTextEdit,
            QWidget#mainPanel QPlainTextEdit {
                color: #EEF6FF;
                background: #061A34;
                border: 1px solid #2A5688;
                border-radius: 4px;
                padding: 6px;
                selection-background-color: #2479D8;
            }
            QWidget#mainPanel QComboBox QAbstractItemView {
                color: #FFFFFF;
                background: #102E50;
                selection-background-color: #2479D8;
            }
            QWidget#mainPanel QPushButton {
                color: #FFFFFF;
                background: #285B9A;
                border: 1px solid #3B72B6;
                border-radius: 4px;
                padding: 6px 12px;
                min-height: 25px;
                font-weight: 600;
            }
            QWidget#mainPanel QPushButton:hover { background: #3371BA; }
            QWidget#mainPanel QPushButton:pressed { background: #1E4B82; }
            QWidget#mainPanel QPushButton:disabled {
                color: #68809A;
                background: #183553;
                border-color: #294663;
            }
            QWidget#mainPanel QPushButton#navButton {
                background: #102D50;
                border-color: #2A5688;
                min-width: 130px;
                min-height: 32px;
            }
            QWidget#mainPanel QPushButton#navButton:checked {
                background: #1976D2;
                border-color: #3B91EA;
            }
            QWidget#mainPanel QPushButton#trackButton {
                color: #FFFFFF;
                background: #1976D2;
                border-color: #3B91EA;
            }
            QWidget#mainPanel QPushButton#trackButton[tracking="true"] {
                background: #C9273B;
                border-color: #E04455;
            }
            QWidget#mainPanel QPushButton#videoButton {
                color: #FFFFFF;
                background: #168747;
                border-color: #2AA966;
            }
            QWidget#mainPanel QPushButton#videoButton[streaming="true"] {
                background: #C86A16;
                border-color: #E18A33;
            }
            QWidget#mainPanel QPushButton#directionButton {
                color: #FFFFFF;
                background: #285B9A;
                border: 1px solid #3B72B6;
                font-size: 18px;
                padding: 2px;
            }
            QWidget#mainPanel QPushButton#directionButton:pressed {
                background: #1976D2;
                border-color: #4BA3F5;
            }
            QWidget#mainPanel QLabel#ptzStateValue {
                color: #62B7FF;
                font-weight: 600;
                background: #0D2948;
                border-radius: 4px;
                padding: 3px 6px;
            }
            QWidget#mainPanel QLabel#statusLabel {
                color: #DCEBFA;
                background: #0D2948;
                border: 1px solid #2A5688;
                border-radius: 5px;
                padding: 5px 10px;
            }
            QWidget#mainPanel QFrame#offlinePreview {
                background: #020A13;
                border: 1px solid #2B5785;
                border-radius: 5px;
            }
            QWidget#mainPanel QProgressBar {
                color: #FFFFFF;
                background: #071A33;
                border: 1px solid #2A5688;
                border-radius: 4px;
                text-align: center;
                min-height: 20px;
            }
            QWidget#mainPanel QProgressBar::chunk { background: #1E88E5; }
            QWidget#mainPanel QSplitter::handle {
                background: #0A2544;
                width: 3px;
            }
            """
        )
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 8, 10, 10)
        main_layout.setSpacing(8)

        header = QFrame()
        header.setObjectName("topHeader")
        set_solid_background(header, "#091E39")
        header.setStyleSheet(
            "QFrame#topHeader {background:#091E39; border:1px solid #173C66; "
            "border-radius:5px;}"
            "QLabel {background:transparent; border:none;}"
        )
        header.setFixedHeight(56)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(14, 5, 14, 5)
        title_layout = QVBoxLayout()
        title_layout.setSpacing(0)
        app_title = QLabel("双云台智能侦察仪山火检测系统")
        app_title.setObjectName("appTitle")
        sub_title = QLabel("对等双机端 · 协同跟踪、事件去重与目标定位")
        sub_title.setObjectName("subTitle")
        title_layout.addWidget(app_title)
        title_layout.addWidget(sub_title)
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        system_state = QLabel("●  系统就绪")
        system_state.setStyleSheet(
            "color:#61D095; font-weight:700; background:transparent;"
        )
        header_layout.addWidget(system_state)
        main_layout.addWidget(header)

        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(6)
        self.btn_online_page = QPushButton("在线实时监测")
        self.btn_offline_page = QPushButton("本地离线分析")
        nav_button_style = (
            "QPushButton {color:#FFFFFF; background:#102D50; "
            "border:1px solid #2A5688; border-radius:4px; padding:7px 16px; "
            "min-width:130px; min-height:32px; font-weight:600;}"
            "QPushButton:hover {background:#285B9A;}"
            "QPushButton:checked {background:#1976D2; border-color:#3B91EA;}"
        )
        for button in (self.btn_online_page, self.btn_offline_page):
            button.setObjectName("navButton")
            button.setCheckable(True)
            button.setFocusPolicy(Qt.NoFocus)
            button.setStyleSheet(nav_button_style)
        self.btn_online_page.setChecked(True)
        self.btn_online_page.clicked.connect(lambda: self.switch_page(0))
        self.btn_offline_page.clicked.connect(lambda: self.switch_page(1))
        nav_layout.addWidget(self.btn_online_page)
        nav_layout.addWidget(self.btn_offline_page)
        nav_layout.addStretch()
        main_layout.addLayout(nav_layout)

        self.content_stack = QStackedWidget()
        set_solid_background(self.content_stack)
        self.content_stack.setStyleSheet(
            "QStackedWidget {background:#071A33; border:none;}"
        )
        online_page = SolidBackgroundWidget()
        online_page.setObjectName("onlinePage")
        set_solid_background(online_page)
        online_page.setStyleSheet("QWidget#onlinePage {background:#071A33;}")
        online_page.setMinimumHeight(760)
        online_page.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        online_layout = QVBoxLayout(online_page)
        online_layout.setContentsMargins(0, 0, 0, 0)
        online_layout.setSpacing(8)

        workspace = SolidBackgroundWidget()
        workspace.setObjectName("onlineWorkspace")
        set_solid_background(workspace)
        workspace.setStyleSheet("QWidget#onlineWorkspace {background:#071A33;}")
        workspace_layout = QHBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(10)
        left_panel = SolidBackgroundWidget()
        left_panel.setObjectName("onlineLeftPanel")
        self.monitor_panel = left_panel
        set_solid_background(left_panel)
        left_panel.setStyleSheet("QWidget#onlineLeftPanel {background:#071A33;}")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        video_area = SolidBackgroundWidget()
        video_area.setObjectName("onlineVideoArea")
        self.video_area = video_area
        set_solid_background(video_area)
        video_area.setStyleSheet("QWidget#onlineVideoArea {background:#071A33;}")
        video_layout = QHBoxLayout(video_area)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(10)
        # 保留足够的视频观察高度；窗口较小时由在线页滚动承载，避免压扁。
        video_area.setMinimumHeight(380)
        video_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        def create_video_card(title, placeholder, minimum_width):
            card = QFrame()
            card.setObjectName("videoCard")
            card.setStyleSheet(
                "QFrame#videoCard {background:#17375E; border:1px solid #2C527E; "
                "border-radius:6px;}"
                "QLabel#videoTitle {color:#EAF4FF; background:transparent; "
                "border:none; font-weight:600;}"
                "QLabel#videoDisplay {color:#6F8BAA; background:#020A13; "
                "border:1px solid #2B5785; border-radius:5px;}"
            )
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(8, 7, 8, 8)
            card_layout.setSpacing(6)
            title_label = QLabel(title)
            title_label.setObjectName("videoTitle")
            title_label.setFixedHeight(22)
            display = AspectRatioVideoLabel(placeholder)
            display.setObjectName("videoDisplay")
            display.setAlignment(Qt.AlignCenter)
            display.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            display.setMinimumSize(minimum_width, 240)
            card_layout.addWidget(title_label)
            card_layout.addWidget(display, 1)
            return card, title_label, display

        local_card, self.lbl_local_title, self.lbl_local = create_video_card(
            "本机画面", "正在连接 RTSP 视频流…", 420
        )
        remote_card, self.lbl_remote_title, self.lbl_remote = create_video_card(
            "对端画面", "等待对端图像回传…", 260
        )
        # 本机是主要跟踪画面，按约 2:1 分配显示宽度。
        video_layout.addWidget(local_card, 2)
        video_layout.addWidget(remote_card, 1)
        left_layout.addWidget(video_area, 1)

        angle_group = QGroupBox("实时网络云台数据")
        self.angle_group = angle_group
        angle_group.setStyleSheet(
            "QGroupBox {color:#FFFFFF; background:#17375E; "
            "border:1px solid #2C527E; border-radius:6px; margin-top:12px;}"
            "QGroupBox::title {subcontrol-origin:margin; left:10px; "
            "padding:0 6px; background:#071A33;}"
            "QLabel {background:transparent;}"
        )
        angle_group.setFixedHeight(84)
        angle_layout = QHBoxLayout(angle_group)
        angle_layout.setContentsMargins(20, 18, 20, 9)
        angle_layout.setSpacing(10)
        self.lbl_az_val = QLabel("0.00°")
        self.lbl_el_val = QLabel("0.00°")
        for value_label, color, background in (
            (self.lbl_az_val, "#1976D2", "#E3F2FD"),
            (self.lbl_el_val, "#388E3C", "#E8F5E9"),
        ):
            value_label.setFixedSize(88, 36)
            value_label.setAlignment(Qt.AlignCenter)
            value_label.setStyleSheet(
                f"font-size:15px; font-weight:bold; color:{color}; "
                f"background:{background}; padding:2px; border-radius:5px;"
            )
        angle_layout.addStretch()
        angle_layout.addWidget(QLabel("方位:"))
        angle_layout.addWidget(self.lbl_az_val)
        angle_layout.addSpacing(24)
        angle_layout.addWidget(QLabel("俯仰:"))
        angle_layout.addWidget(self.lbl_el_val)
        angle_layout.addSpacing(24)
        settings_button = QPushButton("设置坐标")
        settings_button.setFixedWidth(96)
        settings_button.clicked.connect(self.settings_window.show)
        angle_layout.addWidget(settings_button)
        angle_layout.addStretch()
        left_layout.addWidget(angle_group)

        result_group = QGroupBox("目标解算结果")
        self.result_group = result_group
        result_group.setStyleSheet(
            "QGroupBox {color:#FFFFFF; background:#17375E; "
            "border:1px solid #2C527E; border-radius:6px; margin-top:12px;}"
            "QGroupBox::title {subcontrol-origin:margin; left:10px; "
            "padding:0 6px; background:#071A33;}"
        )
        result_group.setFixedHeight(78)
        result_layout = QHBoxLayout(result_group)
        result_layout.setContentsMargins(22, 18, 22, 8)
        result_layout.setSpacing(20)
        self.lbl_geo_lat = QLabel("纬度: --.-----")
        self.lbl_geo_lon = QLabel("经度: ---.-----")
        self.lbl_geo_dist = QLabel("距离: ---- m")
        for label in (self.lbl_geo_lat, self.lbl_geo_lon, self.lbl_geo_dist):
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet(
                "font-size:14px; font-weight:bold; color:#FF8D8D; "
                "background:#102D50; border:1px solid #2A5688; "
                "border-radius:5px; padding:7px 12px;"
            )
            result_layout.addWidget(label, 1)
        left_layout.addWidget(result_group)

        workspace_layout.addWidget(left_panel, 1)
        workspace_layout.addWidget(self._create_manual_ptz_panel())
        online_layout.addWidget(workspace, 1)

        bottom_bar = QFrame()
        bottom_bar.setObjectName("controlBar")
        self.bottom_bar = bottom_bar
        bottom_bar.setStyleSheet(
            "QFrame#controlBar {background:#17375E; border:1px solid #2C527E; "
            "border-radius:6px;}"
            "QLabel {color:#E6F0FF; background:transparent;}"
            "QLineEdit, QComboBox {color:#EEF6FF; background:#061A34; "
            "border:1px solid #2A5688; border-radius:4px; padding:5px;}"
        )
        bottom_bar.setFixedHeight(66)
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(14, 10, 14, 10)
        bottom_layout.setSpacing(8)
        self.input_ip = QLineEdit("127.0.0.1")
        self.input_ip.setFixedWidth(125)
        self.input_port = QLineEdit(str(UDP_DEFAULT_RECEIVE_PORT))
        self.input_port.setFixedWidth(70)
        self.input_rx_port = QLineEdit("56227")
        self.input_rx_port.setFixedWidth(70)
        self.input_rx_port.editingFinished.connect(self.update_rx_port)
        bottom_layout.addWidget(QLabel("IP:"))
        bottom_layout.addWidget(self.input_ip)
        bottom_layout.addWidget(QLabel("Port:"))
        bottom_layout.addWidget(self.input_port)
        bottom_layout.addWidget(QLabel("接收:"))
        bottom_layout.addWidget(self.input_rx_port)
        bottom_layout.addStretch()

        self.combo_target = QComboBox()
        self.combo_target.addItems(["person", "fire"])
        self.combo_target.setFixedWidth(90)
        self.btn_video = QPushButton("启动视频")
        self.btn_video.setObjectName("videoButton")
        self.btn_video.setProperty("streaming", False)
        self.btn_video.setFixedWidth(105)
        self.btn_video.clicked.connect(self.toggle_video)
        self.btn_action = QPushButton("启动跟踪")
        self.btn_action.setObjectName("trackButton")
        self.btn_action.setProperty("tracking", False)
        self.btn_action.setFixedWidth(105)
        self.btn_action.clicked.connect(self.toggle_tracking)
        bottom_layout.addWidget(self.combo_target)
        bottom_layout.addWidget(self.btn_video)
        bottom_layout.addWidget(self.btn_action)
        self.lbl_status = QLabel("系统就绪")
        self.lbl_status.setObjectName("statusLabel")
        self.lbl_status.setMinimumWidth(230)
        self.lbl_status.setMaximumWidth(330)
        self.lbl_status.setAlignment(Qt.AlignCenter)
        bottom_layout.addWidget(self.lbl_status)
        online_layout.addWidget(bottom_bar)
        self.online_scroll = QScrollArea()
        self.online_scroll.setObjectName("onlineScroll")
        self.online_scroll.setWidgetResizable(True)
        self.online_scroll.setFrameShape(QFrame.NoFrame)
        self.online_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.online_scroll.setStyleSheet(
            "QScrollArea#onlineScroll {background:#071A33; border:none;}"
            "QScrollArea#onlineScroll > QWidget > QWidget {background:#071A33;}"
            "QScrollBar:vertical {background:#091E39; width:10px; margin:0;}"
            "QScrollBar::handle:vertical {background:#2A5688; border-radius:5px; "
            "min-height:36px;}"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical "
            "{height:0; background:none;}"
        )
        self.online_scroll.setWidget(online_page)
        self.content_stack.addWidget(self.online_scroll)
        self.content_stack.addWidget(self.build_offline_page())
        main_layout.addWidget(self.content_stack, 1)

    def build_offline_page(self):
        self.offline_thread = None
        self.offline_source_path = ""
        self.offline_is_video = False
        self.offline_current_frame = None
        self.offline_last_log_at = 0.0

        page = SolidBackgroundWidget()
        page.setObjectName("offlinePage")
        set_solid_background(page)
        page.setStyleSheet("QWidget#offlinePage {background:#071A33;}")
        page_layout = QHBoxLayout(page)
        page_layout.setContentsMargins(4, 2, 4, 2)
        page_layout.setSpacing(14)

        control_group = QGroupBox("本地文件与模型")
        offline_group_style = (
            "QGroupBox {color:#FFFFFF; background:#17375E; "
            "border:1px solid #2C527E; border-radius:6px; margin-top:12px; "
            "padding-top:7px; font-weight:600;}"
            "QGroupBox::title {subcontrol-origin:margin; left:10px; padding:0 6px; "
            "color:#FFFFFF; background:#071A33;}"
            "QLabel {background:transparent; border:none;}"
            "QPushButton {color:#FFFFFF; background:#285B9A; "
            "border:1px solid #3B72B6; border-radius:4px; padding:7px 12px; "
            "min-height:26px; font-weight:600;}"
            "QPushButton:hover {background:#3371BA;}"
            "QPushButton:disabled {color:#68809A; background:#183553; "
            "border-color:#294663;}"
            "QTextEdit {color:#EEF6FF; background:#061A34; "
            "border:1px solid #2A5688; border-radius:4px; padding:6px;}"
            "QProgressBar {color:#FFFFFF; background:#071A33; "
            "border:1px solid #2A5688; border-radius:4px; text-align:center;}"
            "QProgressBar::chunk {background:#1E88E5;}"
        )
        set_solid_background(control_group, "#17375E")
        control_group.setStyleSheet(offline_group_style)
        control_group.setFixedWidth(310)
        control_layout = QVBoxLayout(control_group)
        model_hint = QLabel("分析模型")
        model_hint.setStyleSheet("color:#82A9CF;")
        model_name = QLabel("fire_weight.pt")
        model_name.setStyleSheet(
            "color:#65B8FF; font-size:16px; font-weight:700;"
        )
        model_classes = QLabel("检测类别：fire (0)、smoke (1)")
        model_classes.setStyleSheet("color:#82A9CF;")
        model_classes.setWordWrap(True)
        control_layout.addWidget(model_hint)
        control_layout.addWidget(model_name)
        control_layout.addWidget(model_classes)
        control_layout.addSpacing(10)

        self.btn_select_offline_image = QPushButton("选择本地图片")
        self.btn_select_offline_video = QPushButton("选择本地视频")
        self.btn_start_offline = QPushButton("开始分析")
        self.btn_stop_offline = QPushButton("停止分析")
        self.btn_start_offline.setStyleSheet(
            "QPushButton {background:#168747; border-color:#2AA966;}"
            "QPushButton:disabled {background:#183553; border-color:#294663;}"
        )
        self.btn_stop_offline.setStyleSheet(
            "QPushButton {background:#C9273B; border-color:#E04455;}"
            "QPushButton:disabled {background:#183553; border-color:#294663;}"
        )
        self.btn_start_offline.setEnabled(False)
        self.btn_stop_offline.setEnabled(False)
        self.btn_select_offline_image.clicked.connect(self.select_offline_image)
        self.btn_select_offline_video.clicked.connect(self.select_offline_video)
        self.btn_start_offline.clicked.connect(self.start_offline_analysis)
        self.btn_stop_offline.clicked.connect(self.stop_offline_analysis)
        control_layout.addWidget(self.btn_select_offline_image)
        control_layout.addWidget(self.btn_select_offline_video)
        control_layout.addSpacing(6)
        control_layout.addWidget(self.btn_start_offline)
        control_layout.addWidget(self.btn_stop_offline)
        control_layout.addSpacing(12)

        path_hint = QLabel("当前文件")
        path_hint.setStyleSheet("color:#82A9CF;")
        self.offline_path_label = QLabel("尚未选择文件")
        self.offline_path_label.setWordWrap(True)
        self.offline_path_label.setStyleSheet(
            "background:#071A33; border:1px solid #2A5688; "
            "border-radius:4px; padding:8px; color:#DBEAFF;"
        )
        control_layout.addWidget(path_hint)
        control_layout.addWidget(self.offline_path_label)
        control_layout.addStretch()

        preview_group = QGroupBox("离线分析画面")
        set_solid_background(preview_group, "#17375E")
        preview_group.setStyleSheet(offline_group_style)
        preview_layout = QVBoxLayout(preview_group)
        preview_frame = QFrame()
        preview_frame.setObjectName("offlinePreview")
        set_solid_background(preview_frame, "#020A13")
        preview_frame.setStyleSheet(
            "QFrame#offlinePreview {background:#020A13; "
            "border:1px solid #2B5785; border-radius:5px;}"
        )
        preview_frame_layout = QVBoxLayout(preview_frame)
        preview_frame_layout.setContentsMargins(0, 0, 0, 0)
        self.offline_image_label = QLabel("请选择图片或视频")
        self.offline_image_label.setAlignment(Qt.AlignCenter)
        self.offline_image_label.setStyleSheet(
            "color:#6F8BAA; font-size:17px; background:transparent;"
        )
        self.offline_image_label.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Ignored
        )
        preview_frame_layout.addWidget(self.offline_image_label)
        self.offline_progress = QProgressBar()
        self.offline_progress.setRange(0, 100)
        self.offline_progress.setValue(0)
        preview_layout.addWidget(preview_frame, 1)
        preview_layout.addWidget(self.offline_progress)

        result_group = QGroupBox("离线分析记录")
        set_solid_background(result_group, "#17375E")
        result_group.setStyleSheet(offline_group_style)
        result_group.setFixedWidth(350)
        result_layout = QVBoxLayout(result_group)
        self.offline_status_label = QLabel("等待选择本地文件")
        self.offline_status_label.setWordWrap(True)
        self.offline_status_label.setStyleSheet(
            "color:#78C2FF; font-weight:700; padding:6px;"
        )
        self.offline_log = QTextEdit()
        self.offline_log.setReadOnly(True)
        self.offline_log.setPlaceholderText(
            "分析开始后，这里将显示处理进度和检测数量。"
        )
        result_layout.addWidget(self.offline_status_label)
        result_layout.addWidget(self.offline_log, 1)

        page_layout.addWidget(control_group)
        page_layout.addWidget(preview_group, 1)
        page_layout.addWidget(result_group)
        return page

    def switch_page(self, index):
        self.content_stack.setCurrentIndex(index)
        self.btn_online_page.setChecked(index == 0)
        self.btn_offline_page.setChecked(index == 1)
        self.setFocus()

    def select_offline_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择待分析图片",
            "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;所有文件 (*)",
        )
        if path:
            self._set_offline_source(path, False)

    def select_offline_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择待分析视频",
            "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.m4v);;所有文件 (*)",
        )
        if path:
            self._set_offline_source(path, True)

    def _set_offline_source(self, path, is_video):
        if self.offline_thread and self.offline_thread.isRunning():
            self.stop_offline_analysis()
        self.offline_source_path = path
        self.offline_is_video = is_video
        self.offline_current_frame = None
        self.offline_image_label.clear()
        self.offline_image_label.setText(
            "视频已选择，点击开始分析" if is_video
            else "图片已选择，点击开始分析"
        )
        self.offline_path_label.setText(path)
        self.offline_progress.setValue(0)
        source_type = "视频" if is_video else "图片"
        self.offline_status_label.setText(f"已选择{source_type}")
        self.offline_log.clear()
        self.offline_log.append(
            f"[{datetime.now().strftime('%H:%M:%S')}] 已选择{source_type}:\n{path}"
        )
        self.btn_start_offline.setEnabled(True)

    def start_offline_analysis(self):
        if not self.offline_source_path:
            return
        if self.offline_thread and self.offline_thread.isRunning():
            return
        if self.thread.tracking_active:
            self.update_status_label(
                "状态: 请先停止在线跟踪，再启动本地离线分析"
            )
            self.switch_page(0)
            return

        self.offline_progress.setValue(0)
        self.offline_log.append(
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            "开始使用 fire_weight.pt 分析"
        )
        self.offline_thread = HostOfflineAnalysisThread(
            self.offline_source_path, self.offline_is_video
        )
        self.offline_thread.frame_ready.connect(self.update_offline_image)
        self.offline_thread.status_update.connect(self.update_offline_status)
        self.offline_thread.progress_update.connect(
            self.offline_progress.setValue
        )
        self.offline_thread.analysis_finished.connect(
            self.on_offline_analysis_finished
        )
        self.btn_start_offline.setEnabled(False)
        self.btn_select_offline_image.setEnabled(False)
        self.btn_select_offline_video.setEnabled(False)
        self.btn_stop_offline.setEnabled(True)
        self.offline_status_label.setText("正在准备离线分析...")
        self.offline_thread.start()

    def stop_offline_analysis(self):
        thread = getattr(self, "offline_thread", None)
        if thread and thread.isRunning():
            self.offline_status_label.setText("正在停止分析...")
            thread.stop()
        if hasattr(self, "btn_stop_offline"):
            self.btn_stop_offline.setEnabled(False)

    @Slot(QImage)
    def update_offline_image(self, image):
        self.offline_current_frame = image
        self.show_offline_frame()

    @Slot(str)
    def update_offline_status(self, text):
        self.offline_status_label.setText(text)
        now = time.monotonic()
        if now - self.offline_last_log_at >= 1.0 or "完成" in text:
            self.offline_last_log_at = now
            self.offline_log.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] {text}"
            )

    @Slot(bool, str)
    def on_offline_analysis_finished(self, success, message):
        self.btn_select_offline_image.setEnabled(True)
        self.btn_select_offline_video.setEnabled(True)
        self.btn_start_offline.setEnabled(bool(self.offline_source_path))
        self.btn_stop_offline.setEnabled(False)
        if not success or message == "分析已停止":
            self.offline_status_label.setText(message)
            self.offline_log.append(
                f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
            )

    def show_offline_frame(self):
        if self.offline_current_frame:
            pixmap = QPixmap.fromImage(self.offline_current_frame)
            self.offline_image_label.setPixmap(
                pixmap.scaled(
                    self.offline_image_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

    def _create_manual_ptz_panel(self):
        panel = QFrame()
        panel.setObjectName("manualPanel")
        self.manual_panel = panel
        panel.setStyleSheet(
            "QFrame#manualPanel {background:#17375E; border:1px solid #2C527E; "
            "border-radius:6px;}"
            "QGroupBox {color:#FFFFFF; background:#102E50; "
            "border:1px solid #2A5688; border-radius:6px; margin-top:12px;}"
            "QGroupBox::title {subcontrol-origin:margin; left:10px; "
            "padding:0 6px; color:#FFFFFF; background:#17375E;}"
            "QLabel, QCheckBox {color:#E6F0FF; background:transparent;}"
            "QLineEdit {color:#EEF6FF; background:#061A34; "
            "border:1px solid #2A5688; border-radius:4px; padding:5px;}"
            "QPushButton {color:#FFFFFF; background:#285B9A; "
            "border:1px solid #3B72B6; border-radius:4px; padding:5px;}"
            "QPushButton:pressed {background:#1976D2;}"
        )
        panel.setFixedWidth(252)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(10, 8, 10, 10)
        panel_layout.setSpacing(9)

        status_group = QGroupBox("当前状态")
        status_layout = QGridLayout(status_group)
        status_layout.setContentsMargins(12, 18, 12, 10)
        status_layout.setHorizontalSpacing(8)
        status_layout.setVerticalSpacing(6)
        state_fields = [
            ("azimuth", "方位角 (°):"),
            ("elevation", "俯仰角 (°):"),
            ("absoluteZoom", "变焦倍数:"),
            ("focus", "聚焦值:"),
            ("focalLen", "焦距 (mm):"),
            ("horizontalSpeed", "水平速度:"),
            ("verticalSpeed", "垂直速度:"),
        ]
        self.ptz_status_labels = {}
        for row, (key, title) in enumerate(state_fields):
            title_label = QLabel(title)
            title_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value_label = QLabel("N/A")
            value_label.setObjectName("ptzStateValue")
            value_label.setMinimumWidth(78)
            value_label.setAlignment(Qt.AlignCenter)
            self.ptz_status_labels[key] = value_label
            status_layout.addWidget(title_label, row, 0)
            status_layout.addWidget(value_label, row, 1)
        panel_layout.addWidget(status_group)

        self.manual_control_group = QGroupBox("PTZ 手动控制")
        control_layout = QGridLayout(self.manual_control_group)
        control_layout.setContentsMargins(14, 20, 14, 12)
        control_layout.setHorizontalSpacing(10)
        control_layout.setVerticalSpacing(8)
        direction_specs = [
            ("up", "↑", 0, 1),
            ("left", "←", 1, 0),
            ("right", "→", 1, 2),
            ("down", "↓", 2, 1),
        ]
        self.direction_buttons = {}
        for action, text, row, column in direction_specs:
            button = QPushButton(text)
            button.setObjectName("directionButton")
            button.setFixedSize(54, 38)
            button.setFocusPolicy(Qt.NoFocus)
            button.pressed.connect(
                lambda selected=action: self._set_manual_direction(selected, True)
            )
            button.released.connect(
                lambda selected=action: self._set_manual_direction(selected, False)
            )
            self.direction_buttons[action] = button
            control_layout.addWidget(button, row, column, Qt.AlignCenter)
        keyboard_tip = QLabel("键盘控制：使用 ↑ ↓ ← → 方向键")
        keyboard_tip.setAlignment(Qt.AlignCenter)
        keyboard_tip.setStyleSheet("color:#8FB3D9; font-size:11px;")
        control_layout.addWidget(keyboard_tip, 3, 0, 1, 3)
        panel_layout.addWidget(self.manual_control_group)

        self.manual_speed_group = QGroupBox("速度设置")
        speed_layout = QGridLayout(self.manual_speed_group)
        speed_layout.setContentsMargins(12, 20, 12, 12)
        speed_layout.setHorizontalSpacing(7)
        speed_layout.setVerticalSpacing(8)
        validator = QDoubleValidator(1.0, 100.0, 1, self)
        validator.setNotation(QDoubleValidator.StandardNotation)
        self.input_pan_speed = QLineEdit("20.0")
        self.input_tilt_speed = QLineEdit("20.0")
        self.input_pan_speed.setValidator(validator)
        self.input_tilt_speed.setValidator(validator)
        self.input_pan_speed.setFixedWidth(72)
        self.input_tilt_speed.setFixedWidth(72)
        speed_layout.addWidget(QLabel("左右速度 (1-100):"), 0, 0)
        speed_layout.addWidget(self.input_pan_speed, 0, 1)
        speed_layout.addWidget(QLabel("上下速度 (1-100):"), 1, 0)
        speed_layout.addWidget(self.input_tilt_speed, 1, 1)
        apply_speed_button = QPushButton("应用速度")
        apply_speed_button.clicked.connect(self.apply_manual_speed)
        speed_layout.addWidget(apply_speed_button, 2, 0, 1, 2, Qt.AlignCenter)
        panel_layout.addWidget(self.manual_speed_group)
        panel_layout.addStretch()
        return panel

    def update_rx_port(self):
        try:
            port = int(self.input_rx_port.text())
            self.rx_thread.stop()
            self.rx_thread.is_running = True
            self.rx_thread.set_port(port)
            self.rx_thread.start()
        except ValueError:
            self.update_status_label("状态: 接收端口格式错误")

    def eventFilter(self, watched, event):
        if self.isActiveWindow() and event.type() in (
            QEvent.KeyPress,
            QEvent.KeyRelease,
        ):
            key_actions = {
                Qt.Key_Up: "up",
                Qt.Key_Down: "down",
                Qt.Key_Left: "left",
                Qt.Key_Right: "right",
            }
            action = key_actions.get(event.key())
            if action is not None:
                if not event.isAutoRepeat():
                    self._set_manual_direction(
                        action, event.type() == QEvent.KeyPress
                    )
                return True
        return super().eventFilter(watched, event)

    def _set_manual_direction(self, action, pressed):
        if self.thread.tracking_active:
            if pressed:
                self.update_status_label("状态: 请先停止自动跟踪，再使用手动云台")
            return
        if self.manual_direction_state[action] == pressed:
            return
        self.manual_direction_state[action] = pressed

        pan = 0.0
        tilt = 0.0
        if self.manual_direction_state["left"]:
            pan -= self.manual_pan_speed
        if self.manual_direction_state["right"]:
            pan += self.manual_pan_speed
        if self.manual_direction_state["up"]:
            tilt += self.manual_tilt_speed
        if self.manual_direction_state["down"]:
            tilt -= self.manual_tilt_speed
        self.thread.manual_move(pan, tilt)

        if pan or tilt:
            self.update_status_label(
                f"状态: 手动控制 pan={pan:.1f}, tilt={tilt:.1f}"
            )
        else:
            self.update_status_label("状态: 手动云台已停止")

    def _clear_manual_direction_state(self, send_stop=False):
        for action in self.manual_direction_state:
            self.manual_direction_state[action] = False
        if send_stop:
            self.thread.manual_move(0.0, 0.0)

    def apply_manual_speed(self):
        try:
            pan_speed = float(self.input_pan_speed.text())
            tilt_speed = float(self.input_tilt_speed.text())
        except ValueError:
            self.update_status_label("状态: 请输入 1 到 100 之间的速度")
            return

        self.manual_pan_speed = max(1.0, min(100.0, pan_speed))
        self.manual_tilt_speed = max(1.0, min(100.0, tilt_speed))
        self.input_pan_speed.setText(f"{self.manual_pan_speed:.1f}")
        self.input_tilt_speed.setText(f"{self.manual_tilt_speed:.1f}")
        self.update_status_label(
            "状态: 手动速度已更新 "
            f"左右={self.manual_pan_speed:.1f}, 上下={self.manual_tilt_speed:.1f}"
        )

    def toggle_tracking(self):
        active = not self.thread.tracking_active
        if active:
            if not self.thread.video_active:
                self.update_status_label("状态: 请先启动视频流")
                return
            try:
                self.thread.set_network_config(
                    self.input_ip.text(), self.input_port.text()
                )
            except ValueError:
                self.update_status_label("状态: 目标端口格式错误")
                return
            self.thread.set_target(self.combo_target.currentText())
        self.thread.set_tracking_state(active)
        self._set_tracking_ui(active)

    def _set_tracking_ui(self, active):
        self.btn_action.setText("停止跟踪" if active else "启动跟踪")
        self.btn_action.setProperty("tracking", active)
        self.btn_action.style().unpolish(self.btn_action)
        self.btn_action.style().polish(self.btn_action)
        self.combo_target.setEnabled(not active)
        self.input_ip.setEnabled(not active)
        self.input_port.setEnabled(not active)
        self.manual_control_group.setEnabled(not active)
        self.manual_speed_group.setEnabled(not active)
        if active:
            self._clear_manual_direction_state()

    def toggle_video(self):
        active = not self.thread.video_active
        if not active and self.thread.tracking_active:
            self.thread.set_tracking_state(False)
            self._set_tracking_ui(False)

        self.thread.set_video_state(active)
        self.btn_video.setText("停止视频" if active else "启动视频")
        self.btn_video.setProperty("streaming", active)
        self.btn_video.style().unpolish(self.btn_video)
        self.btn_video.style().polish(self.btn_video)

        if active:
            self.lbl_local.clear()
            self.lbl_local.setText("正在连接 RTSP 视频流…")
        else:
            self.lbl_local.clear()
            self.lbl_local.setText("视频流已停止，点击“启动视频”重新连接")

    @Slot(QImage)
    def update_image(self, image):
        if self.lbl_local.width() > 0:
            # 16:9 显示控件会在绘制时根据窗口大小自动缩放。
            self.lbl_local.setPixmap(QPixmap.fromImage(image))

    @Slot(QImage)
    def update_remote_image(self, image):
        if self.lbl_remote.width() > 0:
            self.lbl_remote.setPixmap(QPixmap.fromImage(image))

    @Slot(str)
    def update_status_label(self, text):
        self.lbl_status.setText(text)

    @Slot(float, float)
    def update_angle_display(self, azimuth, elevation):
        self.lbl_az_val.setText(f"{azimuth:.2f}°")
        self.lbl_el_val.setText(f"{elevation:.2f}°")

    @Slot(dict)
    def update_ptz_state_display(self, state):
        for key, label in self.ptz_status_labels.items():
            if key not in state:
                continue
            value = state[key]
            if isinstance(value, float):
                label.setText(f"{value:.3f}")
            else:
                label.setText(str(value))

    @Slot(str, str, str)
    def update_geo_result(self, latitude, longitude, distance):
        self.lbl_geo_lat.setText(f"纬度: {latitude}")
        self.lbl_geo_lon.setText(f"经度: {longitude}")
        self.lbl_geo_dist.setText(f"距离: {distance} m")

    def resizeEvent(self, event):
        if hasattr(self, "offline_image_label"):
            self.show_offline_frame()
        super().resizeEvent(event)

    def closeEvent(self, event):
        self.stop_offline_analysis()
        self.settings_window.close()
        self._clear_manual_direction_state(send_stop=True)
        camera_stopped = self.thread.stop()
        self.rx_thread.stop()
        if camera_stopped:
            QApplication.instance().removeEventFilter(self)
            event.accept()
        else:
            self.update_status_label("状态: 视频线程仍在退出，请稍后再次关闭")
            event.ignore()


def get_absoluteEx(ipc_base_url: str, channel: int, digestAuth: HTTPDigestAuth):
    state = {}
    url = f"{ipc_base_url}/ISAPI/PTZCtrl/channels/{channel}/absoluteEx"
    elements_to_extract = {
        'azimuth': float,
        'elevation': float,
        'absoluteZoom': float,
        'focus': int,
        'focalLen': int,
        'horizontalSpeed': float,
        'verticalSpeed': float,
        'zoomType': str,
    }
    try:
        response = requests.get(
            url,
            auth=digestAuth,
            timeout=5,
        )

        if response.status_code == 200:
            # 解析XML响应
            root = ET.fromstring(response.text)
            for tag_name, converter in elements_to_extract.items():
                elem = root.find(f'.//{{*}}{tag_name}')
                if elem is not None and elem.text is not None:
                    if converter == str:
                        state[tag_name] = elem.text
                    else:
                        state[tag_name] = converter(elem.text)
        else:
            raise RuntimeError(f"HTTP {response.status_code}")
    except Exception as e:
        return {"status": "failed", "error": e}
    return {"status": "success", "state": state}


def try_set_absoluteEx(
    ipc_base_url: str,
    channel: int,
    digestAuth: HTTPDigestAuth,
    config,
):
    xml_template = '''<?xml version="1.0" encoding="UTF-8"?>
<PTZAbsoluteEx version="2.0" xmlns="http://www.std-cgi.com/ver20/XMLSchema">
<elevation>{elevation}</elevation>
<azimuth>{azimuth}</azimuth>
<absoluteZoom>{absoluteZoom}</absoluteZoom>
<focus>{focus}</focus>
<focalLen>{focalLen}</focalLen>
<horizontalSpeed>{horizontalSpeed}</horizontalSpeed>
<verticalSpeed>{verticalSpeed}</verticalSpeed>
<zoomType>{zoomType}</zoomType>
</PTZAbsoluteEx>'''

    try:
        xml_body = xml_template.format(**config)
        url = f"{ipc_base_url}/ISAPI/PTZCtrl/channels/{channel}/absoluteEx"
        response = requests.put(
            url,
            data=xml_body,
            auth=digestAuth,
            headers={'Content-Type': 'application/xml'},
            timeout=10,
        )

        if response.status_code != 200:
            return {"status": "failed", "error": response.text}
    except Exception as e:
        return {"status": "failed", "error": e}
    return {"status": "success"}


def set_continuous(
    nvr_base_url: str,
    channel: int,
    digestAuth: HTTPDigestAuth,
    pan: int,
    tilt: int,
):
    try:
        xml_template = '''<PTZData><pan>{pan}</pan><tilt>{tilt}</tilt></PTZData>'''
        xml_body = xml_template.format(
            pan=int(pan),
            tilt=int(tilt),
        )
        url = f"{nvr_base_url}/ISAPI/ContentMgmt/PTZCtrlProxy/channels/{channel}/continuous"
        response = requests.put(
            url,
            data=xml_body,
            auth=digestAuth,
            headers={'Content-Type': 'application/xml'},
            timeout=10,
        )

        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}")
    except Exception as e:
        return {"status": "failed", "error": e}
    return {"status": "success"}


from PySide6.QtWidgets import (
    QDialog,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QScrollArea,
    QSplitter,
    QStackedWidget,
)

class LatestFrameReader:
    """持续抽取 RTSP 帧，只保存最新的一帧，防止旧帧排队。"""

    def __init__(self, url, status_callback):
        self.url = url
        self.status_callback = status_callback
        self.running = False
        self.capture = None
        self.frame = None
        self.frame_time = 0.0
        self.sequence = 0
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _open_capture(self):
        capture = cv2.VideoCapture()
        params = []
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            params.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000])
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            params.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000])

        try:
            opened = capture.open(self.url, cv2.CAP_ANY, params)
        except (TypeError, cv2.error):
            opened = capture.open(self.url)
        if not opened:
            capture.release()
            return None
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _interruptible_wait(self, seconds):
        end_time = time.monotonic() + seconds
        while self.running and time.monotonic() < end_time:
            time.sleep(0.05)

    def _run(self):
        while self.running:
            self.status_callback("状态: 正在连接低延迟 RTSP 视频流...")
            self.capture = self._open_capture()
            if self.capture is None:
                self.status_callback("状态: RTSP 连接失败，2 秒后重试")
                self._interruptible_wait(2.0)
                continue

            self.status_callback("状态: 低延迟 RTSP 视频流连接成功")
            while self.running:
                ok, frame = self.capture.read()
                if not ok:
                    self.status_callback("状态: 视频帧读取失败，正在重连")
                    break
                with self.lock:
                    self.frame = frame
                    self.frame_time = time.monotonic()
                    self.sequence += 1

            self.capture.release()
            self.capture = None
            if self.running:
                self._interruptible_wait(0.5)

    def get_latest(self, previous_sequence):
        with self.lock:
            if self.frame is None or self.sequence == previous_sequence:
                return None
            return self.sequence, self.frame_time, self.frame

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=6.0)


class StableNetworkPtzBackend:
    """自带完整状态回调，兼容没有 ptz_state_update 的旧版基础脚本。"""

    def __init__(self, state_callback, status_callback):
        self.ipc_base_url = f"http://{IPC_IP}"
        self.nvr_base_url = f"http://{NVR_IP}"
        self.auth = HTTPDigestAuth(USERNAME, PASSWORD)
        self.channel = CHANNEL
        self.state_callback = state_callback
        self.status_callback = status_callback

        self._running = False
        self._command_event = threading.Event()
        self._command_lock = threading.Lock()
        self._desired_command = (0, 0)
        self._last_queued_command = None
        self._last_queue_time = 0.0
        self._threads = []

    def start(self):
        self._running = True
        self._threads = [
            threading.Thread(target=self._control_loop, daemon=True),
            threading.Thread(target=self._status_loop, daemon=True),
        ]
        for worker in self._threads:
            worker.start()

    def move(self, pan, tilt, force=False):
        pan = int(max(-100, min(100, pan)))
        tilt = int(max(-100, min(100, tilt)))
        command = (pan, tilt)
        now = time.monotonic()
        if (
            not force
            and command == self._last_queued_command
            and now - self._last_queue_time < 0.5
        ):
            return
        with self._command_lock:
            self._desired_command = command
            self._last_queued_command = command
            self._last_queue_time = now
        self._command_event.set()

    def _control_loop(self):
        while self._running:
            if not self._command_event.wait(0.2):
                continue
            self._command_event.clear()
            with self._command_lock:
                pan, tilt = self._desired_command
            result = set_continuous(
                self.nvr_base_url, self.channel, self.auth, pan, tilt
            )
            if result.get("status") != "success" and self._running:
                self.status_callback(
                    f"状态: PTZ 控制失败 - {result.get('error', '未知错误')}"
                )

    def _status_loop(self):
        last_error = None
        while self._running:
            result = get_absoluteEx(
                self.ipc_base_url, self.channel, self.auth
            )
            if result.get("status") == "success":
                self.state_callback(dict(result.get("state", {})))
                last_error = None
            else:
                error = str(result.get("error", "未知错误"))
                if error != last_error:
                    self.status_callback(f"状态: 云台状态读取失败 - {error}")
                    last_error = error
            for _ in range(20):
                if not self._running:
                    return
                time.sleep(0.1)

    def stop(self):
        if not self._running:
            return
        set_continuous(
            self.nvr_base_url, self.channel, self.auth, 0, 0
        )
        self._running = False
        self._command_event.set()
        for worker in self._threads:
            worker.join(timeout=1.0)


class StableCameraThread(CameraThread):
    """使用滞回锁定、误差中值滤波和短脉冲控制的跟踪线程。"""

    ptz_state_update = Signal(dict)

    def __init__(self):
        super().__init__()
        if not hasattr(self, "video_active"):
            self.video_active = False
        # 替换基础脚本中的 PTZ 后台，因此基础脚本新旧版本均可运行。
        self.ptz = StableNetworkPtzBackend(
            self._on_full_ptz_state, self._on_stable_ptz_status
        )

        # 稳定控制参数。中心进入区小、保持区大，形成滞回，避免边界来回切换。
        self.center_enter_zone = 38
        self.center_hold_zone = 72
        self.center_confirm_seconds = 0.35
        self.unlock_confirm_frames = 3

        # 网络视频存在延迟时不持续转动，而是短动一下、停止并等待新画面。
        self.pulse_duration = 0.10
        self.settle_duration = 0.32
        self.stable_kp = 0.08
        self.stable_min_speed = 5
        self.stable_max_speed = 18

        self.error_x_history = deque(maxlen=5)
        self.error_y_history = deque(maxlen=5)
        self.centered_since = None
        self.center_locked = False
        self.outside_hold_count = 0

        self._control_lock = threading.Lock()
        self._pulse_timer = None
        self._motion_active = False
        self._next_pulse_time = 0.0
        self._reader = None
        self.previous_target_center = None
        self.missing_target_frames = 0

    def _on_full_ptz_state(self, state):
        azimuth = float(state.get("azimuth", 0.0))
        elevation = float(state.get("elevation", 0.0))
        self.current_az = azimuth
        self.current_el = elevation
        self.angle_update.emit(azimuth, elevation)
        self.ptz_state_update.emit(dict(state))

    def _on_stable_ptz_status(self, message):
        self.status_update.emit(message)

    def _reset_stable_control(self, send_stop=True):
        self.error_x_history.clear()
        self.error_y_history.clear()
        self.centered_since = None
        self.center_locked = False
        self.outside_hold_count = 0
        self.prev_cx = self.prev_cy = 0.0
        self._stop_latency_motion(force=send_stop)

    def _finish_pulse(self):
        self.ptz.move(0, 0, force=True)
        with self._control_lock:
            self._motion_active = False
            self._pulse_timer = None
            self._next_pulse_time = time.monotonic() + self.settle_duration

    def _start_pulse(self, pan, tilt):
        now = time.monotonic()
        with self._control_lock:
            if self._motion_active or now < self._next_pulse_time:
                return False
            self._motion_active = True
            self.ptz.move(pan, tilt, force=True)
            self._pulse_timer = threading.Timer(
                self.pulse_duration, self._finish_pulse
            )
            self._pulse_timer.daemon = True
            self._pulse_timer.start()
        return True

    def _stop_latency_motion(self, force=False):
        with self._control_lock:
            timer = self._pulse_timer
            was_active = self._motion_active
            self._pulse_timer = None
            self._motion_active = False
            self._next_pulse_time = max(
                self._next_pulse_time,
                time.monotonic() + self.settle_duration,
            )
        if timer is not None:
            timer.cancel()
        if was_active or force:
            self.ptz.move(0, 0, force=True)

    def set_tracking_state(self, active):
        self._reset_stable_control(send_stop=True)
        super().set_tracking_state(active)

    def set_target(self, label):
        self._reset_stable_control(send_stop=True)
        super().set_target(label)

    def set_video_state(self, active):
        if not active:
            self._reset_stable_control(send_stop=True)
        parent_method = getattr(super(), "set_video_state", None)
        if parent_method is not None:
            parent_method(active)
        else:
            self.video_active = bool(active)
            if not active:
                self.tracking_active = False
                self.ptz.move(0, 0, force=True)
            self.status_update.emit(
                "状态: 正在启动视频流..." if active else "状态: 视频流已停止"
            )

    @staticmethod
    def _draw_text(frame, text, x, y, color):
        cv2.putText(
            frame,
            text,
            (x, max(22, y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
        )

    def _filtered_error(self, raw_dx, raw_dy):
        self.error_x_history.append(raw_dx)
        self.error_y_history.append(raw_dy)
        return (
            int(np.median(self.error_x_history)),
            int(np.median(self.error_y_history)),
        )

    def _select_detection(self, result, frame_width, frame_height):
        """不用 lap：优先选择距离上一帧目标最近的检测框。"""
        if result.boxes is None or len(result.boxes) == 0:
            self.missing_target_frames += 1
            if self.missing_target_frames >= 5:
                self.previous_target_center = None
                self.locked_id = None
            return None

        boxes = result.boxes.xyxy.cpu().tolist()
        targets = []
        for coords in boxes:
            x1, y1, x2, y2 = map(int, coords)
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            targets.append(
                {
                    "id": 1,
                    "box": (x1, y1, x2, y2),
                    "center": center,
                    "area": max(0, x2 - x1) * max(0, y2 - y1),
                }
            )

        new_lock = self.previous_target_center is None
        if new_lock:
            selected = max(targets, key=lambda target: target["area"])
        else:
            previous_x, previous_y = self.previous_target_center
            selected = min(
                targets,
                key=lambda target: (
                    (target["center"][0] - previous_x) ** 2
                    + (target["center"][1] - previous_y) ** 2
                ),
            )
            distance = (
                (selected["center"][0] - previous_x) ** 2
                + (selected["center"][1] - previous_y) ** 2
            ) ** 0.5
            association_limit = 0.25 * (frame_width**2 + frame_height**2) ** 0.5
            if distance > association_limit:
                selected = max(targets, key=lambda target: target["area"])

        self.previous_target_center = selected["center"]
        self.missing_target_frames = 0
        if new_lock:
            self.locked_id = selected["id"]
            self.is_acked = False
            self.last_send_time = 0.0
            self.temp_local_path = None
            self.memory_az = None
        return selected

    def _calculate_pulse_speed(self, dx, dy):
        pan = int(dx * self.stable_kp)
        tilt = int(-dy * self.stable_kp)
        pan = max(-self.stable_max_speed, min(self.stable_max_speed, pan))
        tilt = max(-self.stable_max_speed, min(self.stable_max_speed, tilt))

        if pan:
            pan = (1 if pan > 0 else -1) * max(abs(pan), self.stable_min_speed)
        if tilt:
            tilt = (1 if tilt > 0 else -1) * max(
                abs(tilt), self.stable_min_speed
            )
        return pan, tilt

    def _archive_stable_lock(self, frame, target_id, x1, y1):
        self.memory_az = self.current_az
        if self.temp_local_path is None:
            timestamp = datetime.now().strftime("%H-%M-%S-%f")[:-3]
            filename = os.path.join(
                IMG_SAVE_DIR, f"img_{timestamp}_LOCAL_STABLE.jpg"
            )
            if cv2.imwrite(filename, frame):
                self.temp_local_path = filename

        now = time.time()
        if not self.is_acked and now - self.last_send_time > 3.0:
            self.send_locked_data(target_id, self.current_az, self.current_el)
            self.last_send_time = now
            self._draw_text(frame, "SENDING...", x1, y1 - 20, (0, 0, 255))
        elif self.is_acked:
            self._draw_text(frame, "ACK OK", x1, y1 - 20, (255, 0, 0))

    def _track_frame(self, frame):
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        cv2.line(frame, (center_x, 0), (center_x, height), (180, 180, 180), 1)
        cv2.line(frame, (0, center_y), (width, center_y), (180, 180, 180), 1)
        cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

        if not self.tracking_active or self.model is None:
            return frame

        try:
            result = self.model.predict(
                frame,
                classes=[self.class_map.get(self.target_label, 0)],
                verbose=False,
                conf=0.25,
            )[0]
        except Exception as error:
            self.status_update.emit(f"状态: 目标检测失败 - {error}")
            self._reset_stable_control(send_stop=False)
            return frame

        target = self._select_detection(result, width, height)
        if target is None:
            self._reset_stable_control(send_stop=False)
            return frame

        x1, y1, x2, y2 = target["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 170, 70), 2)
        self._draw_text(frame, f"ID:{target['id']}", x1, y1 - 38, (0, 150, 255))

        raw_x = (x1 + x2) / 2.0
        raw_y = (y1 + y2) / 2.0
        dx, dy = self._filtered_error(raw_x - center_x, raw_y - center_y)
        now = time.monotonic()

        in_enter_zone = (
            abs(dx) <= self.center_enter_zone
            and abs(dy) <= self.center_enter_zone
        )
        in_hold_zone = (
            abs(dx) <= self.center_hold_zone
            and abs(dy) <= self.center_hold_zone
        )

        if in_enter_zone:
            self.outside_hold_count = 0
            if self.centered_since is None:
                self.centered_since = now
            self._stop_latency_motion()
            if now - self.centered_since >= self.center_confirm_seconds:
                self.center_locked = True
                self._draw_text(frame, "STABLE LOCK", x1, y1 - 10, (0, 190, 0))
                self._archive_stable_lock(frame, target["id"], x1, y1)
            else:
                self._draw_text(frame, "STABILIZING", x1, y1 - 10, (0, 180, 220))
            return frame

        self.centered_since = None
        if self.center_locked and in_hold_zone:
            self.outside_hold_count = 0
            self._stop_latency_motion()
            self._draw_text(frame, "HOLD", x1, y1 - 10, (0, 190, 0))
            self._archive_stable_lock(frame, target["id"], x1, y1)
            return frame

        if self.center_locked:
            self.outside_hold_count += 1
            if self.outside_hold_count < self.unlock_confirm_frames:
                self._stop_latency_motion()
                self._draw_text(frame, "HOLD CHECK", x1, y1 - 10, (0, 180, 220))
                return frame
            self.center_locked = False
            self.outside_hold_count = 0

        pan, tilt = self._calculate_pulse_speed(dx, dy)
        if self._start_pulse(pan, tilt):
            self._draw_text(frame, "PTZ PULSE", x1, y1 - 10, (255, 120, 0))
        else:
            self._draw_text(frame, "WAIT NEW FRAME", x1, y1 - 10, (150, 150, 0))
        return frame

    def run(self):
        self.ptz.start()
        if YOLO is None:
            self.status_update.emit("状态: 缺少 ultralytics，无法启动跟踪")
        else:
            try:
                self.model = YOLO(MODEL_PATH)
                if hasattr(self, "loaded_model_label"):
                    self.loaded_model_label = "person"
            except Exception as error:
                self.status_update.emit(f"状态: 模型加载失败 - {error}")

        last_sequence = -1
        while self.is_running:
            if not self.video_active:
                if self._reader is not None:
                    self._reader.stop()
                    self._reader = None
                    last_sequence = -1
                time.sleep(0.05)
                continue

            if self._reader is None:
                self._reader = LatestFrameReader(
                    self._rtsp_url(), self.status_update.emit
                )
                self._reader.start()

            latest = self._reader.get_latest(last_sequence)
            if latest is None:
                time.sleep(0.005)
                continue
            last_sequence, frame_time, frame = latest

            # 本机已经积压超过 0.5 秒的帧不用于控制。
            if time.monotonic() - frame_time > 0.5:
                continue

            frame = self._track_frame(frame)
            with self.frame_lock:
                self.current_frame = frame.copy()
            self.frame_ready.emit(self._to_qimage(frame))

        if self._reader is not None:
            self._reader.stop()
            self._reader = None
        self._reset_stable_control(send_stop=True)
        self.ptz.stop()
        self.send_sock.close()

    def stop(self):
        self._stop_latency_motion(force=True)
        return super().stop()


class StableMainWindow(MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle("智能检测系统 - 低延迟稳定跟踪版")
        self.resize(1380, 790)
        self.setMinimumSize(1160, 680)

        self.manual_pan_speed = 20.0
        self.manual_tilt_speed = 20.0
        self.manual_direction_state = {
            "up": False,
            "down": False,
            "left": False,
            "right": False,
        }

        self.settings_window = SettingsWindow()
        self.thread = StableCameraThread()
        self.thread.frame_ready.connect(self.update_image)
        self.thread.remote_frame_ready.connect(self.update_remote_image)
        self.thread.status_update.connect(self.update_status_label)
        self.thread.angle_update.connect(self.update_angle_display)
        self.thread.ptz_state_update.connect(self.update_ptz_state_display)
        self.thread.geo_result_signal.connect(self.update_geo_result)
        self.settings_window.params_updated.connect(self.thread.set_gps_coords)
        self.settings_window.emit_params()

        self.rx_thread = UdpReceiverThread()
        self.rx_thread.save_trigger.connect(self.thread.on_save_command)
        self.rx_thread.ack_received.connect(self.thread.on_ack_received)
        self.rx_thread.start()

        self.setup_ui()
        QApplication.instance().installEventFilter(self)
        self.thread.start()

    def _create_manual_ptz_panel(self):
        panel = super()._create_manual_ptz_panel()
        for group in panel.findChildren(QGroupBox):
            if group.title() == "当前状态":
                group.setParent(None)
                group.deleteLater()
                break
        panel.setFixedWidth(300)
        return panel

    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def setup_ui(self):
        super().setup_ui()

        status_group = next(
            group
            for group in self.findChildren(QGroupBox)
            if group.title() == "实时网络云台数据"
        )
        status_group.setFixedHeight(104)
        layout = status_group.layout()
        self._clear_layout(layout)
        layout.setContentsMargins(14, 18, 14, 9)
        layout.setSpacing(8)

        fields = [
            ("azimuth", "方位角", "°"),
            ("elevation", "俯仰角", "°"),
            ("absoluteZoom", "变焦倍数", ""),
            ("focus", "聚焦值", ""),
            ("focalLen", "焦距", "mm"),
            ("horizontalSpeed", "水平速度", ""),
            ("verticalSpeed", "垂直速度", ""),
        ]
        self.ptz_status_labels = {}
        for key, title, unit in fields:
            card = QFrame()
            card.setObjectName("statusItem")
            card.setStyleSheet(
                "QFrame#statusItem {background:#102D50; border:1px solid #2A5688; "
                "border-radius:6px;}"
            )
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(7, 5, 7, 5)
            card_layout.setSpacing(2)
            title_label = QLabel(f"{title}{f' ({unit})' if unit else ''}")
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet("color:#8FB3D9; font-size:11px;")
            value_label = QLabel("N/A")
            value_label.setObjectName("ptzStateValue")
            value_label.setAlignment(Qt.AlignCenter)
            value_label.setStyleSheet(
                "color:#62B7FF; font-size:14px; font-weight:700; "
                "background:transparent; border:none;"
            )
            card_layout.addWidget(title_label)
            card_layout.addWidget(value_label)
            self.ptz_status_labels[key] = value_label
            layout.addWidget(card, 1)

        self.lbl_az_val = self.ptz_status_labels["azimuth"]
        self.lbl_el_val = self.ptz_status_labels["elevation"]
        settings_button = QPushButton("设置坐标")
        settings_button.setFixedWidth(92)
        settings_button.clicked.connect(self.settings_window.show)
        layout.addWidget(settings_button)

    def update_ptz_state_display(self, state):
        for key, label in self.ptz_status_labels.items():
            if key not in state:
                continue
            value = state[key]
            label.setText(f"{value:.3f}" if isinstance(value, float) else str(value))


class SmoothCameraThread(StableCameraThread):
    """连续、渐变、带滞回锁定的云台跟踪线程。"""

    def __init__(self):
        super().__init__()

        # 中心锁定：进入区较小，保持区较大，轻微移动不会退出锁定。
        self.smooth_enter_zone = 42   # 进入锁定区域的半径
        self.smooth_hold_zone = 100    # 锁定后保持锁定的半径
        self.smooth_lock_confirm_seconds = 0.25
        self.smooth_unlock_confirm_frames = 2

        # 连续控制：远距离速度高，靠近中心自动减速。
        self.smooth_kp = 0.105
        self.smooth_min_speed = 5.0
        self.smooth_max_speed = 32.0
        self.command_interval = 0.08
        self.acceleration_per_second = 90.0
        self.deceleration_per_second = 145.0

        # 预测用于补偿一部分 RTSP 延迟，位移限制避免异常检测产生大命令。
        self.position_alpha = 0.58
        self.velocity_beta = 0.12
        self.prediction_horizon = 0.16
        self.max_prediction_pixels = 85.0
        self.max_image_velocity = 900.0

        self.raw_x_history = deque(maxlen=3)
        self.raw_y_history = deque(maxlen=3)
        self.filtered_x = None
        self.filtered_y = None
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.last_filter_time = None

        self.smooth_centered_since = None
        self.smooth_center_locked = False
        self.smooth_outside_count = 0

        self._smooth_command_lock = threading.Lock()
        self._current_pan_command = 0.0
        self._current_tilt_command = 0.0
        self._last_command_update = 0.0
        self._last_command_sent = None
        self._last_command_sent_time = 0.0

    def _reset_motion_filter(self):
        self.raw_x_history.clear()
        self.raw_y_history.clear()
        self.filtered_x = None
        self.filtered_y = None
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.last_filter_time = None

    def _reset_smooth_control(self, send_stop=True, reset_filter=True):
        self.smooth_centered_since = None
        self.smooth_center_locked = False
        self.smooth_outside_count = 0
        if reset_filter:
            self._reset_motion_filter()
        self._send_smooth_command(0.0, 0.0, force=send_stop, immediate_stop=True)

    def set_tracking_state(self, active):
        self._reset_smooth_control(send_stop=True)
        super().set_tracking_state(active)

    def set_target(self, label):
        self._reset_smooth_control(send_stop=True)
        super().set_target(label)

    def set_video_state(self, active):
        if not active:
            self._reset_smooth_control(send_stop=True)
        super().set_video_state(active)

    @staticmethod
    def _approach(current, target, maximum_change):
        if target > current:
            return min(target, current + maximum_change)
        return max(target, current - maximum_change)

    def _send_smooth_command(
        self, desired_pan, desired_tilt, force=False, immediate_stop=False
    ):
        """按固定频率发送命令，并限制相邻命令间的速度变化。"""
        now = time.monotonic()
        command_to_send = None
        with self._smooth_command_lock:
            if immediate_stop:
                self._current_pan_command = 0.0
                self._current_tilt_command = 0.0
                self._last_command_update = now
            else:
                if not force and now - self._last_command_update < self.command_interval:
                    return False
                dt = (
                    self.command_interval
                    if self._last_command_update == 0.0
                    else min(0.25, max(0.02, now - self._last_command_update))
                )

                pan_limit = (
                    self.deceleration_per_second
                    if abs(desired_pan) < abs(self._current_pan_command)
                    or desired_pan * self._current_pan_command < 0
                    else self.acceleration_per_second
                ) * dt
                tilt_limit = (
                    self.deceleration_per_second
                    if abs(desired_tilt) < abs(self._current_tilt_command)
                    or desired_tilt * self._current_tilt_command < 0
                    else self.acceleration_per_second
                ) * dt
                self._current_pan_command = self._approach(
                    self._current_pan_command, desired_pan, pan_limit
                )
                self._current_tilt_command = self._approach(
                    self._current_tilt_command, desired_tilt, tilt_limit
                )
                self._last_command_update = now

            command = (
                int(round(self._current_pan_command)),
                int(round(self._current_tilt_command)),
            )
            keepalive_due = now - self._last_command_sent_time >= 0.45
            if force or command != self._last_command_sent or keepalive_due:
                self._last_command_sent = command
                self._last_command_sent_time = now
                command_to_send = command

        if command_to_send is not None:
            self.ptz.move(*command_to_send, force=True)
            return True
        return False

    def _update_target_filter(self, raw_x, raw_y, timestamp):
        """返回滤波位置和预测位置。"""
        self.raw_x_history.append(raw_x)
        self.raw_y_history.append(raw_y)
        measured_x = float(np.median(self.raw_x_history))
        measured_y = float(np.median(self.raw_y_history))

        if self.filtered_x is None or self.last_filter_time is None:
            self.filtered_x = measured_x
            self.filtered_y = measured_y
            self.velocity_x = 0.0
            self.velocity_y = 0.0
            self.last_filter_time = timestamp
            return measured_x, measured_y, measured_x, measured_y

        dt = min(0.25, max(0.02, timestamp - self.last_filter_time))
        predicted_x = self.filtered_x + self.velocity_x * dt
        predicted_y = self.filtered_y + self.velocity_y * dt
        residual_x = measured_x - predicted_x
        residual_y = measured_y - predicted_y

        self.filtered_x = predicted_x + self.position_alpha * residual_x
        self.filtered_y = predicted_y + self.position_alpha * residual_y
        self.velocity_x += self.velocity_beta * residual_x / dt
        self.velocity_y += self.velocity_beta * residual_y / dt
        self.velocity_x = max(
            -self.max_image_velocity,
            min(self.max_image_velocity, self.velocity_x),
        )
        self.velocity_y = max(
            -self.max_image_velocity,
            min(self.max_image_velocity, self.velocity_y),
        )
        self.last_filter_time = timestamp

        advance_x = max(
            -self.max_prediction_pixels,
            min(
                self.max_prediction_pixels,
                self.velocity_x * self.prediction_horizon,
            ),
        )
        advance_y = max(
            -self.max_prediction_pixels,
            min(
                self.max_prediction_pixels,
                self.velocity_y * self.prediction_horizon,
            ),
        )
        return (
            self.filtered_x,
            self.filtered_y,
            self.filtered_x + advance_x,
            self.filtered_y + advance_y,
        )

    def _axis_speed(self, error):
        magnitude = abs(error)
        if magnitude <= self.smooth_enter_zone:
            return 0.0
        speed = self.smooth_min_speed + (
            magnitude - self.smooth_enter_zone
        ) * self.smooth_kp
        speed = min(self.smooth_max_speed, speed)
        return speed if error > 0 else -speed

    def _track_frame(self, frame):
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        cv2.line(frame, (center_x, 0), (center_x, height), (180, 180, 180), 1)
        cv2.line(frame, (0, center_y), (width, center_y), (180, 180, 180), 1)
        cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

        if not self.tracking_active or self.model is None:
            return frame

        try:
            result = self.model.predict(
                frame,
                classes=[self.class_map.get(self.target_label, 0)],
                verbose=False,
                conf=0.25,
            )[0]
        except Exception as error:
            self.status_update.emit(f"状态: 目标检测失败 - {error}")
            self._send_smooth_command(0, 0, immediate_stop=True)
            return frame

        target = self._select_detection(result, width, height)
        if target is None:
            self._send_smooth_command(0, 0, immediate_stop=True)
            if self.missing_target_frames >= 5:
                self._reset_motion_filter()
                self.smooth_center_locked = False
                self.smooth_centered_since = None
            return frame

        x1, y1, x2, y2 = target["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 170, 70), 2)
        self._draw_text(frame, f"ID:{target['id']}", x1, y1 - 38, (0, 150, 255))

        raw_x = (x1 + x2) / 2.0
        raw_y = (y1 + y2) / 2.0
        now = time.monotonic()
        filtered_x, filtered_y, predicted_x, predicted_y = self._update_target_filter(
            raw_x, raw_y, now
        )
        lock_dx = filtered_x - center_x
        lock_dy = filtered_y - center_y
        control_dx = predicted_x - center_x
        control_dy = predicted_y - center_y

        inside_enter = (
            abs(lock_dx) <= self.smooth_enter_zone
            and abs(lock_dy) <= self.smooth_enter_zone
        )
        inside_hold = (
            abs(lock_dx) <= self.smooth_hold_zone
            and abs(lock_dy) <= self.smooth_hold_zone
        )

        if inside_enter:
            self.smooth_outside_count = 0
            if self.smooth_centered_since is None:
                self.smooth_centered_since = now
            self._send_smooth_command(0, 0, immediate_stop=True)
            if now - self.smooth_centered_since >= self.smooth_lock_confirm_seconds:
                self.smooth_center_locked = True
                self._draw_text(frame, "SMOOTH LOCK", x1, y1 - 10, (0, 190, 0))
                self._archive_stable_lock(frame, target["id"], x1, y1)
            else:
                self._draw_text(frame, "LOCKING", x1, y1 - 10, (0, 180, 220))
            return frame

        self.smooth_centered_since = None
        if self.smooth_center_locked and inside_hold:
            self.smooth_outside_count = 0
            self._send_smooth_command(0, 0, immediate_stop=True)
            self._draw_text(frame, "LOCK HOLD", x1, y1 - 10, (0, 190, 0))
            self._archive_stable_lock(frame, target["id"], x1, y1)
            return frame

        if self.smooth_center_locked:
            self.smooth_outside_count += 1
            if self.smooth_outside_count < self.smooth_unlock_confirm_frames:
                self._send_smooth_command(0, 0, immediate_stop=True)
                self._draw_text(frame, "HOLD CHECK", x1, y1 - 10, (0, 180, 220))
                return frame
            self.smooth_center_locked = False
            self.smooth_outside_count = 0

        desired_pan = self._axis_speed(control_dx)
        desired_tilt = -self._axis_speed(control_dy)
        self._send_smooth_command(desired_pan, desired_tilt)
        self._draw_text(
            frame,
            f"SMOOTH P:{int(desired_pan)} T:{int(desired_tilt)}",
            x1,
            y1 - 10,
            (255, 120, 0),
        )
        return frame

    def stop(self):
        self._reset_smooth_control(send_stop=True)
        return super().stop()


class SmoothMainWindow(StableMainWindow):
    """界面完全沿用稳定版，只替换自动跟踪线程。"""

    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle("智能检测系统 - 低延迟稳定跟踪版")
        self.resize(1380, 790)
        self.setMinimumSize(1160, 680)

        self.manual_pan_speed = 20.0
        self.manual_tilt_speed = 20.0
        self.manual_direction_state = {
            "up": False,
            "down": False,
            "left": False,
            "right": False,
        }

        self.settings_window = SettingsWindow()
        self.thread = SmoothCameraThread()
        self.thread.frame_ready.connect(self.update_image)
        self.thread.remote_frame_ready.connect(self.update_remote_image)
        self.thread.status_update.connect(self.update_status_label)
        self.thread.angle_update.connect(self.update_angle_display)
        self.thread.ptz_state_update.connect(self.update_ptz_state_display)
        self.thread.geo_result_signal.connect(self.update_geo_result)
        self.settings_window.params_updated.connect(self.thread.set_gps_coords)
        self.settings_window.emit_params()

        self.rx_thread = UdpReceiverThread()
        self.rx_thread.save_trigger.connect(self.thread.on_save_command)
        self.rx_thread.ack_received.connect(self.thread.on_ack_received)
        self.rx_thread.start()

        self.setup_ui()
        QApplication.instance().installEventFilter(self)
        self.thread.start()


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_ROOT = os.path.join(SCRIPT_DIR, "save")
FIRE_IMAGE_DIR = os.path.join(SAVE_ROOT, "image_fire")
DISTANCE_DIR = os.path.join(SAVE_ROOT, "distance")
os.makedirs(FIRE_IMAGE_DIR, exist_ok=True)
os.makedirs(DISTANCE_DIR, exist_ok=True)


class SaveBrowserDialog(QDialog):
    """在程序内浏览 save 文件夹中的图片与文本记录。"""

    SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("保存记录查看")
        self.resize(1050, 700)
        self.setMinimumSize(820, 560)
        self.original_pixmap = None

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(9)

        toolbar = QHBoxLayout()
        title = QLabel("本地保存记录")
        title.setStyleSheet("font-size:16px; font-weight:600; color:#FFFFFF;")
        self.record_count_label = QLabel("共 0 个文件")
        self.record_count_label.setStyleSheet("color:#82A9CF;")
        refresh_button = QPushButton("刷新")
        refresh_button.setFixedWidth(80)
        refresh_button.clicked.connect(self.refresh_files)
        toolbar.addWidget(title)
        toolbar.addSpacing(12)
        toolbar.addWidget(self.record_count_label)
        toolbar.addStretch()
        toolbar.addWidget(refresh_button)
        root_layout.addLayout(toolbar)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        file_panel = QWidget()
        file_layout = QVBoxLayout(file_panel)
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.setSpacing(6)
        file_tip = QLabel("图片与 TXT 文件（按保存时间倒序）")
        file_tip.setStyleSheet("color:#A9C7E5;")
        self.file_list = QListWidget()
        self.file_list.setAlternatingRowColors(True)
        self.file_list.currentItemChanged.connect(self.show_selected_file)
        file_layout.addWidget(file_tip)
        file_layout.addWidget(self.file_list, 1)
        splitter.addWidget(file_panel)

        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(8, 0, 0, 0)
        preview_layout.setSpacing(6)
        self.preview_title = QLabel("请选择左侧文件")
        self.preview_title.setWordWrap(True)
        self.preview_title.setStyleSheet(
            "color:#E6F0FF; font-size:13px; font-weight:600; padding:4px;"
        )
        self.preview_stack = QStackedWidget()

        self.placeholder_label = QLabel("暂无预览内容")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet(
            "color:#82A9CF; background:#061A34; border:1px solid #2A5688;"
        )
        self.preview_stack.addWidget(self.placeholder_label)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background:#020A13;")
        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(True)
        self.image_scroll.setAlignment(Qt.AlignCenter)
        self.image_scroll.setWidget(self.image_label)
        self.preview_stack.addWidget(self.image_scroll)

        self.text_preview = QPlainTextEdit()
        self.text_preview.setReadOnly(True)
        self.text_preview.setStyleSheet(
            "QPlainTextEdit {color:#DCEBFA; background:#061A34; "
            "border:1px solid #2A5688; padding:8px; font-size:13px;}"
        )
        self.preview_stack.addWidget(self.text_preview)

        preview_layout.addWidget(self.preview_title)
        preview_layout.addWidget(self.preview_stack, 1)
        splitter.addWidget(preview_panel)
        splitter.setSizes([330, 700])
        root_layout.addWidget(splitter, 1)

        self.setStyleSheet(
            "QDialog {background:#071A33; color:#E6F0FF;}"
            "QWidget {color:#E6F0FF;}"
            "QListWidget {color:#DCEBFA; background:#0D2948; "
            "border:1px solid #2A5688; "
            "border-radius:5px; padding:4px;}"
            "QListWidget::item:alternate {background:#102E50;}"
            "QListWidget::item {padding:7px 5px;}"
            "QListWidget::item:selected {color:#FFFFFF; background:#1976D2;}"
            "QPushButton {color:#FFFFFF; background:#285B9A; "
            "border:1px solid #3B72B6; border-radius:4px; padding:6px;}"
        )
        self.refresh_files()

    def refresh_files(self):
        current_path = None
        current_item = self.file_list.currentItem()
        if current_item is not None:
            current_path = current_item.data(Qt.UserRole)

        records = []
        if os.path.isdir(SAVE_ROOT):
            for folder, _, filenames in os.walk(SAVE_ROOT):
                for filename in filenames:
                    extension = os.path.splitext(filename)[1].lower()
                    if extension not in self.SUPPORTED_IMAGE_EXTENSIONS | {".txt"}:
                        continue
                    full_path = os.path.join(folder, filename)
                    try:
                        modified_time = os.path.getmtime(full_path)
                    except OSError:
                        continue
                    records.append((modified_time, full_path))
        records.sort(key=lambda record: record[0], reverse=True)

        self.file_list.blockSignals(True)
        self.file_list.clear()
        selected_item = None
        for modified_time, full_path in records:
            relative_path = os.path.relpath(full_path, SAVE_ROOT)
            time_text = datetime.fromtimestamp(modified_time).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            item = QListWidgetItem(f"{relative_path}\n{time_text}")
            item.setData(Qt.UserRole, full_path)
            item.setToolTip(full_path)
            self.file_list.addItem(item)
            if full_path == current_path:
                selected_item = item
        self.file_list.blockSignals(False)

        self.record_count_label.setText(f"共 {len(records)} 个文件")
        if selected_item is not None:
            self.file_list.setCurrentItem(selected_item)
        elif self.file_list.count() > 0:
            self.file_list.setCurrentRow(0)
        else:
            self.original_pixmap = None
            self.preview_title.setText("save 文件夹中暂无图片或 TXT 文件")
            self.preview_stack.setCurrentWidget(self.placeholder_label)

    def show_selected_file(self, current, _previous=None):
        if current is None:
            return
        full_path = current.data(Qt.UserRole)
        if not full_path or not os.path.isfile(full_path):
            self.preview_title.setText("文件不存在，请点击刷新")
            self.preview_stack.setCurrentWidget(self.placeholder_label)
            return

        relative_path = os.path.relpath(full_path, SAVE_ROOT)
        self.preview_title.setText(relative_path)
        extension = os.path.splitext(full_path)[1].lower()
        if extension in self.SUPPORTED_IMAGE_EXTENSIONS:
            pixmap = QPixmap(full_path)
            if pixmap.isNull():
                self.original_pixmap = None
                self.placeholder_label.setText("图片读取失败")
                self.preview_stack.setCurrentWidget(self.placeholder_label)
                return
            self.original_pixmap = pixmap
            self.preview_stack.setCurrentWidget(self.image_scroll)
            self._update_image_preview()
            return

        self.original_pixmap = None
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as file:
                content = file.read()
        except OSError as error:
            content = f"TXT 文件读取失败：{error}"
        self.text_preview.setPlainText(content)
        self.preview_stack.setCurrentWidget(self.text_preview)

    def _update_image_preview(self):
        if self.original_pixmap is None or self.original_pixmap.isNull():
            return
        viewport_size = self.image_scroll.viewport().size()
        width = max(240, viewport_size.width() - 8)
        height = max(200, viewport_size.height() - 8)
        self.image_label.setPixmap(
            self.original_pixmap.scaled(
                width,
                height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.preview_stack.currentWidget() is self.image_scroll:
            self._update_image_preview()


class UserStartedPtzBackend(StableNetworkPtzBackend):
    """只有用户点击“启动云台”后才建立控制和状态读取线程。"""

    def __init__(self, state_callback, status_callback):
        super().__init__(state_callback, status_callback)
        self._start_authorized = False
        self._start_lock = threading.Lock()

    def start(self):
        with self._start_lock:
            if not self._start_authorized or self._running:
                return False
            super().start()
            return True

    def start_by_user(self):
        self._start_authorized = True
        return self.start()

    def is_started(self):
        return bool(self._running)

    def move(self, pan, tilt, force=False):
        # 未点击启动云台时不缓存控制命令，避免启动后执行过期动作。
        if not self._running:
            return False
        super().move(pan, tilt, force=force)
        return True


class FireCameraThread(SmoothCameraThread):
    fire_alert = Signal(str)
    event_state_update = Signal(str, str)

    def __init__(self):
        super().__init__()
        # person 使用 yolo11n.pt；fire 使用 fire_weight.pt（0=fire，1=smoke）。
        # 模型只在视频工作线程内切换，避免点击界面时阻塞 GUI。
        self.model_lock = threading.RLock()
        self.loaded_model_label = None
        self.locked_class_id = None
        self.active_target_name = "person"
        # 替换父类后台：线程启动时的 ptz.start() 不再自动读取状态，
        # 必须由界面上的“启动云台”按钮显式授权。
        self.ptz = UserStartedPtzBackend(
            self._on_full_ptz_state, self._on_stable_ptz_status
        )
        self.fire_locked_frame = None
        self.fire_frame_lock = threading.Lock()

        self.search_interval_seconds = 30.0
        self.search_move_seconds = 5.0
        self.search_pan_speed = 20.0
        self.search_next_at = None
        self.search_until = None
        self.search_active = False
        self.search_stop_timer = None
        self.search_state_lock = threading.Lock()

        # UDP事件状态与视觉锁定状态分离。短暂丢帧或重新居中不会清除ACK。
        # 部分事件信号会立即刷新界面并再次读取事件快照，使用可重入锁
        # 避免同一线程在信号回调中重复取锁造成界面卡死。
        self.udp_event_lock = threading.RLock()
        self.udp_event_id = None
        self.udp_event_state = "IDLE"
        # 事件使用固定默认规则：连续无目标达到阈值后结束。
        self.udp_event_manual_hold = False
        self.udp_event_last_seen = None
        self.udp_event_last_send = 0.0
        self.udp_event_retry_count = 0
        self.udp_ack_receive_port = UDP_DEFAULT_RECEIVE_PORT
        self.event_role = "IDLE"  # IDLE / ORIGIN / RESPONDER
        self.event_target_name = ""
        self.last_local_detection_save = 0.0
        self.last_local_saved_target = ""
        self.last_peer_live_image_send = 0.0
        self.last_alert_key = None
        self.latest_detections = []
        # 本机作为局部坐标原点；对端位置由方位角和基线距离给出。
        # 角度统一为正东 0°、顺时针增加。
        self.peer_bearing_deg = 0.0
        self.peer_distance_m = 100.0

    def is_ptz_started(self):
        return self.ptz.is_started()

    def start_ptz(self):
        started = self.ptz.start_by_user()
        if started:
            self.status_update.emit("状态: 云台已启动，正在读取云台状态")
        return started

    @Slot(float, float)
    def set_relative_pose(self, peer_bearing_degrees, peer_distance_metres):
        bearing = float(peer_bearing_degrees) % 360.0
        distance = float(peer_distance_metres)
        if not math.isfinite(distance) or distance <= 0.0:
            raise ValueError("两设备间距离必须大于 0")
        self.peer_bearing_deg = bearing
        self.peer_distance_m = distance

    def set_ack_receive_port(self, port):
        self.udp_ack_receive_port = int(port)

    @Slot(float, float)
    def set_search_timing(self, interval_seconds, move_seconds):
        """更新无目标自动搜寻周期，并从下一轮按新参数计时。"""
        interval = max(1.0, min(3600.0, float(interval_seconds)))
        duration = max(1.0, min(600.0, float(move_seconds)))
        # 如果正在搜寻，先安全停止当前轮；下一帧无目标
        # 画面会使用新间隔重新计时，避免旧定时器继续生效。
        self._reset_auto_search(stop_motion=True)
        with self.search_state_lock:
            self.search_interval_seconds = interval
            self.search_move_seconds = duration
        self.status_update.emit(
            "状态: 自动搜寻参数已更新 "
            f"间隔={interval:g}秒，搜寻={duration:g}秒"
        )

    def get_event_snapshot(self):
        """供界面定时读取当前事件号和状态。"""
        with self.udp_event_lock:
            event_id = self.udp_event_id or ""
            state = self.udp_event_state
            manual_hold = self.udp_event_manual_hold
            role = self.event_role
            target_name = self.event_target_name
        return {
            "event_id": event_id,
            "state": state,
            "manual_hold": manual_hold,
            "role": role,
            "target": target_name,
        }

    @Slot()
    def manual_end_event(self):
        with self.udp_event_lock:
            has_event = self.udp_event_id is not None
        self._end_udp_event("用户手动结束", manual_hold=True)
        if not has_event:
            self.status_update.emit("状态: 当前没有活动事件，已暂停自动建事件")

    def _release_manual_event_hold_if_target_gone(self):
        """手动结束后，等当前目标离开画面再恢复自动建事件。"""
        released = False
        with self.udp_event_lock:
            if self.udp_event_id is None and self.udp_event_manual_hold:
                self.udp_event_manual_hold = False
                released = True
        if released:
            self.event_state_update.emit("", "IDLE")
            self.status_update.emit("状态: 当前目标已离开，可自动建立下一事件")

    @Slot(str, float, float, str, str, int)
    def on_peer_event_started(
        self, event_id, azimuth, elevation, target_name, sender_ip, reply_port
    ):
        """接收对端事件；本事件内本机动态成为响应方。"""
        event_id = str(event_id).strip()
        if not event_id:
            return

        target_name = str(target_name).strip().lower()
        if target_name not in ("person", "fire", "smoke"):
            target_name = "person" if self.target_label == "person" else "fire"

        with self.udp_event_lock:
            current_id = self.udp_event_id
            current_role = self.event_role
            if current_id == event_id:
                self.udp_event_last_seen = time.monotonic()
                return
            if current_role == "RESPONDER" and current_id:
                # 一个事件结束前响应方身份保持不变，不被另一事件抢占。
                return
            if current_role == "ORIGIN" and current_id and current_id < event_id:
                # 两端近乎同时发现目标时，事件号较小的一方保持发起方，
                # 另一端收到该事件后会切换为响应方。
                self.status_update.emit(
                    f"状态: 同时事件冲突，保留本机事件 {current_id}"
                )
                return

        if current_role == "ORIGIN" and current_id:
            self._end_udp_event("同时事件合并", notify_peer=False)

        desired_mode = "person" if target_name == "person" else "fire"
        if desired_mode != self.target_label:
            # 切换到对端事件的检测模型，但不触发 FireCameraThread.set_target()
            # 中的事件结束逻辑。
            CameraThread.set_target(self, desired_mode)
            self.locked_class_id = None
            self.previous_target_center = None
            self._reset_motion_filter()

        now = time.monotonic()
        with self.udp_event_lock:
            self.udp_event_id = event_id
            self.udp_event_state = "REMOTE_ACTIVE"
            self.udp_event_manual_hold = False
            self.udp_event_last_seen = now
            self.udp_event_last_send = 0.0
            self.udp_event_retry_count = 0
            self.event_role = "RESPONDER"
            self.event_target_name = target_name
            self.is_acked = True
            self.last_local_detection_save = 0.0
            self.last_local_saved_target = ""
            self.last_peer_live_image_send = 0.0
            self.last_alert_key = None

        self.target_ip = str(sender_ip)
        self.target_port = int(reply_port)
        self.start_ptz()
        self.video_active = True
        self.tracking_active = True
        self._reset_auto_search(stop_motion=True)
        if self.is_ptz_started():
            with self.search_state_lock:
                self.search_active = True
                self.search_next_at = None
                self.search_until = None
            self.ptz.move(-self.search_pan_speed, 0, force=True)

        self.event_state_update.emit(event_id, "REMOTE_ACTIVE")
        self.status_update.emit(
            f"状态: 收到对端事件 {event_id}，本机作为响应方持续搜寻 {target_name}"
        )

    @Slot(str)
    def on_peer_event_ended(self, event_id):
        with self.udp_event_lock:
            matches = (
                self.event_role == "RESPONDER"
                and self.udp_event_id == str(event_id)
            )
        if matches:
            self._end_udp_event("对端通知事件结束", notify_peer=False)

    def _create_udp_event(self, initial_state="WAIT_ACK", target_name=None):
        event_id = (
            f"{DEVICE_ID}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_"
            f"{uuid.uuid4().hex[:8]}"
        )
        now = time.monotonic()
        target_name = str(target_name or self.active_target_name).strip().lower()
        if target_name not in ("person", "fire", "smoke"):
            target_name = "person" if self.target_label == "person" else "fire"
        with self.udp_event_lock:
            self.udp_event_id = event_id
            self.udp_event_state = initial_state
            self.udp_event_manual_hold = False
            self.udp_event_last_seen = now
            self.udp_event_last_send = 0.0
            self.udp_event_retry_count = 0
            self.event_role = "ORIGIN"
            self.event_target_name = target_name
            self.last_local_detection_save = 0.0
            self.last_local_saved_target = ""
            self.last_peer_live_image_send = 0.0
            self.last_alert_key = None
            self.is_acked = False
        self.event_state_update.emit(event_id, initial_state)
        self.status_update.emit(
            f"状态: 新建 {target_name} 事件 {event_id}，本机为发起方"
        )
        return event_id

    def _send_event_end(self, event_id, target_name):
        payload = {
            "type": "event_end",
            "device_id": DEVICE_ID,
            "event_id": event_id,
            "message_id": f"{event_id}_END",
            "reply_port": self.udp_ack_receive_port,
            "target": target_name,
            "angle_reference": "east_zero_clockwise",
            "az": round(self.current_az, 2),
            "el": round(self.current_el, 2),
            "timestamp": datetime.now().isoformat(
                sep=" ", timespec="milliseconds"
            ),
        }
        packet = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        for _ in range(3):
            try:
                self.send_sock.sendto(packet, (self.target_ip, self.target_port))
            except OSError:
                break

    def _end_udp_event(
        self, reason, manual_hold=False, notify_peer=True
    ):
        with self.udp_event_lock:
            event_id = self.udp_event_id
            event_role = self.event_role
            target_name = self.event_target_name
            self.udp_event_manual_hold = bool(manual_hold)
            if event_id is None:
                self.event_state_update.emit("", "IDLE")
                return
            self.udp_event_id = None
            self.udp_event_state = "IDLE"
            self.udp_event_last_seen = None
            self.udp_event_last_send = 0.0
            self.udp_event_retry_count = 0
            self.event_role = "IDLE"
            self.event_target_name = ""
            self.last_local_detection_save = 0.0
            self.last_local_saved_target = ""
            self.last_peer_live_image_send = 0.0
            self.last_alert_key = None
            self.is_acked = False
        self._reset_auto_search(stop_motion=True)
        if notify_peer and event_role in ("ORIGIN", "RESPONDER"):
            self._send_event_end(event_id, target_name)
        self.event_state_update.emit("", "IDLE")
        self.status_update.emit(f"状态: 目标事件已结束（{reason}）")

    def _mark_udp_event_seen(self, now):
        with self.udp_event_lock:
            if self.udp_event_id is not None:
                self.udp_event_last_seen = now
                self.is_acked = self.udp_event_state in (
                    "ACKED", "REMOTE_ACTIVE"
                )

    def _check_udp_event_missing(self, now):
        should_end = False
        with self.udp_event_lock:
            if (
                self.event_role == "ORIGIN"
                and self.udp_event_id is not None
                and self.udp_event_last_seen is not None
                and now - self.udp_event_last_seen
                >= UDP_EVENT_MISSING_TIMEOUT_SECONDS
            ):
                should_end = True
        if should_end:
            self._end_udp_event(
                f"连续 {UDP_EVENT_MISSING_TIMEOUT_SECONDS:.0f} 秒未检测到目标"
            )

    def _send_udp_event_if_needed(self, target_id, target_name=None):
        now = time.monotonic()
        with self.udp_event_lock:
            if self.event_role == "RESPONDER":
                return "RESPONDER"
            if self.udp_event_id is None:
                event_id = None
            else:
                event_id = self.udp_event_id

        if event_id is None:
            with self.udp_event_lock:
                if self.udp_event_manual_hold:
                    return "MANUAL_HOLD"
            event_id = self._create_udp_event(target_name=target_name)

        with self.udp_event_lock:
            if self.udp_event_state == "READY":
                self.udp_event_state = "WAIT_ACK"
                self.event_state_update.emit(event_id, "WAIT_ACK")
            if self.udp_event_state == "ACKED":
                return "ACKED"
            if self.udp_event_state != "WAIT_ACK":
                return self.udp_event_state
            if self.udp_event_retry_count >= UDP_MAX_RETRIES:
                return "RETRY_LIMIT"
            if now - self.udp_event_last_send < UDP_RETRY_INTERVAL_SECONDS:
                return "WAIT_ACK"

            self.udp_event_retry_count += 1
            self.udp_event_last_send = now
            retry_count = self.udp_event_retry_count
            event_id = self.udp_event_id
            event_target = self.event_target_name

        payload = {
            "type": "locked_target",
            "device_id": DEVICE_ID,
            "event_id": event_id,
            "message_id": f"{event_id}_LOCK",
            "reply_port": self.udp_ack_receive_port,
            "retry": retry_count,
            "id": target_id,
            "target": event_target,
            "angle_reference": "east_zero_clockwise",
            "az": round(self.current_az, 2),
            "el": round(self.current_el, 2),
            "timestamp": datetime.now().isoformat(sep=" ", timespec="milliseconds"),
        }
        try:
            self.send_sock.sendto(
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                (self.target_ip, self.target_port),
            )
            self.status_update.emit(
                f"状态: 发送目标事件 {event_id}（第 {retry_count} 次）"
            )
        except OSError as error:
            self.status_update.emit(f"状态: UDP 发送失败 - {error}")
        return "SENT"

    @Slot(str)
    def on_ack_received(self, event_id):
        with self.udp_event_lock:
            if (
                not event_id
                or self.udp_event_id is None
                or event_id != self.udp_event_id
                or self.udp_event_state != "WAIT_ACK"
            ):
                return
            self.udp_event_state = "ACKED"
            self.is_acked = True
            confirmed_event = self.udp_event_id
        self.event_state_update.emit(confirmed_event, "ACKED")
        self.status_update.emit(
            f"状态: 对方已确认事件 {confirmed_event}，后续重新锁定不再发送"
        )

    def _reset_auto_search(self, stop_motion=True, announce=False):
        with self.search_state_lock:
            was_searching = self.search_active
            stop_timer = self.search_stop_timer
            if stop_motion and was_searching and self.is_ptz_started():
                self.ptz.move(0, 0, force=True)
            self.search_active = False
            self.search_next_at = None
            self.search_until = None
            self.search_stop_timer = None
        if stop_timer is not None:
            stop_timer.cancel()
        if announce and was_searching:
            self.status_update.emit("状态: 检测到目标，自动搜寻已停止")

    def _finish_auto_search(self, finished_at=None):
        with self.search_state_lock:
            if not self.search_active:
                return
            stop_timer = self.search_stop_timer
            if self.is_ptz_started():
                self.ptz.move(0, 0, force=True)
            self.search_active = False
            self.search_until = None
            self.search_stop_timer = None
            cycle_end = time.monotonic() if finished_at is None else finished_at
            self.search_next_at = cycle_end + self.search_interval_seconds
        if stop_timer is not None and stop_timer is not threading.current_thread():
            stop_timer.cancel()
        self.status_update.emit("状态: 自动向左搜寻完成，云台已停止")

    def _handle_no_target_search(self, now):
        """无目标后按当前自定义间隔和时长向左搜寻。"""
        if not self.is_ptz_started():
            return

        with self.udp_event_lock:
            responder_mode = self.event_role == "RESPONDER"

        if responder_mode:
            # 响应方在发起方结束事件前持续搜寻；检测到目标时由
            # _reset_auto_search() 暂停，目标丢失后立即继续。
            with self.search_state_lock:
                already_searching = self.search_active
            if not already_searching:
                self._send_smooth_command(0, 0, immediate_stop=True)
                with self.search_state_lock:
                    self.search_active = True
                    self.search_next_at = None
                    self.search_until = None
                self.ptz.move(-self.search_pan_speed, 0, force=True)
                self.status_update.emit("状态: 响应方正在持续向左搜寻目标")
            return

        with self.search_state_lock:
            search_active = self.search_active
            search_until = self.search_until
            search_next_at = self.search_next_at

        if search_active:
            if now >= search_until:
                self._finish_auto_search(finished_at=now)
            return

        if search_next_at is None:
            # 刚刚丢失目标时先停止原跟踪命令，再按当前间隔计时。
            self._send_smooth_command(0, 0, immediate_stop=True)
            with self.search_state_lock:
                self.search_next_at = now + self.search_interval_seconds
            return

        if now >= search_next_at:
            self._send_smooth_command(0, 0, immediate_stop=True)
            stop_timer = threading.Timer(
                self.search_move_seconds, self._finish_auto_search
            )
            stop_timer.daemon = True
            with self.search_state_lock:
                self.search_active = True
                self.search_until = now + self.search_move_seconds
                self.search_stop_timer = stop_timer
                self.ptz.move(-self.search_pan_speed, 0, force=True)
                stop_timer.start()
            self.status_update.emit(
                "状态: 未检测到目标，云台自动向左搜寻 "
                f"{self.search_move_seconds:g} 秒"
            )

    def _clear_fire_locked_frame(self):
        with self.fire_frame_lock:
            self.fire_locked_frame = None

    def set_tracking_state(self, active):
        self._reset_auto_search(stop_motion=True)
        self._clear_fire_locked_frame()
        if active:
            with self.udp_event_lock:
                self.udp_event_manual_hold = False
        if not active:
            self._end_udp_event("手动停止跟踪")
        super().set_tracking_state(active)

    def set_video_state(self, active):
        if not active:
            self._reset_auto_search(stop_motion=True)
            self._end_udp_event("视频流停止")
        super().set_video_state(active)

    def stop(self):
        self._reset_auto_search(stop_motion=True)
        self._end_udp_event("程序退出")
        return super().stop()

    def set_target(self, label):
        new_label = label if label in ("person", "fire") else "person"
        if new_label == self.target_label:
            return
        self._clear_fire_locked_frame()
        self._end_udp_event("检测类别切换")
        super().set_target(new_label)
        self.locked_class_id = None
        self.active_target_name = new_label

    def _ensure_target_model(self):
        """按当前选择加载对应模型；该方法只从视频工作线程调用。"""
        if YOLO is None:
            self.status_update.emit("状态: 缺少 ultralytics，无法启动跟踪")
            return False

        desired_label = self.target_label if self.target_label in (
            "person", "fire"
        ) else "person"
        with self.model_lock:
            if self.model is not None and self.loaded_model_label == desired_label:
                return True

            model_path = (
                FIRE_TRACK_MODEL_PATH
                if desired_label == "fire"
                else PERSON_MODEL_PATH
            )
            if not os.path.exists(model_path):
                self.status_update.emit(
                    f"状态: 未找到{desired_label}模型 - {model_path}"
                )
                return False
            try:
                self.status_update.emit(
                    f"状态: 正在加载 {os.path.basename(model_path)}..."
                )
                self.model = YOLO(model_path)
                self.loaded_model_label = desired_label
            except Exception as error:
                self.model = None
                self.loaded_model_label = None
                self.status_update.emit(f"状态: 模型加载失败 - {error}")
                return False

        self.previous_target_center = None
        self.locked_id = None
        self.locked_class_id = None
        self.active_target_name = desired_label
        self.missing_target_frames = 0
        self._reset_motion_filter()
        self.smooth_center_locked = False
        self.smooth_centered_since = None
        self.smooth_outside_count = 0
        self.status_update.emit(
            f"状态: 已切换为 {os.path.basename(model_path)}"
        )
        return True

    def _select_detection(self, result, frame_width, frame_height):
        """选择追踪目标并保留全部检测框供界面显示。

        fire 模式：只有 fire 跟踪 fire，只有 smoke 跟踪 smoke；两者同时
        出现时跟踪 fire。同一优先类别存在多个框时始终跟踪面积最大的。
        """
        acquiring_new_target = self.previous_target_center is None
        if result.boxes is None or len(result.boxes) == 0:
            self.latest_detections = []
            self.missing_target_frames += 1
            if self.missing_target_frames >= 5:
                self.previous_target_center = None
                self.locked_id = None
            with self.udp_event_lock:
                self.is_acked = self.udp_event_state in (
                    "ACKED", "REMOTE_ACTIVE"
                )
            return None

        boxes = result.boxes.xyxy.cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()
        try:
            confidences = result.boxes.conf.cpu().tolist()
        except AttributeError:
            confidences = [0.0] * len(boxes)
        active_label = self.loaded_model_label or self.target_label
        if active_label == "fire":
            if 0 in classes:
                preferred_class = 0
                target_name = "fire"
            elif 1 in classes:
                preferred_class = 1
                target_name = "smoke"
            else:
                preferred_class = None
                target_name = "fire"
        else:
            preferred_class = 0 if 0 in classes else None
            target_name = "person"

        if preferred_class is None:
            self.latest_detections = []
            self.missing_target_frames += 1
            with self.udp_event_lock:
                self.is_acked = self.udp_event_state in (
                    "ACKED", "REMOTE_ACTIVE"
                )
            return None

        if self.locked_class_id != preferred_class:
            previous_class = self.locked_class_id
            self.previous_target_center = None
            self.locked_id = None
            self.locked_class_id = preferred_class
            if previous_class is not None:
                self._clear_fire_locked_frame()
            self._reset_motion_filter()
            self.smooth_center_locked = False
            self.smooth_centered_since = None
            self.smooth_outside_count = 0

        self.active_target_name = target_name
        targets = []
        all_detections = []
        for detection_id, (coords, class_id, confidence) in enumerate(
            zip(boxes, classes, confidences), start=1
        ):
            if active_label == "fire":
                if class_id not in (0, 1):
                    continue
                detection_name = "fire" if class_id == 0 else "smoke"
            else:
                if class_id != 0:
                    continue
                detection_name = "person"

            x1, y1, x2, y2 = map(int, coords)
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            detection = {
                "id": detection_id,
                "class_id": class_id,
                "name": detection_name,
                "confidence": float(confidence),
                "box": (x1, y1, x2, y2),
                "center": center,
                "area": max(0, x2 - x1) * max(0, y2 - y1),
            }
            all_detections.append(detection)
            if class_id != preferred_class:
                continue
            targets.append(detection)

        self.latest_detections = all_detections

        if not targets:
            self.missing_target_frames += 1
            return None

        new_lock = self.previous_target_center is None
        if active_label == "fire" or new_lock:
            # 火情目标始终按面积选取，不因历史中心继续追踪较小目标。
            selected = max(targets, key=lambda target: target["area"])
        else:
            previous_x, previous_y = self.previous_target_center
            selected = min(
                targets,
                key=lambda target: (
                    (target["center"][0] - previous_x) ** 2
                    + (target["center"][1] - previous_y) ** 2
                ),
            )
            distance = (
                (selected["center"][0] - previous_x) ** 2
                + (selected["center"][1] - previous_y) ** 2
            ) ** 0.5
            association_limit = 0.25 * (
                frame_width**2 + frame_height**2
            ) ** 0.5
            if distance > association_limit:
                selected = max(targets, key=lambda target: target["area"])

        self.previous_target_center = selected["center"]
        self.missing_target_frames = 0
        if new_lock:
            self.locked_id = selected["id"]
            self.temp_local_path = None
            self.memory_az = None

        with self.udp_event_lock:
            self.is_acked = self.udp_event_state in (
                "ACKED", "REMOTE_ACTIVE"
            )
            if self.udp_event_id is not None:
                self.event_target_name = target_name
        if selected is not None and acquiring_new_target:
            self._clear_fire_locked_frame()
        return selected

    def _archive_stable_lock(self, frame, target_id, x1, y1):
        with self.fire_frame_lock:
            if self.fire_locked_frame is None:
                self.fire_locked_frame = frame.copy()

        self.memory_az = self.current_az
        if self.temp_local_path is None:
            timestamp = datetime.now().strftime("%H-%M-%S-%f")[:-3]
            filename = os.path.join(
                IMG_SAVE_DIR, f"img_{timestamp}_LOCAL_STABLE.jpg"
            )
            if cv2.imwrite(filename, frame):
                self.temp_local_path = filename

        event_status = self._send_udp_event_if_needed(
            target_id, self.active_target_name
        )
        if event_status == "ACKED":
            self._draw_text(frame, "ACK OK", x1, y1 - 20, (255, 0, 0))
        elif event_status == "RETRY_LIMIT":
            self._draw_text(frame, "ACK TIMEOUT", x1, y1 - 20, (0, 165, 255))
        elif event_status == "MANUAL_HOLD":
            self._draw_text(frame, "EVENT ENDED", x1, y1 - 20, (0, 165, 255))
        elif event_status == "RESPONDER":
            self._draw_text(frame, "PEER EVENT", x1, y1 - 20, (255, 170, 0))
        else:
            self._draw_text(frame, "WAIT ACK", x1, y1 - 20, (0, 0, 255))

    @staticmethod
    def _draw_detection_box(frame, target, locked):
        x1, y1, x2, y2 = target["box"]
        color = (0, 0, 255) if locked else (0, 255, 0)
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color=color,
            thickness=4,
            lineType=cv2.LINE_AA,
        )

    def _draw_other_detections(self, frame, selected):
        """显示全部检测结果，只有选中的最大目标用于云台追踪。"""
        for detection in self.latest_detections:
            if (
                detection["class_id"] == selected["class_id"]
                and detection["box"] == selected["box"]
            ):
                continue
            x1, y1, x2, y2 = detection["box"]
            color = (
                (0, 165, 255)
                if detection["name"] == "fire"
                else (0, 215, 255)
                if detection["name"] == "smoke"
                else (255, 170, 0)
            )
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2,
                cv2.LINE_AA,
            )
            self._draw_text(
                frame,
                f"{detection['name']} {detection['confidence']:.2f}",
                x1,
                y1 - 8,
                color,
            )

    def _save_local_detection_if_due(self, frame, target, now):
        target_name = target.get("name", self.active_target_name)
        with self.udp_event_lock:
            current_event_id = self.udp_event_id
            event_role = self.event_role
            manual_hold = self.udp_event_manual_hold
        if current_event_id is None and manual_hold:
            return
        event_id = current_event_id or "NO_EVENT"

        alert_key = (event_id, target_name)
        if self.last_alert_key != alert_key:
            self.last_alert_key = alert_key
            self.fire_alert.emit(f"发现{target_name}！")

        save_due = (
            target_name != self.last_local_saved_target
            or now - self.last_local_detection_save
            >= LOCAL_DETECTION_SAVE_INTERVAL_SECONDS
        )
        if not save_due:
            return

        file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_event = "".join(
            char if char.isalnum() or char in "-_" else "_"
            for char in event_id
        )[-72:]
        image_path = os.path.join(
            FIRE_IMAGE_DIR,
            f"{DEVICE_ID}_{target_name}_{safe_event}_{file_timestamp}.jpg",
        )
        if cv2.imwrite(image_path, frame):
            self.last_local_detection_save = now
            self.last_local_saved_target = target_name
            self.status_update.emit(
                f"状态: 已保存{target_name}检测图像（{event_role}）- "
                f"{os.path.basename(image_path)}"
            )

    @staticmethod
    def _encode_peer_frame(frame):
        """压缩为单个 UDP 数据报可承载的实时预览图。"""
        height, width = frame.shape[:2]
        max_width, max_height = 640, 360
        scale = min(1.0, max_width / width, max_height / height)
        preview = frame
        if scale < 1.0:
            preview = cv2.resize(
                frame,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        for quality in (70, 55, 40):
            success, encoded = cv2.imencode(
                ".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, quality]
            )
            if success and len(encoded) <= 45_000:
                return base64.b64encode(encoded.tobytes()).decode("ascii")
        return ""

    def _send_peer_live_image_if_due(self, frame, target, now):
        with self.udp_event_lock:
            # 首次事件通知仍需稳定锁定；事件建立后双方都按固定周期
            # 回传当前画面，搜寻中或暂时丢失目标时也不停止传图。
            if (
                self.event_role not in ("ORIGIN", "RESPONDER")
                or not self.udp_event_id
            ):
                return
            if (
                now - self.last_peer_live_image_send
                < PEER_LIVE_IMAGE_INTERVAL_SECONDS
            ):
                return
            event_id = self.udp_event_id
            event_target = self.event_target_name
            self.last_peer_live_image_send = now

        image_b64 = self._encode_peer_frame(frame)
        if not image_b64:
            self.status_update.emit("状态: 对端实时图像压缩失败")
            return

        current_target = (
            target.get("name", event_target) if target is not None else event_target
        )
        localization_ready = target is not None and self.memory_az is not None
        localization_az = self.memory_az if localization_ready else self.current_az
        message_time = datetime.now()
        payload = {
            "type": "peer_live_image",
            "device_id": DEVICE_ID,
            "event_id": event_id,
            "message_id": (
                f"{event_id}_LIVE_{message_time.strftime('%Y%m%d_%H%M%S_%f')}"
            ),
            "reply_port": self.udp_ack_receive_port,
            "target": current_target,
            "angle_reference": "east_zero_clockwise",
            "az": round(localization_az, 2),
            "el": round(self.current_el, 2),
            "localization_ready": localization_ready,
            "timestamp": message_time.isoformat(
                sep=" ", timespec="milliseconds"
            ),
            "image": image_b64,
        }
        packet = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if len(packet) > 65_000:
            self.status_update.emit("状态: 对端实时图像数据过大，本次未发送")
            return
        try:
            self.send_sock.sendto(packet, (self.target_ip, self.target_port))
            self.status_update.emit(
                f"状态: 已向对端发送实时图像 {event_id}"
            )
        except OSError as error:
            self.status_update.emit(f"状态: 实时图像发送失败 - {error}")

    @Slot(str, str, float, float, str, str, bool)
    def on_peer_image(
        self,
        event_id,
        timestamp,
        az_remote,
        el_remote,
        image_b64,
        target_name,
        remote_pose_available,
    ):
        """接收对端画面，并使用双方方位角和已知基线完成定位。"""
        try:
            image_bytes = base64.b64decode(image_b64)
            remote_image = cv2.imdecode(
                np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR
            )
        except (ValueError, TypeError):
            remote_image = None
        if remote_image is None:
            self.status_update.emit("状态: 对端实时图像解码失败")
            return

        self.remote_frame_ready.emit(self._to_qimage(remote_image))
        if not remote_pose_available:
            self.status_update.emit(
                f"状态: 收到对端事件 {event_id} 的实时图像，"
                "但缺少正东零位、顺时针角度标识或对端方位角，暂时无法计算"
            )
            return
        result = self._calculate_fire_result(float(az_remote))
        self.geo_result_signal.emit(
            str(result["bearing_deg"]),
            str(result["position_xy"]),
            str(result["distance_m"]),
        )
        display_target = str(target_name).strip() or "目标"
        self.status_update.emit(
            f"状态: 收到对端事件 {event_id} 的实时图像（{display_target}）"
        )

    def _finalize_detected_frame(self, frame, target, now):
        self._save_local_detection_if_due(frame, target, now)
        self._send_peer_live_image_if_due(frame, target, now)
        return frame

    def _track_frame(self, frame):
        """沿用流畅版控制逻辑，只改变搜索与锁定框的颜色和粗细。"""
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        cv2.line(frame, (center_x, 0), (center_x, height), (180, 180, 180), 1)
        cv2.line(frame, (0, center_y), (width, center_y), (180, 180, 180), 1)
        cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

        if not self.tracking_active:
            return frame
        if not self._ensure_target_model():
            return frame

        try:
            active_label = self.loaded_model_label or self.target_label
            detect_classes = [0, 1] if active_label == "fire" else [0]
            with self.model_lock:
                result = self.model.predict(
                    frame,
                    classes=detect_classes,
                    verbose=False,
                    conf=0.25,
                )[0]
        except Exception as error:
            self.status_update.emit(f"状态: 目标检测失败 - {error}")
            self._reset_auto_search(stop_motion=True)
            self._send_smooth_command(0, 0, immediate_stop=True)
            return frame

        target = self._select_detection(result, width, height)
        if target is None:
            missing_now = time.monotonic()
            self._handle_no_target_search(missing_now)
            self._check_udp_event_missing(missing_now)
            self._send_peer_live_image_if_due(frame, None, missing_now)
            if self.missing_target_frames >= 5:
                self._release_manual_event_hold_if_target_gone()
                self._reset_motion_filter()
                self.smooth_center_locked = False
                self.smooth_centered_since = None
            return frame

        now = time.monotonic()
        # 首次事件通知由 _archive_stable_lock() 在目标稳定居中后发送；
        # 事件建立后的实时图像不再要求目标保持锁定。
        self._mark_udp_event_seen(now)
        self._reset_auto_search(stop_motion=True, announce=True)
        self._draw_other_detections(frame, target)

        x1, y1, x2, y2 = target["box"]
        target_caption = (
            f"{target.get('name', self.active_target_name)} "
            f"{target.get('confidence', 0.0):.2f} ID:{target['id']}"
        )
        raw_x = (x1 + x2) / 2.0
        raw_y = (y1 + y2) / 2.0
        filtered_x, filtered_y, predicted_x, predicted_y = self._update_target_filter(
            raw_x, raw_y, now
        )
        lock_dx = filtered_x - center_x
        lock_dy = filtered_y - center_y
        control_dx = predicted_x - center_x
        control_dy = predicted_y - center_y

        inside_enter = (
            abs(lock_dx) <= self.smooth_enter_zone
            and abs(lock_dy) <= self.smooth_enter_zone
        )
        inside_hold = (
            abs(lock_dx) <= self.smooth_hold_zone
            and abs(lock_dy) <= self.smooth_hold_zone
        )

        if inside_enter:
            self.smooth_outside_count = 0
            if self.smooth_centered_since is None:
                self.smooth_centered_since = now
            self._send_smooth_command(0, 0, immediate_stop=True)
            if now - self.smooth_centered_since >= self.smooth_lock_confirm_seconds:
                self.smooth_center_locked = True
                self._draw_detection_box(frame, target, locked=True)
                self._draw_text(frame, target_caption, x1, y1 - 38, (0, 0, 255))
                self._draw_text(frame, "SMOOTH LOCK", x1, y1 - 10, (0, 0, 255))
                self._archive_stable_lock(frame, target["id"], x1, y1)
            else:
                self._draw_detection_box(frame, target, locked=False)
                self._draw_text(frame, target_caption, x1, y1 - 38, (0, 190, 0))
                self._draw_text(frame, "LOCKING", x1, y1 - 10, (0, 190, 0))
            return self._finalize_detected_frame(frame, target, now)

        self.smooth_centered_since = None
        if self.smooth_center_locked and inside_hold:
            self.smooth_outside_count = 0
            self._send_smooth_command(0, 0, immediate_stop=True)
            self._draw_detection_box(frame, target, locked=True)
            self._draw_text(frame, target_caption, x1, y1 - 38, (0, 0, 255))
            self._draw_text(frame, "LOCK HOLD", x1, y1 - 10, (0, 0, 255))
            self._archive_stable_lock(frame, target["id"], x1, y1)
            return self._finalize_detected_frame(frame, target, now)

        if self.smooth_center_locked:
            self.smooth_outside_count += 1
            if self.smooth_outside_count < self.smooth_unlock_confirm_frames:
                self._send_smooth_command(0, 0, immediate_stop=True)
                self._draw_detection_box(frame, target, locked=True)
                self._draw_text(frame, target_caption, x1, y1 - 38, (0, 0, 255))
                self._draw_text(frame, "HOLD CHECK", x1, y1 - 10, (0, 0, 255))
                return self._finalize_detected_frame(frame, target, now)
            self.smooth_center_locked = False
            self.smooth_outside_count = 0

        desired_pan = self._axis_speed(control_dx)
        desired_tilt = -self._axis_speed(control_dy)
        self._send_smooth_command(desired_pan, desired_tilt)
        self._draw_detection_box(frame, target, locked=False)
        self._draw_text(frame, target_caption, x1, y1 - 38, (0, 190, 0))
        self._draw_text(
            frame,
            f"SMOOTH P:{int(desired_pan)} T:{int(desired_tilt)}",
            x1,
            y1 - 10,
            (0, 190, 0),
        )
        return self._finalize_detected_frame(frame, target, now)

    def _calculate_fire_result(self, az_remote):
        """以本机为原点，用双方正东零位、顺时针方位角进行射线交会。"""
        local_azimuth = (
            self.memory_az if self.memory_az is not None else self.current_az
        )
        try:
            solved = RelativeBearingCalculator.solve(
                self.peer_bearing_deg,
                self.peer_distance_m,
                local_azimuth,
                az_remote,
            )
            return {
                "bearing_deg": f"{solved['bearing_deg']:.2f}°",
                "position_xy": (
                    f"东 {solved['east_m']:.1f} m，"
                    f"南 {solved['south_m']:.1f} m"
                ),
                "east_m": f"{solved['east_m']:.3f}",
                "south_m": f"{solved['south_m']:.3f}",
                "distance_m": f"{solved['distance_m']:.1f}",
                "peer_distance_m": f"{solved['peer_distance_m']:.1f}",
                "local_azimuth": f"{float(local_azimuth) % 360.0:.3f}",
                "remote_azimuth": f"{float(az_remote) % 360.0:.3f}",
                "status": "计算成功",
            }
        except (ArithmeticError, ValueError, OverflowError) as error:
            return {
                "bearing_deg": "无法计算",
                "position_xy": "无法计算",
                "east_m": "无法计算",
                "south_m": "无法计算",
                "distance_m": "无法计算",
                "peer_distance_m": "无法计算",
                "local_azimuth": f"{float(local_azimuth) % 360.0:.3f}",
                "remote_azimuth": f"{float(az_remote) % 360.0:.3f}",
                "status": str(error),
            }

    def _save_fire_event(
        self, timestamp, az_remote, el_remote, remote_img_b64
    ):
        if not remote_img_b64:
            return False

        try:
            image_bytes = base64.b64decode(remote_img_b64)
            remote_image = cv2.imdecode(
                np.frombuffer(image_bytes, np.uint8),
                cv2.IMREAD_COLOR,
            )
        except (ValueError, TypeError):
            return False
        if remote_image is None:
            return False
        self.remote_frame_ready.emit(self._to_qimage(remote_image))

        file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        host_path = os.path.join(
            FIRE_IMAGE_DIR, f"host{file_timestamp}.jpg"
        )
        friend_path = os.path.join(
            FIRE_IMAGE_DIR, f"f{file_timestamp}.jpg"
        )
        distance_path = os.path.join(
            DISTANCE_DIR, f"distance_{file_timestamp}.txt"
        )

        with self.fire_frame_lock:
            host_image = (
                None
                if self.fire_locked_frame is None
                else self.fire_locked_frame.copy()
            )
        if host_image is None:
            with self.frame_lock:
                if self.current_frame is not None:
                    host_image = self.current_frame.copy()

        host_saved = bool(
            host_image is not None and cv2.imwrite(host_path, host_image)
        )
        friend_saved = bool(cv2.imwrite(friend_path, remote_image))
        result = self._calculate_fire_result(az_remote)
        self.geo_result_signal.emit(
            str(result["bearing_deg"]),
            str(result["position_xy"]),
            str(result["distance_m"]),
        )

        with open(distance_path, "w", encoding="utf-8") as file:
            file.write(f"事件时间戳: {file_timestamp}\n")
            file.write(f"对端消息时间: {timestamp}\n")
            file.write("角度基准: 正东0度，顺时针增加\n")
            file.write("本机局部坐标: 东0.000 m，南0.000 m\n")
            file.write(f"对端相对方位角: {self.peer_bearing_deg:.3f}°\n")
            file.write(f"两设备间距离: {self.peer_distance_m:.3f} m\n")
            file.write(f"本机观测方位角: {result['local_azimuth']}°\n")
            file.write(f"对端观测方位角: {result['remote_azimuth']}°\n")
            file.write(f"对端俯仰角: {el_remote:.3f}\n")
            file.write(f"目标相对方位角: {result['bearing_deg']}\n")
            file.write(f"目标东向坐标: {result['east_m']} m\n")
            file.write(f"目标南向坐标: {result['south_m']} m\n")
            file.write(f"本机到目标水平距离: {result['distance_m']} m\n")
            file.write(f"对端到目标水平距离: {result['peer_distance_m']} m\n")
            file.write(f"计算状态: {result['status']}\n")
            file.write(f"本机锁定图像: {host_path if host_saved else '保存失败'}\n")
            file.write(f"对端回传图像: {friend_path if friend_saved else '保存失败'}\n")

        with open(CSV_FILENAME, "a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(
                [
                    timestamp,
                    result["local_azimuth"],
                    result["remote_azimuth"],
                    host_path if host_saved else "",
                    friend_path if friend_saved else "",
                    result["bearing_deg"],
                    result["east_m"],
                    result["south_m"],
                    result["distance_m"],
                ]
            )

        self.status_update.emit(
            f"状态: {self.active_target_name}事件已保存，时间戳 {file_timestamp}"
        )
        return friend_saved

    @Slot(str, float, float, str)
    def on_save_command(self, timestamp, az_remote, el_remote, remote_img_b64):
        fire_image_received = self._save_fire_event(
            timestamp, az_remote, el_remote, remote_img_b64
        )

        if fire_image_received:
            self.fire_alert.emit(f"发现{self.active_target_name}！")


class FireMainWindow(SmoothMainWindow):
    """两端通用的对等协同检测界面。"""

    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle("双云台对等协同智能检测系统")
        self.resize(1800, 1000)
        self.setMinimumSize(1280, 840)

        self.manual_pan_speed = 20.0
        self.manual_tilt_speed = 20.0
        self.manual_direction_state = {
            "up": False,
            "down": False,
            "left": False,
            "right": False,
        }
        self.save_browser = None

        self.settings_window = SettingsWindow()
        self.thread = FireCameraThread()
        self.thread.frame_ready.connect(self.update_image)
        self.thread.remote_frame_ready.connect(self.update_remote_image)
        self.thread.status_update.connect(self.update_status_label)
        self.thread.angle_update.connect(self.update_angle_display)
        self.thread.ptz_state_update.connect(self.update_ptz_state_display)
        self.thread.geo_result_signal.connect(self.update_geo_result)
        self.thread.fire_alert.connect(self.show_fire_alert)
        self.thread.event_state_update.connect(self.update_event_state_display)
        self.settings_window.params_updated.connect(self.thread.set_relative_pose)
        self.settings_window.emit_params()

        self.rx_thread = UdpReceiverThread()
        self.rx_thread.save_trigger.connect(self.thread.on_save_command)
        self.rx_thread.ack_received.connect(self.thread.on_ack_received)
        self.rx_thread.remote_event_received.connect(self.update_remote_event_id)
        self.rx_thread.peer_event_started.connect(
            self.thread.on_peer_event_started
        )
        self.rx_thread.peer_event_ended.connect(self.thread.on_peer_event_ended)
        self.rx_thread.peer_image_received.connect(self.thread.on_peer_image)
        self.setup_ui()
        self.lbl_az_val.setText("N/A")
        self.lbl_el_val.setText("N/A")
        self.manual_control_group.setEnabled(False)
        self.update_status_label("状态: 云台未启动，请点击“启动云台”")
        self.append_operation_log("系统界面初始化完成")
        self.event_display_timer = QTimer(self)
        self.event_display_timer.timeout.connect(self.refresh_event_display)
        self.event_display_timer.start(1000)
        self.refresh_event_display()
        QApplication.instance().installEventFilter(self)
        # 界面完整创建后再启动后台线程，初始化异常时不会遗留 QThread。
        self.rx_thread.start()
        self.thread.start()

    def setup_ui(self):
        super().setup_ui()

        self.result_group.setTitle("目标双站交会结果")
        self.lbl_geo_lat.setText("方位(东0°顺时针): --")
        self.lbl_geo_lon.setText("坐标(东/南): --")
        self.lbl_geo_dist.setText("本机距离: ---- m")

        # 对端事件号与“对端画面”共用标题行，不再单独占用一行。
        self.lbl_remote_event = self.lbl_remote_title
        self.lbl_remote_event.setText("对端画面    事件号：--")
        self.lbl_remote_event.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_remote_event.setToolTip("当前尚无对端事件")

        event_control_bar = self._create_event_control_bar()
        bottom_bar = self.btn_video.parentWidget()
        online_layout = bottom_bar.parentWidget().layout()
        online_layout.insertWidget(online_layout.indexOf(bottom_bar), event_control_bar)

        # 最终窗口层再次锁定背景色，防止 Windows 原生主题覆盖页面空隙。
        window_palette = self.palette()
        window_palette.setColor(QPalette.Window, QColor("#071A33"))
        self.setPalette(window_palette)
        self.setAutoFillBackground(True)
        self.setStyleSheet(
            self.styleSheet()
            + "QMainWindow {background-color:#071A33;}"
              "QWidget#mainPanel {background-color:#071A33;}"
        )
        central = self.centralWidget()
        if central is not None:
            central.setAttribute(Qt.WA_StyledBackground, True)
            central.setStyleSheet(
                "SolidBackgroundWidget#mainPanel {background-color:#071A33;}"
            )

        self.btn_ptz = QPushButton("启动云台")
        self.btn_ptz.setObjectName("ptzButton")
        self.btn_ptz.setFixedWidth(105)
        self.btn_ptz.setStyleSheet(
            "QPushButton#ptzButton {color:#FFFFFF; background:#7C3AED; "
            "border:1px solid #7C3AED;}"
            "QPushButton#ptzButton:hover {background:#6D28D9;}"
            "QPushButton#ptzButton:disabled {color:#FFFFFF; background:#64748B; "
            "border-color:#64748B;}"
        )
        self.btn_ptz.clicked.connect(self.start_ptz)

        # 先放入原控制栏，待下方三栏布局重组时一起迁移。
        bottom_layout = self.btn_video.parentWidget().layout()
        video_button_index = bottom_layout.indexOf(self.btn_video)
        bottom_layout.insertWidget(video_button_index, self.btn_ptz)

        # 在所有功能控件创建完成后，只重组在线页的显示布局。
        # 原有按钮、信号连接、追踪与 UDP 逻辑均保持不变。
        self._rebuild_three_column_layout()

    def _rebuild_three_column_layout(self):
        """将在线监测页重排为：左侧控制、中间画面、右侧数据与事件。"""
        old_workspace = self.monitor_panel.parentWidget()
        old_manual_panel = self.manual_panel
        old_event_group = self.event_control_group
        old_bottom_bar = self.bottom_bar
        online_page = old_bottom_bar.parentWidget()
        online_layout = online_page.layout()

        three_columns = SolidBackgroundWidget()
        three_columns.setObjectName("threeColumnWorkspace")
        set_solid_background(three_columns)
        three_columns.setStyleSheet(
            "QWidget#threeColumnWorkspace {background:#071A33;}"
            "QFrame#leftControlSidebar, QFrame#rightInfoSidebar {"
            "background:#102D50; border:1px solid #2C527E; border-radius:6px;}"
            "QFrame#centerDisplayPanel {background:#071A33; border:none;}"
        )
        columns_layout = QHBoxLayout(three_columns)
        columns_layout.setContentsMargins(2, 0, 2, 0)
        columns_layout.setSpacing(12)

        # ---------------- 左侧：设备、启停和云台控制 ----------------
        left_sidebar = QFrame()
        left_sidebar.setObjectName("leftControlSidebar")
        left_sidebar.setFixedWidth(300)
        left_layout = QVBoxLayout(left_sidebar)
        left_layout.setContentsMargins(10, 8, 10, 10)
        left_layout.setSpacing(9)

        network_group = QGroupBox("设备与通信")
        network_layout = QGridLayout(network_group)
        network_layout.setContentsMargins(12, 20, 12, 11)
        network_layout.setHorizontalSpacing(8)
        network_layout.setVerticalSpacing(8)
        for field in (self.input_ip, self.input_port, self.input_rx_port):
            field.setMinimumWidth(0)
            field.setMaximumWidth(16777215)
        network_layout.addWidget(QLabel("对端 IP："), 0, 0)
        network_layout.addWidget(self.input_ip, 0, 1)
        network_layout.addWidget(QLabel("发送端口："), 1, 0)
        network_layout.addWidget(self.input_port, 1, 1)
        network_layout.addWidget(QLabel("接收端口："), 2, 0)
        network_layout.addWidget(self.input_rx_port, 2, 1)
        network_layout.setColumnStretch(1, 1)
        left_layout.addWidget(network_group)

        action_group = QGroupBox("系统控制")
        action_layout = QVBoxLayout(action_group)
        action_layout.setContentsMargins(12, 20, 12, 11)
        action_layout.setSpacing(8)
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("追踪目标："))
        self.combo_target.setMinimumWidth(0)
        self.combo_target.setMaximumWidth(16777215)
        target_row.addWidget(self.combo_target, 1)
        action_layout.addLayout(target_row)
        for button in (self.btn_ptz, self.btn_video, self.btn_action):
            button.setMinimumWidth(0)
            button.setMaximumWidth(16777215)
            action_layout.addWidget(button)
        left_layout.addWidget(action_group)

        left_layout.addWidget(self.manual_control_group)
        # 手动速度固定使用代码中的安全默认值，界面不再显示速度设置区。
        self.manual_speed_group.hide()
        self.manual_speed_group.setParent(three_columns)

        self.search_settings_group = QGroupBox("自动搜寻设置")
        search_settings_layout = QGridLayout(self.search_settings_group)
        search_settings_layout.setContentsMargins(12, 20, 12, 11)
        search_settings_layout.setHorizontalSpacing(7)
        search_settings_layout.setVerticalSpacing(8)
        self.input_search_interval = QLineEdit(
            f"{self.thread.search_interval_seconds:g}"
        )
        self.input_search_duration = QLineEdit(
            f"{self.thread.search_move_seconds:g}"
        )
        interval_validator = QDoubleValidator(1.0, 3600.0, 1, self)
        interval_validator.setNotation(QDoubleValidator.StandardNotation)
        duration_validator = QDoubleValidator(1.0, 600.0, 1, self)
        duration_validator.setNotation(QDoubleValidator.StandardNotation)
        self.input_search_interval.setValidator(interval_validator)
        self.input_search_duration.setValidator(duration_validator)
        self.input_search_interval.setMaximumWidth(78)
        self.input_search_duration.setMaximumWidth(78)
        search_settings_layout.addWidget(QLabel("搜寻间隔(秒):"), 0, 0)
        search_settings_layout.addWidget(self.input_search_interval, 0, 1)
        search_settings_layout.addWidget(QLabel("单次搜寻(秒):"), 1, 0)
        search_settings_layout.addWidget(self.input_search_duration, 1, 1)
        self.apply_search_button = QPushButton("应用搜寻设置")
        self.apply_search_button.clicked.connect(self.apply_search_settings)
        search_settings_layout.addWidget(
            self.apply_search_button, 2, 0, 1, 2, Qt.AlignCenter
        )
        left_layout.addWidget(self.search_settings_group)

        left_layout.addStretch(1)
        self.lbl_status.setMinimumWidth(0)
        self.lbl_status.setMaximumWidth(16777215)
        self.lbl_status.setMinimumHeight(54)
        self.lbl_status.setWordWrap(True)
        left_layout.addWidget(self.lbl_status)

        # ---------------- 中间：本机与对端画面 ----------------
        center_panel = QFrame()
        center_panel.setObjectName("centerDisplayPanel")
        center_panel.setMinimumWidth(450)
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(8)

        # 中间保持上下两层：上层是本机主画面；下层将对端
        # 画面和异常状态块改为左右并排。
        video_layout = self.video_area.layout()
        local_card = self.lbl_local.parentWidget()
        remote_card = self.lbl_remote.parentWidget()
        while video_layout.count():
            item = video_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        vertical_video_layout = QVBoxLayout()
        vertical_video_layout.setContentsMargins(0, 0, 0, 0)
        vertical_video_layout.setSpacing(10)
        vertical_video_layout.addWidget(local_card, 2)
        lower_display_layout = QHBoxLayout()
        lower_display_layout.setContentsMargins(0, 0, 0, 0)
        lower_display_layout.setSpacing(10)
        lower_display_layout.addWidget(remote_card, 1)

        # 状态只做提示卡片，不再占满对端画面右侧全部高度。
        self.fire_status_indicator.setFixedSize(190, 64)
        lower_display_layout.addWidget(
            self.fire_status_indicator, 0, Qt.AlignCenter
        )
        vertical_video_layout.addLayout(lower_display_layout, 1)
        video_layout.addLayout(vertical_video_layout)

        self.video_area.setMinimumHeight(610)
        self.video_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_local.setMinimumSize(0, 350)
        self.lbl_remote.setMinimumSize(0, 205)
        center_layout.addWidget(self.video_area, 1)

        # ---------------- 右侧：数据、定位、事件与保存记录 ----------------
        right_sidebar = QFrame()
        right_sidebar.setObjectName("rightInfoSidebar")
        right_sidebar.setFixedWidth(420)
        right_layout = QVBoxLayout(right_sidebar)
        right_layout.setContentsMargins(10, 8, 10, 10)
        right_layout.setSpacing(9)

        # 七项状态使用横向信息卡：名称在左、数值在右，避免上下两行
        # 被压缩；“设置坐标”另占整行。
        state_cards = []
        for value_label in self.ptz_status_labels.values():
            card = value_label.parentWidget()
            if card is not None and card not in state_cards:
                state_cards.append(card)
        settings_button = next(
            (
                button
                for button in self.angle_group.findChildren(QPushButton)
                if button.text() == "设置坐标"
            ),
            None,
        )
        if settings_button is not None:
            settings_button.setText("设置相对位置")
        angle_layout = self.angle_group.layout()
        while angle_layout.count():
            item = angle_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        state_grid = QGridLayout()
        state_grid.setContentsMargins(0, 0, 0, 0)
        state_grid.setHorizontalSpacing(10)
        state_grid.setVerticalSpacing(9)
        for index, card in enumerate(state_cards):
            card.setFixedHeight(50)
            card.setMinimumWidth(0)
            card_layout = card.layout()
            if isinstance(card_layout, QBoxLayout):
                card_layout.setDirection(QBoxLayout.LeftToRight)
                card_layout.setContentsMargins(10, 6, 10, 6)
                card_layout.setSpacing(8)
                title_label = card_layout.itemAt(0).widget()
                value_label = card_layout.itemAt(1).widget()
                if isinstance(title_label, QLabel):
                    title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                    title_label.setMinimumWidth(68)
                    title_label.setStyleSheet(
                        "color:#9FC0E2; font-size:11px; background:transparent;"
                    )
                if isinstance(value_label, QLabel):
                    value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    value_label.setMinimumWidth(48)
            if index == len(state_cards) - 1:
                state_grid.addWidget(card, 3, 0, 1, 2)
            else:
                state_grid.addWidget(card, index // 2, index % 2)
        if settings_button is not None:
            settings_button.setMinimumWidth(0)
            settings_button.setMaximumWidth(16777215)
            settings_button.setFixedHeight(42)
            state_grid.addWidget(settings_button, 4, 0, 1, 2)
        state_grid.setColumnStretch(0, 1)
        state_grid.setColumnStretch(1, 1)
        angle_layout.setContentsMargins(10, 20, 10, 9)
        angle_layout.addLayout(state_grid)
        self.angle_group.setFixedHeight(310)
        right_layout.addWidget(self.angle_group)

        # 经纬度和距离改为竖向排列，避免文字被压缩。
        result_layout = self.result_group.layout()
        result_labels = (self.lbl_geo_lat, self.lbl_geo_lon, self.lbl_geo_dist)
        while result_layout.count():
            item = result_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        result_column = QVBoxLayout()
        result_column.setContentsMargins(0, 0, 0, 0)
        result_column.setSpacing(5)
        for label in result_labels:
            label.setMinimumHeight(29)
            result_column.addWidget(label)
        result_layout.setContentsMargins(10, 20, 10, 9)
        result_layout.addLayout(result_column)
        self.result_group.setFixedHeight(120)
        right_layout.addWidget(self.result_group)

        event_group = QGroupBox("事件与对端协同")
        self.event_control_group = event_group
        event_layout = QVBoxLayout(event_group)
        event_layout.setContentsMargins(10, 20, 10, 9)
        event_layout.setSpacing(6)
        self.current_event_label.setMinimumWidth(0)
        self.current_event_label.setMaximumWidth(16777215)
        self.current_event_label.setMinimumHeight(44)
        event_layout.addWidget(self.current_event_label)
        self.end_event_button.setMinimumWidth(0)
        self.end_event_button.setMaximumWidth(16777215)
        event_layout.addWidget(self.end_event_button)
        event_group.setFixedHeight(118)
        right_layout.addWidget(event_group)

        self.operation_log.setMinimumHeight(90)
        self.log_group.setMinimumHeight(175)
        self.log_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(self.log_group, 1)

        def create_sidebar_scroll(sidebar, object_name, width):
            """为左右侧栏创建独立滚动区，中间显示区不使用滚动。"""
            sidebar.setMinimumWidth(0)
            sidebar.setMaximumWidth(16777215)
            sidebar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
            scroll = QScrollArea()
            scroll.setObjectName(object_name)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            scroll.setFixedWidth(width)
            scroll.setStyleSheet(
                f"QScrollArea#{object_name} {{background:#071A33; border:none;}}"
                f"QScrollArea#{object_name} > QWidget > QWidget "
                "{background:#102D50;}"
                "QScrollBar:vertical {background:#091E39; width:10px; margin:0;}"
                "QScrollBar::handle:vertical {background:#2A5688; "
                "border-radius:5px; min-height:36px;}"
                "QScrollBar::handle:vertical:hover {background:#3971A9;}"
                "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical "
                "{height:0; background:none;}"
                "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical "
                "{background:none;}"
            )
            scroll.setWidget(sidebar)
            return scroll

        left_scroll = create_sidebar_scroll(
            left_sidebar, "leftControlScroll", 310
        )
        # create_sidebar_scroll() 会把内容改成 Expanding，按钮文字在
        # “启动/停止”之间变化时会重新计算最小宽度并撑开整个左栏。
        # 左侧滚动条占10像素，因此将内容和视口固定为300像素。
        left_sidebar.setFixedWidth(300)
        left_sidebar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.MinimumExpanding)
        left_scroll.viewport().setFixedWidth(300)
        right_scroll = create_sidebar_scroll(
            right_sidebar, "rightInfoScroll", 430
        )

        columns_layout.addWidget(left_scroll)
        columns_layout.addWidget(center_panel, 1)
        columns_layout.addWidget(right_scroll)

        # 移除旧的“上下分区+底部长条”容器，放入新三栏容器。
        while online_layout.count():
            item = online_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        online_layout.addWidget(three_columns, 1)
        online_page.setMinimumWidth(0)
        online_page.setMinimumHeight(0)

        # 不再仅隐藏外层滚动条，而是彻底移除外层 QScrollArea。
        # 这样鼠标滚轮也无法带动中间画面上下偏移。
        outer_scroll = self.online_scroll
        online_index = self.content_stack.indexOf(outer_scroll)
        outer_scroll.takeWidget()
        self.content_stack.removeWidget(outer_scroll)
        self.content_stack.insertWidget(online_index, online_page)
        self.content_stack.setCurrentIndex(online_index)
        outer_scroll.deleteLater()
        self.online_scroll = None
        self.online_page = online_page

        self.online_three_columns = three_columns
        self.left_control_sidebar = left_sidebar
        self.center_display_panel = center_panel
        self.right_info_sidebar = right_sidebar
        self.left_control_scroll = left_scroll
        self.right_info_scroll = right_scroll

        # 旧容器中的功能控件都已重新挂载，可安全释放空容器。
        for old_container in (old_workspace, old_event_group, old_bottom_bar):
            old_container.deleteLater()
        old_manual_panel.deleteLater()

    @Slot()
    def start_ptz(self):
        if not self.thread.start_ptz():
            if self.thread.is_ptz_started():
                self.update_status_label("状态: 云台已经启动")
            return

        self.btn_ptz.setText("云台已启动")
        self.btn_ptz.setEnabled(False)
        self.manual_control_group.setEnabled(not self.thread.tracking_active)

    @Slot(str, str, str)
    def update_geo_result(self, bearing, position_xy, distance):
        self.lbl_geo_lat.setText(f"方位(东0°顺时针): {bearing}")
        self.lbl_geo_lon.setText(f"坐标(东/南): {position_xy}")
        self.lbl_geo_dist.setText(f"本机距离: {distance} m")

    def update_rx_port(self):
        try:
            self.thread.set_ack_receive_port(int(self.input_rx_port.text()))
        except ValueError:
            pass
        super().update_rx_port()

    def toggle_tracking(self):
        if not self.thread.tracking_active and not self.thread.is_ptz_started():
            self.update_status_label("状态: 请先点击“启动云台”")
            return
        super().toggle_tracking()

    def _set_tracking_ui(self, active):
        super()._set_tracking_ui(active)
        if not self.thread.is_ptz_started():
            self.manual_control_group.setEnabled(False)

    def _set_manual_direction(self, action, pressed):
        if pressed and not self.thread.is_ptz_started():
            self.update_status_label("状态: 请先点击“启动云台”")
            return
        super()._set_manual_direction(action, pressed)

    def _create_event_control_bar(self):
        event_group = QGroupBox("事件与对端协同")
        self.event_control_group = event_group
        event_group.setFixedHeight(90)
        event_layout = QHBoxLayout(event_group)
        event_layout.setContentsMargins(16, 21, 16, 11)
        event_layout.setSpacing(11)

        self.current_event_label = QLabel("当前事件：无")
        self.current_event_label.setWordWrap(True)
        self.current_event_label.setMinimumWidth(290)
        self.current_event_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.current_event_label.setStyleSheet(
            "color:#72C3FF; background:#0D2948; border:1px solid #2A5688; "
            "border-radius:4px; padding:7px; font-size:11px;"
        )
        event_layout.addWidget(self.current_event_label, 2)

        self.end_event_button = QPushButton("结束事件")
        self.end_event_button.setFixedWidth(82)
        self.end_event_button.clicked.connect(self.end_current_event)
        self.end_event_button.setStyleSheet(
            "QPushButton {background:#B83244; border-color:#D44859;}"
            "QPushButton:hover {background:#D13B50;}"
        )
        event_layout.addWidget(self.end_event_button)
        return event_group

    def _create_manual_ptz_panel(self):
        panel = super()._create_manual_ptz_panel()
        panel_layout = panel.layout()

        log_group = QGroupBox("操作日志")
        self.log_group = log_group
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(8, 18, 8, 8)
        self.operation_log = QPlainTextEdit()
        self.operation_log.setReadOnly(True)
        self.operation_log.setMinimumHeight(100)
        self.operation_log.setStyleSheet(
            "QPlainTextEdit {"
            "color:#DCEBFA; background:#061A34; border:1px solid #2A5688; "
            "border-radius:5px; padding:5px; font-size:11px;"
            "}"
        )
        self.operation_log.document().setMaximumBlockCount(300)
        view_saved_button = QPushButton("查看保存记录")
        self.view_saved_button = view_saved_button
        view_saved_button.setStyleSheet(
            "QPushButton {color:#FFFFFF; background:#0F766E; "
            "border:1px solid #0F766E;}"
            "QPushButton:hover {background:#0D9488;}"
        )
        view_saved_button.clicked.connect(self.open_save_browser)
        log_layout.addWidget(view_saved_button)
        log_layout.addWidget(self.operation_log)

        self.fire_status_indicator = QLabel("无异常")
        self.fire_status_indicator.setAlignment(Qt.AlignCenter)
        self.fire_status_indicator.setMinimumHeight(58)
        self.fire_status_indicator.setStyleSheet(
            "QLabel {color:#FFFFFF; background:#16A34A; "
            "border:2px solid #15803D; border-radius:6px; "
            "font-size:18px; font-weight:700; padding:8px;}"
        )
        log_layout.addWidget(self.fire_status_indicator)

        # 原面板最后一项是弹性空白，把日志插入空白之前并占据右下区域。
        insert_index = max(0, panel_layout.count() - 1)
        panel_layout.insertWidget(insert_index, log_group, 1)
        # 加宽右侧区域，避免状态文字换行过多和日志区过窄。
        panel.setFixedWidth(340)
        return panel

    def apply_search_settings(self):
        try:
            interval = float(self.input_search_interval.text())
            duration = float(self.input_search_duration.text())
        except ValueError:
            self.update_status_label("状态: 搜寻时间请输入有效数字")
            return
        if not 1.0 <= interval <= 3600.0:
            self.update_status_label("状态: 搜寻间隔请输入 1 到 3600 秒")
            return
        if not 1.0 <= duration <= 600.0:
            self.update_status_label("状态: 单次搜寻请输入 1 到 600 秒")
            return
        self.input_search_interval.setText(f"{interval:g}")
        self.input_search_duration.setText(f"{duration:g}")
        self.thread.set_search_timing(interval, duration)

    def end_current_event(self):
        self.thread.manual_end_event()
        self.refresh_event_display()

    def _update_friend_network_target(self):
        try:
            self.thread.set_network_config(
                self.input_ip.text().strip(), int(self.input_port.text())
            )
            return True
        except ValueError:
            self.update_status_label("状态: 对端目标端口格式错误")
            return False

    @Slot(str, str)
    def update_event_state_display(self, event_id, state):
        state_names = {
            "IDLE": "无活动事件",
            "READY": "已建立，等待目标锁定",
            "WAIT_ACK": "等待对端确认",
            "ACKED": "对端已确认",
            "REMOTE_ACTIVE": "响应方持续搜寻中",
        }
        snapshot = self.thread.get_event_snapshot()
        role_name = {
            "ORIGIN": "发起方",
            "RESPONDER": "响应方",
            "IDLE": "未分配",
        }.get(snapshot.get("role", "IDLE"), "未分配")
        target_name = snapshot.get("target") or "--"
        end_rule = (
            "等待发起方结束事件"
            if snapshot.get("role") == "RESPONDER"
            else f"连续 {UDP_EVENT_MISSING_TIMEOUT_SECONDS:g} 秒无目标结束"
        )
        if event_id:
            self.current_event_label.setText(
                f"当前事件：{event_id}\n"
                f"角色：{role_name}｜目标：{target_name}｜"
                f"状态：{state_names.get(state, state)}｜{end_rule}"
            )
            self.current_event_label.setToolTip(event_id)
        else:
            self.current_event_label.setText("当前事件：无\n状态：无活动事件")
            self.current_event_label.setToolTip("")
            self.clear_remote_event_display()
            self.set_fire_status_normal()

    def refresh_event_display(self):
        snapshot = self.thread.get_event_snapshot()
        event_id = snapshot["event_id"]
        state = snapshot["state"]
        state_names = {
            "IDLE": "无活动事件",
            "READY": "已建立，等待目标锁定",
            "WAIT_ACK": "等待对端确认",
            "ACKED": "对端已确认",
            "REMOTE_ACTIVE": "持续搜寻并每5秒回传图像",
        }
        role_name = {
            "ORIGIN": "发起方",
            "RESPONDER": "响应方",
            "IDLE": "未分配",
        }.get(snapshot.get("role", "IDLE"), "未分配")
        target_name = snapshot.get("target") or "--"
        end_rule = (
            "等待发起方结束事件"
            if snapshot.get("role") == "RESPONDER"
            else f"连续 {UDP_EVENT_MISSING_TIMEOUT_SECONDS:g} 秒无目标结束"
        )
        if event_id:
            self.current_event_label.setText(
                f"当前事件：{event_id}\n"
                f"角色：{role_name}｜目标：{target_name}｜"
                f"状态：{state_names.get(state, state)}｜{end_rule}"
            )
            self.current_event_label.setToolTip(event_id)
            if snapshot.get("role") == "RESPONDER":
                desired_mode = "person" if target_name == "person" else "fire"
                if self.combo_target.currentText() != desired_mode:
                    self.combo_target.blockSignals(True)
                    self.combo_target.setCurrentText(desired_mode)
                    self.combo_target.blockSignals(False)
                if self.thread.is_ptz_started():
                    self.btn_ptz.setText("云台已启动")
                    self.btn_ptz.setEnabled(False)
                else:
                    self.btn_ptz.setText("启动云台")
                    self.btn_ptz.setEnabled(True)
                self.btn_video.setText("停止视频")
                self.btn_video.setProperty("streaming", True)
                self.btn_video.style().unpolish(self.btn_video)
                self.btn_video.style().polish(self.btn_video)
                if not bool(self.btn_action.property("tracking")):
                    self._set_tracking_ui(True)
        elif snapshot["manual_hold"]:
            self.current_event_label.setText(
                "当前事件：无\n状态：已手动结束，等待当前目标离开"
            )
        else:
            self.current_event_label.setText("当前事件：无\n状态：等待目标锁定")

    @Slot(str)
    def update_remote_event_id(self, event_id):
        event_text = str(event_id).strip() or "未知事件"
        if "__REFRESH_PREP__" in event_text:
            event_text = event_text.split("__REFRESH_PREP__", 1)[0]
        self.lbl_remote_event.setText(f"对端画面    事件号：{event_text}")
        self.lbl_remote_event.setToolTip(event_text)
        self.append_operation_log(f"收到对端通信，事件号：{event_text}")

    def clear_remote_event_display(self):
        if not hasattr(self, "lbl_remote"):
            return
        self.lbl_remote.clear()
        self.lbl_remote.setText("等待对端图像回传…")
        if hasattr(self, "lbl_remote_event"):
            self.lbl_remote_event.setText("对端画面    事件号：--")
            self.lbl_remote_event.setToolTip("当前尚无对端事件")

    def set_fire_status_normal(self):
        if not hasattr(self, "fire_status_indicator"):
            return
        self.fire_status_indicator.setText("无异常")
        self.fire_status_indicator.setStyleSheet(
            "QLabel {color:#FFFFFF; background:#16A34A; "
            "border:2px solid #15803D; border-radius:6px; "
            "font-size:18px; font-weight:700; padding:8px;}"
        )

    def append_operation_log(self, message):
        if not hasattr(self, "operation_log"):
            return
        log_time = datetime.now().strftime("%H:%M:%S")
        self.operation_log.appendPlainText(f"[{log_time}] {message}")

    @Slot()
    def open_save_browser(self):
        if self.save_browser is None:
            self.save_browser = SaveBrowserDialog(self)
        else:
            self.save_browser.refresh_files()
        self.save_browser.show()
        self.save_browser.raise_()
        self.save_browser.activateWindow()
        self.append_operation_log("打开保存记录查看窗口")

    @Slot(str)
    def update_status_label(self, text):
        super().update_status_label(text)
        self.append_operation_log(text)
        if (
            (
                text.startswith("状态: 火源事件已保存")
                or text.startswith("状态: 已保存")
            )
            and self.save_browser is not None
            and self.save_browser.isVisible()
        ):
            self.save_browser.refresh_files()

    @Slot(str)
    def show_fire_alert(self, message):
        message = str(message).strip() or "发现目标！"
        self.append_operation_log(f"目标提示: {message}")
        self.fire_status_indicator.setText(message)
        self.fire_status_indicator.setStyleSheet(
            "QLabel {color:#FFFFFF; background:#DC2626; "
            "border:2px solid #B91C1C; border-radius:6px; "
            "font-size:18px; font-weight:700; padding:8px;}"
        )


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 9))
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#071A33"))
    palette.setColor(QPalette.WindowText, QColor("#E6F0FF"))
    palette.setColor(QPalette.Base, QColor("#061A34"))
    palette.setColor(QPalette.AlternateBase, QColor("#102E50"))
    palette.setColor(QPalette.Text, QColor("#E6F0FF"))
    palette.setColor(QPalette.Button, QColor("#285B9A"))
    palette.setColor(QPalette.ButtonText, QColor("#FFFFFF"))
    palette.setColor(QPalette.Highlight, QColor("#1976D2"))
    palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    app.setPalette(palette)
    window = FireMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
