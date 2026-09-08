"""Read-only Vega camera adapter; no motor commands or mandatory dexcontrol import."""

from __future__ import annotations

import threading
import time
from typing import Any, Mapping

import numpy as np


def validate_dexmate_camera(config: Mapping[str, Any]) -> np.ndarray:
    """Require explicit calibration of the selected rectified optical image."""
    if config.get("rectified") is not True:
        raise ValueError("Dexmate requires rectified: true and rectified image intrinsics")
    if config.get("image_key", "left_rgb") not in ("left_rgb", "right_rgb"):
        raise ValueError("image_key must be left_rgb or right_rgb")
    if not config.get("frame_id"):
        raise ValueError("camera.frame_id must identify the selected optical frame")
    if int(config["width"]) <= 0 or int(config["height"]) <= 0:
        raise ValueError("Calibrated width and height must be positive")
    K = np.asarray(config.get("K"), dtype=np.float64)
    if (K.shape != (3, 3) or not np.isfinite(K).all()
            or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1])):
        raise ValueError("Set camera.K to measured rectified intrinsics; no default is provided")
    for field in ("requested_fps", "read_timeout_s"):
        value = float(config[field])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"camera.{field} must be finite and positive")
    return K


class DexmateLatestFrameCamera:
    """Poll the SDK into one latest BGR buffer, dropping identical cached images.

    The manual documents no acquisition timestamp API. The returned timestamp is
    local receipt time, NOT sensor exposure time. Equal images are conservatively
    treated as cached frames; a perfectly static stream can therefore time out.
    """

    timestamp_source = "host_receive_time"

    def __init__(self, config: Mapping[str, Any], *, robot: Any = None):
        validate_dexmate_camera(config)
        self._shape = (int(config["height"]), int(config["width"]), 3)
        self._image_key = config.get("image_key", "left_rgb")
        self._poll_s = 1.0 / float(config["requested_fps"])
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._frame = None
        self._sequence = -1
        self._received = 0.0
        self._received_monotonic = 0.0
        self._error = None
        self._robot = robot
        self._owns_robot = robot is None
        sensor_name = config.get("sensor", "head_camera")
        if robot is None:
            try:
                from dexcontrol.core.config import get_robot_config
                from dexcontrol.robot import Robot
            except ImportError as exc:
                raise RuntimeError("Install the vendor dexcontrol SDK on the inference host") from exc
            configs = get_robot_config()
            if sensor_name not in configs.sensors:
                raise ValueError(f"Sensor {sensor_name!r} is absent from ROBOT_CONFIG")
            # Enable only the requested client sensor; do not mutate robot/service settings.
            for name, sensor in configs.sensors.items():
                sensor.enabled = name == sensor_name
            self._robot = Robot(configs=configs)
        try:
            self._sensor = getattr(self._robot.sensors, sensor_name)
            self._thread = threading.Thread(target=self._reader, daemon=True,
                                            name="dexmate-brick-camera")
            self._thread.start()
        except BaseException:
            if self._owns_robot:
                self._robot.shutdown()
            raise

    def _reader(self) -> None:
        previous = None
        try:
            while not self._stop.is_set():
                data = self._sensor.get_obs(obs_keys=[self._image_key])
                rgb = None if data is None else data.get(self._image_key)
                if rgb is not None:
                    rgb = np.asarray(rgb)
                    if rgb.shape != self._shape or rgb.dtype != np.uint8:
                        raise ValueError(
                            f"Expected calibrated uint8 RGB {self._shape}; got "
                            f"{rgb.shape} {rgb.dtype}. No implicit resize is allowed."
                        )
                    rgb = rgb.copy()  # SDK may reuse its image buffer.
                    if previous is None or not np.array_equal(previous, rgb):
                        previous = rgb
                        with self._condition:
                            self._frame = rgb[:, :, ::-1].copy()
                            self._received = time.time()
                            self._received_monotonic = time.monotonic()
                            self._sequence += 1
                            self._condition.notify_all()
                self._stop.wait(self._poll_s)
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()

    def read_latest(self, after_sequence: int, timeout_s: float):
        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                if self._stop.is_set():
                    raise RuntimeError("Dexmate camera is closed")
                if self._error is not None:
                    raise RuntimeError("Dexmate camera read failed") from self._error
                if (self._sequence > after_sequence and self._frame is not None
                        and time.monotonic() - self._received_monotonic < timeout_s):
                    return self._sequence, self._received, self._frame.copy()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("No changed Dexmate RGB frame; check dexsensor and transport")
                self._condition.wait(remaining)

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        try:
            if self._owns_robot:
                self._robot.shutdown()
        finally:
            self._thread.join(timeout=1.0)
