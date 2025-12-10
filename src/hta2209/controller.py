"""
Hardware abstraction and simple autopilot for the HTA2209 platform.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import cv2

try:
    from adafruit_servokit import ServoKit

    _SERVOKIT_IMPORT_ERROR = None
except Exception as exc:  # Platform may not be supported (e.g., Windows dev)
    ServoKit = None
    _SERVOKIT_IMPORT_ERROR = exc

LOGGER = logging.getLogger(__name__)

WHEEL_CHANNELS = {
    "front_left": 0,
    "front_right": 1,
    "rear_left": 2,
    "rear_right": 3,
}

ARM_CHANNELS = {
    "base": 4,
    "shoulder": 5,
    "elbow": 6,
    "wrist": 7,
    "gripper": 8,
}

SUPPORTED_MODES = ("manual", "auto")
DEFAULT_COLORS = ("red", "green", "blue")


@dataclass
class Threshold:
    hue_min: int = 0
    hue_max: int = 30
    sat_min: int = 50
    sat_max: int = 255
    val_min: int = 50
    val_max: int = 255

    def clamp(self) -> "Threshold":
        self.hue_min = max(0, min(179, self.hue_min))
        self.hue_max = max(0, min(179, self.hue_max))
        self.sat_min = max(0, min(255, self.sat_min))
        self.sat_max = max(0, min(255, self.sat_max))
        self.val_min = max(0, min(255, self.val_min))
        self.val_max = max(0, min(255, self.val_max))
        return self

    def as_dict(self) -> Dict[str, int]:
        return {
            "hue_min": self.hue_min,
            "hue_max": self.hue_max,
            "sat_min": self.sat_min,
            "sat_max": self.sat_max,
            "val_min": self.val_min,
            "val_max": self.val_max,
        }


class RobotController:
    """
    PCA9685 channel abstraction, threshold management, and a basic color-follow autopilot.
    """

    def __init__(self, config_path: Path | str = "config/settings.json", channels: int = 16) -> None:
        self.config_path = Path(config_path)
        self.channels = channels
        self.kit = None
        self.hardware_ready = False

        self.wheel_state: Dict[str, float] = {wheel: 0.0 for wheel in WHEEL_CHANNELS}
        self.wheel_polarity: Dict[str, int] = {wheel: 1 for wheel in WHEEL_CHANNELS}
        self.arm_state: Dict[str, float] = {joint: 90.0 for joint in ARM_CHANNELS}
        self.color_thresholds: Dict[str, Threshold] = {
            color: Threshold().clamp() for color in DEFAULT_COLORS
        }
        self.auto_threshold_enabled: bool = False
        self.auto_target_color: str = DEFAULT_COLORS[0]
        self.auto_grasped: bool = False
        self.mode: str = "manual"
        self.simulation_enabled: bool = False
        self.last_target: Optional[Tuple[int, int, int]] = None  # (cx, cy, area)
        self.supply_voltage: float = 5.0
        self.supply_current: float = 0.0
        self.power_consumption_w: float = 0.0
        self.pwm_frequency_hz: float = 50.0
        self.pwm_output: Dict[str, float] = {wheel: 0.0 for wheel in WHEEL_CHANNELS}

        self._connect_to_hardware()
        self.load_config()

    # ------------------------------------------------------------------ #
    # Hardware interaction
    # ------------------------------------------------------------------ #
    def _connect_to_hardware(self) -> None:
        if ServoKit is None:
            if _SERVOKIT_IMPORT_ERROR:
                LOGGER.info(
                    "ServoKit import edilemedi veya desteklenmeyen platform: %s. Simulasyon modunda kalinacak.",
                    _SERVOKIT_IMPORT_ERROR,
                )
            else:
                LOGGER.info("ServoKit modulu bulunamadi; simulasyon modunda calisiyor.")
            return
        try:
            self.kit = ServoKit(channels=self.channels)
            self.hardware_ready = True
            LOGGER.info("PCA9685 icin ServoKit baslatildi.")
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("ServoKit baslatilamadi, simulasyon moduna geciliyor: %s", exc)
            self.hardware_ready = False

    # ------------------------------------------------------------------ #
    # State management helpers
    # ------------------------------------------------------------------ #
    def set_wheel_speed(self, wheel: str, speed_percentage: float) -> None:
        if wheel not in self.wheel_state:
            raise ValueError(f"Bilinmeyen tekerlek: {wheel}")
        clamped = max(-100.0, min(100.0, speed_percentage))
        self.wheel_state[wheel] = clamped
        throttle = (self.wheel_polarity.get(wheel, 1) * clamped) / 100.0
        self.pwm_output[wheel] = throttle
        LOGGER.debug("Tekerlek %s hizi %.1f%%", wheel, clamped)
        if self.hardware_ready and not self.is_simulation():  # pragma: no cover
            self._apply_wheel_speed_to_hardware(wheel, clamped)
        self._recompute_power()

    def _apply_wheel_speed_to_hardware(self, wheel: str, speed: float) -> None:
        channel = WHEEL_CHANNELS[wheel]
        try:
            servo = self.kit.continuous_servo[channel]
            servo.throttle = (self.wheel_polarity.get(wheel, 1) * speed) / 100.0
        except Exception as exc:
            LOGGER.error("Tekerlek %s icin throttle ayarlanamadi: %s", wheel, exc)

    def set_joint_angle(self, joint: str, angle: float) -> None:
        if joint not in self.arm_state:
            raise ValueError(f"Bilinmeyen eklem: {joint}")
        clamped = max(0.0, min(180.0, angle))
        self.arm_state[joint] = clamped
        LOGGER.debug("Eklem %s acisi %.1f", joint, clamped)
        if self.hardware_ready and not self.is_simulation():  # pragma: no cover
            self._apply_joint_angle_to_hardware(joint, clamped)

    def _apply_joint_angle_to_hardware(self, joint: str, angle: float) -> None:
        channel = ARM_CHANNELS[joint]
        try:
            servo = self.kit.servo[channel]
            servo.angle = angle
        except Exception as exc:
            LOGGER.error("Eklem %s icin aci uygulanamadi: %s", joint, exc)

    def set_color_threshold(self, color: str, field_name: str, value: int) -> None:
        if color not in self.color_thresholds:
            raise ValueError(f"Bilinmeyen renk: {color}")
        if not hasattr(self.color_thresholds[color], field_name):
            raise ValueError(f"Bilinmeyen alan: {field_name}")
        setattr(self.color_thresholds[color], field_name, int(value))
        self.color_thresholds[color].clamp()
        LOGGER.debug("Renk %s alan %s degeri %s olarak kaydedildi.", color, field_name, value)

    # ------------------------------------------------------------------ #
    # Persistence helpers
    # ------------------------------------------------------------------ #
    def serialize_state(self) -> Dict[str, Dict[str, float]]:
        return {
            "mode": self.mode,
            "simulation": self.simulation_enabled,
            "auto_threshold": self.auto_threshold_enabled,
            "auto_target_color": self.auto_target_color,
            "wheel_polarity": self.wheel_polarity,
            "wheels": self.wheel_state,
            "arm": self.arm_state,
            "thresholds": {color: thr.as_dict() for color, thr in self.color_thresholds.items()},
        }

    def save_config(self) -> None:
        data = self.serialize_state()
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        LOGGER.info("Yapilandirma %s dosyasina kaydedildi.", self.config_path)

    def load_config(self) -> None:
        if not self.config_path.exists():
            LOGGER.info("Konfigurasyon dosyasi bulunamadi, varsayilan degerler kullanilacak.")
            return
        try:
            with self.config_path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except json.JSONDecodeError as exc:
            LOGGER.error("Konfigurasyon okunamadi (%s), varsayilan degerler kullanilacak.", exc)
            return
        for wheel, value in raw.get("wheels", {}).items():
            if wheel in self.wheel_state:
                self.wheel_state[wheel] = float(value)
        for wheel, val in raw.get("wheel_polarity", {}).items():
            if wheel in self.wheel_polarity:
                self.wheel_polarity[wheel] = 1 if int(val) >= 0 else -1
        for joint, value in raw.get("arm", {}).items():
            if joint in self.arm_state:
                self.arm_state[joint] = float(value)
        for color, payload in raw.get("thresholds", {}).items():
            if color in self.color_thresholds:
                for field_name, val in payload.items():
                    if hasattr(self.color_thresholds[color], field_name):
                        setattr(self.color_thresholds[color], field_name, int(val))
                self.color_thresholds[color].clamp()
        mode_name = raw.get("mode", "manual")
        # eski konfigurasyonlarda simulation bir mode olarak gelebilir
        if mode_name.lower() == "simulation":
            self.simulation_enabled = True
            mode_name = "manual"
        try:
            self.set_mode(mode_name)
        except ValueError:
            LOGGER.warning("Bilinmeyen kontrol modu %s, manuel mod kullanilacak.", mode_name)
        self.auto_threshold_enabled = bool(raw.get("auto_threshold", False))
        target = raw.get("auto_target_color", DEFAULT_COLORS[0])
        if target in self.color_thresholds:
            self.auto_target_color = target
        self.simulation_enabled = bool(raw.get("simulation", self.simulation_enabled))

        self._recompute_power()
        LOGGER.info("Konfigurasyon %s dosyasindan yuklendi.", self.config_path)

    # ------------------------------------------------------------------ #
    # Convenience queries
    # ------------------------------------------------------------------ #
    def wheels(self) -> Tuple[str, ...]:
        return tuple(self.wheel_state.keys())

    def joints(self) -> Tuple[str, ...]:
        return tuple(self.arm_state.keys())

    def colors(self) -> Tuple[str, ...]:
        return tuple(self.color_thresholds.keys())

    # ------------------------------------------------------------------ #
    # Mode helpers
    # ------------------------------------------------------------------ #
    def set_mode(self, mode: str) -> None:
        normalized = mode.lower()
        if normalized not in SUPPORTED_MODES:
            raise ValueError(f"Bilinmeyen kontrol modu: {mode}")
        if normalized == self.mode:
            return
        self.mode = normalized
        LOGGER.info("Kontrol modu %s olarak ayarlandi.", normalized)
        self.stop_all_motion()
        self.auto_grasped = False

    def is_manual(self) -> bool:
        return self.mode == "manual"

    def is_simulation(self) -> bool:
        return self.simulation_enabled

    def set_simulation(self, enabled: bool) -> None:
        self.simulation_enabled = bool(enabled)
        LOGGER.info("Simulasyon modu %s.", "acik" if self.simulation_enabled else "kapali")

    def stop_all_motion(self) -> None:
        for wheel in self.wheel_state:
            self.set_wheel_speed(wheel, 0)
        if self.hardware_ready and not self.is_simulation():
            for wheel in self.wheel_state:
                self._apply_wheel_speed_to_hardware(wheel, 0)
        self._recompute_power()

    # ------------------------------------------------------------------ #
    # Auto threshold helpers
    # ------------------------------------------------------------------ #
    def set_auto_threshold_enabled(self, enabled: bool) -> None:
        self.auto_threshold_enabled = bool(enabled)
        LOGGER.info("Otomatik esikleme %s.", "acik" if self.auto_threshold_enabled else "kapali")

    def auto_calibrate_from_frame(self, hsv_frame: np.ndarray, low_pct: float = 5.0, high_pct: float = 95.0) -> None:
        if hsv_frame.size == 0:
            LOGGER.warning("Auto calibrate: bos kare.")
            return
        sat = hsv_frame[:, :, 1].astype(np.float32).ravel()
        val = hsv_frame[:, :, 2].astype(np.float32).ravel()
        sat_min = int(np.percentile(sat, low_pct))
        sat_max = int(np.percentile(sat, high_pct))
        val_min = int(np.percentile(val, low_pct))
        val_max = int(np.percentile(val, high_pct))
        sat_min = max(0, min(255, sat_min))
        sat_max = max(0, min(255, sat_max))
        val_min = max(0, min(255, val_min))
        val_max = max(0, min(255, val_max))
        for thr in self.color_thresholds.values():
            thr.sat_min = sat_min
            thr.sat_max = sat_max
            thr.val_min = val_min
            thr.val_max = val_max
            thr.clamp()
        LOGGER.debug(
            "Auto threshold applied: sat [%s-%s], val [%s-%s]",
            sat_min,
            sat_max,
            val_min,
            val_max,
        )

    # ------------------------------------------------------------------ #
    # Autopilot (color approach + grasp)
    # ------------------------------------------------------------------ #
    def _mask_target(self, hsv_frame: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int, int]]]:
        thr = self.color_thresholds.get(self.auto_target_color)
        if thr is None:
            return np.zeros(hsv_frame.shape[:2], dtype=np.uint8), None
        lower = np.array([thr.hue_min, thr.sat_min, thr.val_min], dtype=np.uint8)
        upper = np.array([thr.hue_max, thr.sat_max, thr.val_max], dtype=np.uint8)
        mask = cv2.inRange(hsv_frame, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask, None
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return mask, None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return mask, (cx, cy, int(area))

    def _aim_arm(self, target: Tuple[int, int, int], frame_size: Tuple[int, int]) -> None:
        """
        Move arm joints toward the target based on its image position/size.
        Assumes camera is mounted above the arm; uses simple heuristics:
        - base tracks horizontal error
        - shoulder/elbow adjust with perceived area (distance proxy)
        """
        cx, _cy, area = target
        width, height = frame_size
        center_x = width // 2
        error_x = cx - center_x
        norm_error = max(-1.0, min(1.0, error_x / (width / 2)))

        # Base pans with horizontal error
        base_target = 90 + norm_error * 60  # 30 deg each side, limited to reasonable swing
        self.set_joint_angle("base", base_target)

        # Shoulder/elbow based on distance (area)
        # Larger area -> closer -> raise shoulder and bend elbow for grasp
        if area < 4000:
            shoulder_tgt = 70
            elbow_tgt = 100
        elif area < 12000:
            shoulder_tgt = 85
            elbow_tgt = 115
        else:
            shoulder_tgt = 105
            elbow_tgt = 130
        self.set_joint_angle("shoulder", shoulder_tgt)
        self.set_joint_angle("elbow", elbow_tgt)

    def autopilot_step(self, hsv_frame: np.ndarray, frame_size: Tuple[int, int]) -> None:
        """
        Basic color-follow and grasp routine.
        - Rotate/drive to search if no target.
        - Aim arm joints toward target; keep tracking even in simulation.
        - Approach when target centered.
        - Trigger grasp when close enough (area threshold).
        """
        if self.is_manual():
            return
        width, height = frame_size
        mask, target = self._mask_target(hsv_frame)
        self.last_target = target
        if target is None:
            # search: slow rotate left
            self._set_drive(turn=20.0, forward=0.0)
            return

        cx, cy, area = target
        center_x = width // 2
        error_x = cx - center_x
        turn = -error_x * 0.1  # simple P controller
        turn = max(-50.0, min(50.0, turn))

        # Aim the arm toward the target based on camera position
        self._aim_arm(target, (width, height))

        # Forward speed based on area (smaller area => further)
        if area < 5000:
            forward = 40.0
        elif area < 15000:
            forward = 25.0
        else:
            forward = 0.0
            if not self.auto_grasped:
                self._execute_grasp()
                self.auto_grasped = True
        self._set_drive(turn=turn, forward=forward)

    def _set_drive(self, turn: float, forward: float) -> None:
        # simple differential mix
        left = max(-100.0, min(100.0, forward + turn))
        right = max(-100.0, min(100.0, forward - turn))
        for wheel in self.wheels():
            val = left if "left" in wheel else right
            self.set_wheel_speed(wheel, val)

    def _execute_grasp(self) -> None:
        LOGGER.info("Grasp sequence triggered.")
        # simple preset: stop, close gripper
        self.stop_all_motion()
        # Basic joint moves (tune as needed for real hardware)
        self.arm_state["shoulder"] = 60.0
        self.arm_state["elbow"] = 120.0
        self.arm_state["wrist"] = 90.0
        self.arm_state["gripper"] = 10.0  # close
        if self.hardware_ready:  # pragma: no cover
            try:
                self.kit.servo[ARM_CHANNELS["shoulder"]].angle = 60
                self.kit.servo[ARM_CHANNELS["elbow"]].angle = 120
                self.kit.servo[ARM_CHANNELS["wrist"]].angle = 90
                self.kit.servo[ARM_CHANNELS["gripper"]].angle = 10
            except Exception as exc:
                LOGGER.error("Grasp servo hareketi uygulanamadi: %s", exc)

    # ------------------------------------------------------------------ #
    # Metrics helpers
    # ------------------------------------------------------------------ #
    def _recompute_power(self) -> None:
        # Very rough simulation of current draw based on wheel speeds
        wheel_factor = sum(abs(v) for v in self.wheel_state.values()) / (100.0 * max(len(self.wheel_state), 1))
        self.supply_current = 0.1 + 0.6 * wheel_factor  # amper
        self.power_consumption_w = self.supply_voltage * self.supply_current

    def get_metrics(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "simulation": self.simulation_enabled,
            "auto_target_color": self.auto_target_color,
            "auto_threshold": self.auto_threshold_enabled,
            "auto_grasped": self.auto_grasped,
            "supply_voltage": self.supply_voltage,
            "supply_current": self.supply_current,
            "power_w": self.power_consumption_w,
            "pwm_frequency_hz": self.pwm_frequency_hz,
            "pwm_outputs": dict(self.pwm_output),
        }
