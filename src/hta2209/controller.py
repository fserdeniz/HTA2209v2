"""
Hardware abstraction and simple autopilot for the HTA2209 platform.
"""
from __future__ import annotations

import json
import logging
import atexit
import time
from dataclasses import dataclass
from enum import Enum, auto
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

try:
    import RPi.GPIO as GPIO

    _GPIO_IMPORT_ERROR = None
except Exception as exc:  # Windows/dev host
    GPIO = None
    _GPIO_IMPORT_ERROR = exc

LOGGER = logging.getLogger(__name__)

WHEEL_CHANNELS = {
    # Motor kanallarini yukari tasiyarak kol (joint/gripper) ile cakismayi engelle
    "front_left": 9,
    "front_right": 10,
    "rear_left": 11,
    "rear_right": 12,
}

# Two-joint arm: one positional servo and one continuous (gripper)
# Varsayilan kanal eşleşmesi:
# - joint (180° servo): PCA9685 kanal 0
# - gripper (360° continuous servo): PCA9685 kanal 3
DEFAULT_ARM_CHANNELS = {
    "joint": 0,    # normal servo (0-180 deg)
    "gripper": 3,  # continuous rotation servo (open/close)
}

# L298N pinleri (BCM). ENA/ENB PWM, IN pinleri yön. Varsayilan; config ile override edilebilir.
DEFAULT_L298N_PINS = {
    "ena": 12,  # PWM0 (alternatif 18)
    "in1": 23,
    "in2": 24,
    "enb": 13,  # PWM1 (alternatif 19)
    "in3": 27,
    "in4": 22,
}

SUPPORTED_MODES = ("manual", "auto")
DEFAULT_COLORS = ("red", "green", "blue", "yellow", "orange", "purple", "cyan")


class AutopilotState(Enum):
    """Explicit states for the autonomous behavior."""
    IDLE = auto()
    SCANNING = auto()
    TRACKING = auto()
    APPROACHING = auto()
    GRASPING = auto()
    SEARCH_FAILED = auto()


class GuiLogHandler(logging.Handler):
    """Redirect controller logs to GUI via callback."""

    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def emit(self, record) -> None:  # pragma: no cover
        try:
            msg = self.format(record)
            self.callback(msg)
        except Exception:
            pass


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
        self.hbridge_ready = False
        self._ena_pwm = None
        self._enb_pwm = None
        self._gpio_initialized = False
        self.l298n_pins: Dict[str, int] = dict(DEFAULT_L298N_PINS)
        self.l298n_invert_right: bool = True

        self.wheel_state: Dict[str, float] = {wheel: 0.0 for wheel in WHEEL_CHANNELS}
        self.wheel_polarity: Dict[str, int] = {wheel: 1 for wheel in WHEEL_CHANNELS}
        # Gripper continuous-servo olarak % hiz, digerleri derece
        self.continuous_joints = {"gripper"}
        self.servo_channels: Dict[str, int] = dict(DEFAULT_ARM_CHANNELS)
        self.arm_state: Dict[str, float] = {
            joint: 0.0 if joint in self.continuous_joints else 90.0 for joint in self.servo_channels
        }
        default_thr = {
            "red": Threshold(0, 15, 80, 255, 80, 255),
            "orange": Threshold(5, 25, 80, 255, 80, 255),
            "yellow": Threshold(20, 40, 80, 255, 80, 255),
            "green": Threshold(45, 90, 80, 255, 80, 255),
            "cyan": Threshold(80, 100, 80, 255, 80, 255),
            "blue": Threshold(100, 140, 80, 255, 80, 255),
            "purple": Threshold(130, 165, 80, 255, 80, 255),
        }
        self.color_thresholds: Dict[str, Threshold] = {color: default_thr.get(color, Threshold()).clamp() for color in DEFAULT_COLORS}
        self.auto_threshold_enabled: bool = False
        self.auto_target_color: str = DEFAULT_COLORS[0]
        self.auto_grasped: bool = False
        self.mode: str = "manual"
        self.simulation_enabled: bool = False
        self.run_state: str = "stopped"  # started, paused, stopped
        self.last_target: Optional[Tuple[int, int, int, float, float, str]] = None  # (cx, cy, area, depth, mask_ratio, color)
        self.last_target_color: Optional[str] = None
        # Autopilot state machine
        self.autopilot_state: AutopilotState = AutopilotState.IDLE
        self.auto_state_started_at: float = 0.0
        self.auto_scan_step: int = 0
        self.auto_detect_hits: int = 0
        self.supply_voltage: float = 5.0
        self.supply_current: float = 0.0
        self.power_consumption_w: float = 0.0
        self.pwm_frequency_hz: float = 50.0
        self.pwm_output: Dict[str, float] = {wheel: 0.0 for wheel in WHEEL_CHANNELS}
        self.gripper_burst_until: float = 0.0
        self.gripper_reverse_until: float = 0.0
        self._target_was_visible: bool = False
        self.gripper_cooldown_until: float = 0.0
        self._last_auto_calib: float = 0.0
        self._coverage_smooth: float = 0.0
        self._last_seen_time: float = 0.0
        self._cx_smooth: float = 0.0
        self._cy_smooth: float = 0.0
        self._last_drive_time: float = 0.0
        self._last_drive_cmd: Tuple[float, float] = (0.0, 0.0)
        self._gripper_armed: bool = True
        self._gripper_fired: bool = False
        self._drive_turn_smooth: float = 0.0
        self._drive_fwd_smooth: float = 0.0
        self._scan_turning_until: float = 0.0
        self._scan_next_turn_at: float = 0.0
        self._blue_release_cooldown: float = 0.0
        self.autopilot_config: Dict[str, float] = {}

        # Once config yüklensin, sonra donanim baglansin (pinler config'ten gelsin)
        self.load_config()
        self._connect_to_hardware()
        # Donanim geldikten sonra tum cikislari sifirla
        self.stop_all_motion()
        # Uygulama kapanirken PWM'leri sifirla
        atexit.register(self.cleanup)
        # Donanim geldikten sonra mevcut eklem acilarini uygula
        if self.hardware_ready and not self.is_simulation():
            for joint, val in self.arm_state.items():
                if joint in self.continuous_joints:
                    self.set_continuous_speed(joint, val)
                else:
                    self._apply_joint_angle_to_hardware(joint, val)

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
            # Standard analog servolar icin güvenli darbe araligi ve frekans
            try:
                self.kit.frequency = 50
                # Baslangicta tum kanallari sifirla (onceki seanslardan kalan PWM'leri temizle)
                try:
                    for ch in range(self.channels):
                        self.kit._pca.channels[ch].duty_cycle = 0  # type: ignore[attr-defined]
                except Exception:
                    pass
                for joint, ch in self.servo_channels.items():
                    try:
                        if joint in self.continuous_joints:
                            srv = self.kit.continuous_servo[ch]
                            try:
                                srv.set_pulse_width_range(500, 2500)
                            except Exception:
                                pass
                            srv.throttle = 0
                        else:
                            srv = self.kit.servo[ch]
                            srv.set_pulse_width_range(500, 2500)
                    except Exception:
                        LOGGER.debug("Kanal %s icin darbe araligi ayarlanamadi.", ch)
            except Exception:
                pass
            self.hardware_ready = True
            LOGGER.info("PCA9685 icin ServoKit baslatildi.")
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("ServoKit baslatilamadi, simulasyon moduna geciliyor: %s", exc)
            self.hardware_ready = False

        # L298N baslat
        if GPIO is None:
            if _GPIO_IMPORT_ERROR:
                LOGGER.info("RPi.GPIO import edilemedi: %s. L298N pinleri devre disi.", _GPIO_IMPORT_ERROR)
            return
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            for key in ("in1", "in2", "in3", "in4", "ena", "enb"):
                pin = self.l298n_pins[key]
                if key.startswith("in"):
                    GPIO.setup(pin, GPIO.OUT)
                    GPIO.output(pin, GPIO.LOW)
                else:  # ena/enb
                    GPIO.setup(pin, GPIO.OUT)
            # PWM pinlerini ayarla
            self._ena_pwm = GPIO.PWM(self.l298n_pins["ena"], 1000)
            self._enb_pwm = GPIO.PWM(self.l298n_pins["enb"], 1000)
            self._ena_pwm.start(0)
            self._enb_pwm.start(0)
            self.hbridge_ready = True
            self._gpio_initialized = True
            LOGGER.info(
                "L298N pinleri hazirlandi (ENA=%s, ENB=%s, IN1=%s, IN2=%s, IN3=%s, IN4=%s).",
                self.l298n_pins["ena"],
                self.l298n_pins["enb"],
                self.l298n_pins["in1"],
                self.l298n_pins["in2"],
                self.l298n_pins["in3"],
                self.l298n_pins["in4"],
            )
        except Exception as exc:
            LOGGER.warning("L298N pinleri baslatilamadi: %s", exc)
            self.hbridge_ready = False

    # ------------------------------------------------------------------ #
    # State management helpers
    # ------------------------------------------------------------------ #
    def set_wheel_speed(self, wheel: str, speed_percentage: float) -> None:
        if wheel not in self.wheel_state:
            raise ValueError(f"Bilinmeyen tekerlek: {wheel}")
        clamped = max(-100.0, min(100.0, speed_percentage))
        if not self.is_running():
            clamped = 0.0
        self.wheel_state[wheel] = clamped
        throttle = (self.wheel_polarity.get(wheel, 1) * clamped) / 100.0
        self.pwm_output[wheel] = throttle
        LOGGER.info("Tekerlek %s hizi %.1f%%", wheel, clamped)
        if self.hbridge_ready and not self.is_simulation():  # pragma: no cover
            self._apply_wheels_to_hbridge()
        self._recompute_power()

    def stop_wheels(self) -> None:
        """Sadece tekerlekleri durdur (kol/gripper etkilenmez)."""
        for wheel in self.wheel_state:
            self.wheel_state[wheel] = 0.0
            self.pwm_output[wheel] = 0.0
        if self.hbridge_ready and not self.is_simulation():  # pragma: no cover
            self._apply_wheels_to_hbridge()
        self._recompute_power()

    def _apply_wheels_to_hbridge(self) -> None:
        """L298N uzerinden sol/sağ taraf hizlarini uygula."""
        if not self.hbridge_ready:
            return
        # Ortalama ile sol/sag hizini cikar, polariteyi uygula
        left_vals = [self.wheel_state[w] * self.wheel_polarity.get(w, 1) for w in self.wheel_state if "left" in w]
        right_vals = [self.wheel_state[w] * self.wheel_polarity.get(w, 1) for w in self.wheel_state if "right" in w]
        left = sum(left_vals) / max(1, len(left_vals))
        right = sum(right_vals) / max(1, len(right_vals))
        
        if not self.is_running():
            left = 0.0
            right = 0.0

        def drive_side(speed: float, pwm, pin_a: int, pin_b: int, side_name: str) -> None:
            duty = max(-100.0, min(100.0, speed))
            if abs(duty) < 1e-3:
                GPIO.output(pin_a, GPIO.LOW)
                GPIO.output(pin_b, GPIO.LOW)
                pwm.ChangeDutyCycle(0)
                return
            if duty >= 0:
                GPIO.output(pin_a, GPIO.HIGH)
                GPIO.output(pin_b, GPIO.LOW)
            else:
                GPIO.output(pin_a, GPIO.LOW)
                GPIO.output(pin_b, GPIO.HIGH)
            pwm.ChangeDutyCycle(abs(duty))
            LOGGER.debug("L298N %s duty %.1f", side_name, duty)

        try:
            drive_side(left, self._ena_pwm, self.l298n_pins["in1"], self.l298n_pins["in2"], "SOL")
            right_cmd = -right if self.l298n_invert_right else right
            drive_side(right_cmd, self._enb_pwm, self.l298n_pins["in3"], self.l298n_pins["in4"], "SAG")
        except Exception as exc:  # pragma: no cover
            LOGGER.error("L298N hiz uygulanamadi: %s", exc)

    def set_joint_angle(self, joint: str, angle: float) -> None:
        if joint not in self.arm_state:
            raise ValueError(f"Bilinmeyen eklem: {joint}")
        if joint in self.continuous_joints:
            # continuous servo: angle parami hiz yuzdesi gibi yorumlaniyor (-100..100)
            self.set_continuous_speed(joint, angle)
            return
        clamped = max(0.0, min(180.0, angle))
        self.arm_state[joint] = clamped
        LOGGER.info("Eklem %s -> %.1f deg (kanal %s)", joint, clamped, self.servo_channels.get(joint, "bilinmiyor"))
        if self.hardware_ready and not self.is_simulation():  # pragma: no cover
            self._apply_joint_angle_to_hardware(joint, clamped)

    def _apply_joint_angle_to_hardware(self, joint: str, angle: float) -> None:
        channel = self.servo_channels[joint]
        try:
            servo = self.kit.servo[channel]
            servo.angle = angle
        except Exception as exc:
            LOGGER.error("Eklem %s icin aci uygulanamadi: %s", joint, exc)

    def set_continuous_speed(self, joint: str, speed_percentage: float) -> None:
        if joint not in self.continuous_joints:
            raise ValueError(f"Bilinmeyen continuous eklem: {joint}")
        clamped = max(-100.0, min(100.0, speed_percentage))
        if not self.is_running():
            clamped = 0.0
        self.arm_state[joint] = clamped
        LOGGER.info("Continuous eklem %s -> %.1f%% (kanal %s)", joint, clamped, self.servo_channels.get(joint, "bilinmiyor"))
        if self.hardware_ready and not self.is_simulation():  # pragma: no cover
            self._apply_continuous_to_hardware(joint, clamped)

    def _apply_continuous_to_hardware(self, joint: str, speed: float) -> None:
        channel = self.servo_channels[joint]
        try:
            # Stopped/pasif durumda darbeyi tamamen kapat (duty=0) ki servo donmesin
            if not self.is_running() and speed == 0:
                self.kit._pca.channels[channel].duty_cycle = 0  # type: ignore[attr-defined]
                return
            servo = self.kit.continuous_servo[channel]
            if speed == 0:
                # Stop komutu: duty'yi 0'a cek, serbest birak
                self.kit._pca.channels[channel].duty_cycle = 0  # type: ignore[attr-defined]
            else:
                servo.throttle = speed / 100.0
        except Exception as exc:
            LOGGER.error("Continuous eklem %s icin throttle ayarlanamadi: %s", joint, exc)

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
            "run_state": self.run_state,
            "auto_threshold": self.auto_threshold_enabled,
            "auto_target_color": self.auto_target_color,
            "wheel_polarity": self.wheel_polarity,
            "wheels": self.wheel_state,
            "arm": self.arm_state,
            "servo_channels": self.servo_channels,
            "l298n_pins": dict(self.l298n_pins),
            "l298n_invert_right": self.l298n_invert_right,
            "autopilot": dict(self.autopilot_config),
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
                if joint in self.continuous_joints:
                    self.arm_state[joint] = max(-100.0, min(100.0, float(value)))
                else:
                    self.arm_state[joint] = max(0.0, min(180.0, float(value)))
        # Opsiyonel L298N pin konfigurasyonu
        l298_conf = raw.get("l298n_pins", {})
        for key, val in l298_conf.items():
            if key in self.l298n_pins:
                try:
                    self.l298n_pins[key] = int(val)
                except Exception:
                    pass
        if "l298n_invert_right" in raw:
            self.l298n_invert_right = bool(raw.get("l298n_invert_right"))
        # Opsiyonel servo kanal konfigurasyonu
        servo_ch = raw.get("servo_channels", {})
        for joint, ch in servo_ch.items():
            if joint in self.continuous_joints or joint == "joint":
                try:
                    self.servo_channels[joint] = int(ch)
                except Exception:
                    pass
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
        # Kaydedilen calisma durumu okunur ama guvenlik icin acilis her zaman durdurulmus olsun
        self.run_state = raw.get("run_state", "stopped")
        self.set_run_state("stopped")

        # Autopilot ayarlarını yükle
        self.autopilot_config = {
            "scan_initial_wait": 0.5,
            "scan_rotate_duration": 0.6,   # ~45° adim
            "scan_hold_duration": 1.4,     # 2 sn toplam dongu
            "scan_steps": 9999,            # surekli tarama
            "scan_turn_speed": 45.0,
            "approach_area_threshold": 0.1,
            "stop_area_threshold": 0.15,
            "backward_area_threshold": 0.2,
            "backward_speed": -6.0,
            "tracking_turn_gain": 20.0,
            "tracking_forward_gain": 50.0,
            "tracking_turn_speed_max": 15.0,
            "tracking_forward_speed_max": 15.0,
            "tracking_deadband": 0.1,
        }
        autopilot_conf = raw.get("autopilot", {})
        for key, val in autopilot_conf.items():
            if key in self.autopilot_config:
                try:
                    self.autopilot_config[key] = float(val)
                except (ValueError, TypeError):
                    LOGGER.warning("Autopilot config degeri gecersiz: %s=%s", key, val)

        # Pozisyonel eklemleri mevcut acida tut (servo salmasin)
        if self.hardware_ready and not self.is_simulation():
            for joint in self.arm_state:
                if joint in self.continuous_joints:
                    continue
                try:
                    self._apply_joint_angle_to_hardware(joint, self.arm_state[joint])
                except Exception:
                    pass

        self._recompute_power()
        LOGGER.info("Konfigurasyon %s dosyasindan yuklendi.", self.config_path)

    # ------------------------------------------------------------------ #
    # Convenience queries
    # ------------------------------------------------------------------ #
    def wheels(self) -> Tuple[str, ...]:
        return tuple(self.wheel_state.keys())

    def joints(self) -> Tuple[str, ...]:
        return tuple(self.arm_state.keys())

    def continuous(self) -> Tuple[str, ...]:
        return tuple(self.continuous_joints)

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
        self._reset_autopilot()

    def is_manual(self) -> bool:
        return self.mode == "manual"

    def is_simulation(self) -> bool:
        return self.simulation_enabled

    def set_simulation(self, enabled: bool) -> None:
        self.simulation_enabled = bool(enabled)
        LOGGER.info("Simulasyon modu %s.", "acik" if self.simulation_enabled else "kapali")

    # ------------------------------------------------------------------ #
    # Run state helpers
    # ------------------------------------------------------------------ #
    def set_run_state(self, state: str) -> None:
        normalized = state.lower()
        if normalized not in ("started", "paused", "stopped"):
            raise ValueError(f"Bilinmeyen calisma durumu: {state}")
        self.run_state = normalized
        if normalized in ("paused", "stopped"):
            self.stop_all_motion()
            self._reset_autopilot()
        if normalized == "stopped":
            self.auto_grasped = False
        LOGGER.info("Calisma durumu %s olarak ayarlandi.", normalized)

    def is_running(self) -> bool:
        return self.run_state == "started"

    def is_paused(self) -> bool:
        return self.run_state == "paused"

    def stop_all_motion(self) -> None:
        self.stop_wheels()
        for joint in self.continuous_joints:
            self.set_continuous_speed(joint, 0)
        self.gripper_burst_until = 0.0
        self.gripper_reverse_until = 0.0
        self._target_was_visible = False
        self.gripper_cooldown_until = 0.0
        self._last_auto_calib = 0.0
        self._coverage_smooth = 0.0
        self._last_seen_time = 0.0
        self._cx_smooth = 0.0
        self._cy_smooth = 0.0
        self._last_drive_time = 0.0
        self._last_drive_cmd = (0.0, 0.0)
        self._gripper_armed = True
        self._gripper_fired = False
        self._drive_turn_smooth = 0.0
        self._drive_fwd_smooth = 0.0
        self._scan_turning_until = 0.0
        self._scan_next_turn_at = 0.0
        self._blue_release_cooldown = 0.0
        self._recompute_power()
        # otomatik tarama sirasinda baslatilan hareketleri de temizle
        if self.hbridge_ready and GPIO is not None:
            try:
                GPIO.output(self.l298n_pins["in1"], GPIO.LOW)
                GPIO.output(self.l298n_pins["in2"], GPIO.LOW)
                GPIO.output(self.l298n_pins["in3"], GPIO.LOW)
                GPIO.output(self.l298n_pins["in4"], GPIO.LOW)
            except Exception:
                pass

    def cleanup(self) -> None:
        """PWM'leri kapat, GPIO'yu temizle."""
        try:
            self.stop_all_motion()
        except Exception:
            pass
        # PCA9685 tum kanallari sifirla
        if self.kit:
            try:
                for ch in range(self.channels):
                    try:
                        self.kit._pca.channels[ch].duty_cycle = 0  # type: ignore[attr-defined]
                    except Exception:
                        pass
            except Exception:
                pass
        if self._ena_pwm:
            try:
                self._ena_pwm.stop()
            except Exception:
                pass
        if self._enb_pwm:
            try:
                self._enb_pwm.stop()
            except Exception:
                pass
        if self._gpio_initialized:
            try:
                GPIO.cleanup()
            except Exception:
                pass

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
        # Ortam aydinligina gore (V ortalama / std) min-max belirle
        v_mean = float(np.mean(val))
        v_std = float(np.std(val))
        sat_min = float(np.percentile(sat, low_pct))
        sat_max = float(np.percentile(sat, high_pct))
        v_span = max(10.0, v_std * 1.5)
        val_min = v_mean - v_span
        val_max = v_mean + v_span

        sat_min = int(max(0.0, min(255.0, sat_min)))
        sat_max = int(max(0.0, min(255.0, sat_max)))
        if sat_max <= sat_min:
            sat_min = max(0, sat_min - 10)
            sat_max = min(255, sat_max + 10)
        if sat_max <= sat_min:
            sat_min, sat_max = 0, 255

        val_min = int(max(0.0, min(255.0, val_min)))
        val_max = int(max(0.0, min(255.0, val_max)))
        if val_max <= val_min:
            val_min = max(0, int(v_mean) - 10)
            val_max = min(255, int(v_mean) + 10)
        if val_max <= val_min:
            val_min, val_max = 0, 255
        for thr in self.color_thresholds.values():
            thr.sat_min = sat_min
            thr.sat_max = sat_max
            thr.val_min = val_min
            thr.val_max = val_max
            thr.clamp()
        LOGGER.debug(
            "Auto threshold (ambient) sat [%s-%s], val [%s-%s], Vmean=%.1f std=%.1f",
            sat_min,
            sat_max,
            val_min,
            val_max,
            v_mean,
            v_std,
        )

    # ------------------------------------------------------------------ #
    # Autopilot (color approach + grasp)
    # ------------------------------------------------------------------ #
    def _mask_target(self, hsv_frame: np.ndarray, color: str) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, float, float, str]]]:
        """Return mask and largest contour info for given color."""

        def find_contour(mask_img: np.ndarray) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, float, float, str]]]:
            contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            # Kucuk hedefler icin limit: en az %0.2 alan veya 300 piksel
            min_area = max(120.0, mask_img.shape[0] * mask_img.shape[1] * 0.002)
            if area < min_area:
                return None
            mask_ratio = float(cv2.countNonZero(mask_img)) / float(mask_img.shape[0] * mask_img.shape[1])
            # Maske orani cok dusukse ele (min %0.3)
            if mask_ratio < 0.003:
                return None
            x, y, w, h = cv2.boundingRect(largest)
            rect_area = max(1.0, float(w * h))
            solidity = area / rect_area
            aspect = w / max(1.0, h)
            peri = cv2.arcLength(largest, True)
            circularity = 0.0 if peri == 0 else 4.0 * np.pi * (area / (peri * peri))
            # Balon hedefler için: dolgun (solidity), yuvarlak (aspect ~1, circularity yüksek) olmasını bekle
            if solidity < 0.20:
                return None
            if aspect < 0.35 or aspect > 2.8:
                return None
            if circularity < 0.20:
                return None
            M = cv2.moments(largest)
            if M["m00"] == 0:
                cx = int(x + w / 2)
                cy = int(y + h / 2)
            else:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            depth_norm = 1.0 / max(1.0, min(area, mask_img.shape[0] * mask_img.shape[1]))
            depth_norm = max(0.0, min(1.0, depth_norm * 1000))
            return mask_img, (cx, cy, int(area), depth_norm, float(mask_ratio), color)

        thr = self.color_thresholds.get(color)
        if thr is None:
            return np.zeros(hsv_frame.shape[:2], dtype=np.uint8), None
        ranges = [(thr.hue_min, thr.hue_max)]
        # Hue sarmalaması: kullanici min>max verirse veya kirmizi kenarlara gelirse iki bant oluştur
        if thr.hue_min > thr.hue_max:
            ranges = [(thr.hue_min, 179), (0, thr.hue_max)]
        elif color == "red":
            if thr.hue_min <= 10:
                ranges.append((170, 179))
            elif thr.hue_max >= 170:
                ranges.append((0, 10))

        mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
        for h_min, h_max in ranges:
            lower = np.array([h_min, thr.sat_min, thr.val_min], dtype=np.uint8)
            upper = np.array([h_max, thr.sat_max, thr.val_max], dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv_frame, lower, upper))
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        result = find_contour(mask)
        if result:
            mask_img, target = result
            # Ek renk dogrulama: median hue/sat/val kontrolü, yalanci pozitifleri eleyelim
            try:
                masked_pixels = hsv_frame[mask_img > 0]
                if masked_pixels.size == 0:
                    return mask, None
                h_med = float(np.median(masked_pixels[:, 0]))
                s_med = float(np.median(masked_pixels[:, 1]))
                v_med = float(np.median(masked_pixels[:, 2]))
                sat_floor = max(thr.sat_min, 70)
                val_floor = max(thr.val_min, 50)
                hue_diff = None
                hue_span = None
                for h_min, h_max in ranges:
                    center = (h_min + h_max) / 2.0
                    span = (h_max - h_min) / 2.0 + 5.0  # hue toleransi
                    diff = min(abs(h_med - center), 180 - abs(h_med - center))
                    if hue_diff is None or diff < hue_diff:
                        hue_diff = diff
                        hue_span = span
                if hue_diff is None or hue_span is None:
                    return mask, None
                if s_med < sat_floor or v_med < val_floor or hue_diff > hue_span:
                    LOGGER.debug(
                        "Maske reddedildi: hue_med=%.1f sat_med=%.1f val_med=%.1f (sat>=%.0f, val>=%.0f, hue_diff<=%.1f)",
                        h_med,
                        s_med,
                        v_med,
                        sat_floor,
                        val_floor,
                        hue_span,
                    )
                    return mask, None
                return mask_img, target
            except Exception:
                return mask_img, target
        return mask, None

    def _aim_arm(self, target: Tuple[int, int, int, float, float, str], frame_size: Tuple[int, int]) -> None:
        """
        Move single positional joint toward target (horizontal pan); continuous gripper is not auto-driven here.
        """
        if "joint" not in self.arm_state:
            return
        cx = int(target[0])
        width, _height = frame_size
        center_x = width // 2
        error_x = cx - center_x
        norm_error = max(-1.0, min(1.0, error_x / (width / 2)))

        joint_target = 90 + norm_error * 60  # ±60 deg
        self.set_joint_angle("joint", joint_target)

    def _find_best_target(self, hsv_frame: np.ndarray, frame_size: Tuple[int, int]) -> Optional[Tuple]:
        """Find the best target in the frame by iterating through allowed colors."""
        # Yalnızca sarı ve kırmızı takip edilebilir; mavi tamamen devre dışı
        if self.auto_target_color in ("yellow", "red"):
            allowed_colors = (self.auto_target_color,)
        else:
            allowed_colors = ("yellow", "red")

        best_target = None
        best_conf = -1.0
        frame_area = max(1, int(frame_size[0] * frame_size[1]))
        center_x = frame_size[0] // 2

        for color in allowed_colors:
            _mask, candidate = self._mask_target(hsv_frame, color)
            if candidate is None:
                continue
            cx, cy, area, depth_norm, mask_ratio, _col = candidate
            area_norm = area / frame_area
            center_score = max(0.0, 1.0 - abs(cx - center_x) / max(1.0, center_x))
            conf = area_norm * 0.6 + mask_ratio * 0.8 + center_score * 0.4
            if conf > best_conf:
                best_conf = conf
                best_target = candidate
        
        if best_target:
            return best_target
        return None

    def _reset_autopilot(self) -> None:
        self.autopilot_state = AutopilotState.IDLE
        self.auto_state_started_at = 0.0
        self.auto_scan_step = 0
        self.auto_detect_hits = 0
        self.last_target = None
        self.last_target_color = None
        self.gripper_burst_until = 0.0
        self.gripper_reverse_until = 0.0
        self._target_was_visible = False
        self.gripper_cooldown_until = 0.0
        self._last_auto_calib = 0.0
        self._coverage_smooth = 0.0
        self._last_seen_time = 0.0
        self._cx_smooth = 0.0
        self._cy_smooth = 0.0
        self._drive_turn_smooth = 0.0
        self._drive_fwd_smooth = 0.0
        self._scan_turning_until = 0.0
        self._scan_next_turn_at = 0.0
        self._blue_release_cooldown = 0.0

    def _trigger_blue_release(self, now: float) -> bool:
        """Open gripper when blue balloon is seen."""
        if now < self._blue_release_cooldown:
            return False
        self.stop_all_motion()
        try:
            self.set_continuous_speed("gripper", -30.0)
        except Exception:
            pass
        self.gripper_reverse_until = now + 3.0
        self._blue_release_cooldown = now + 6.0
        LOGGER.info("Auto: Blue balloon detected, releasing grip.")
        return True

    def autopilot_step(self, hsv_frame: np.ndarray, frame_size: Tuple[int, int]) -> None:
        """
        Main autopilot state machine.
        """
        if self.is_manual() or not self.is_running():
            if self.autopilot_state != AutopilotState.IDLE:
                self._reset_autopilot()
                self.stop_all_motion()
            self.last_target = None
            self.last_target_color = None
            return

        now = time.monotonic()
        if self.gripper_reverse_until and now >= self.gripper_reverse_until:
            self.gripper_reverse_until = 0.0
            try:
                self.set_continuous_speed("gripper", 0.0)
            except Exception:
                pass
        if self.gripper_burst_until and now >= self.gripper_burst_until:
            self.gripper_burst_until = 0.0
            try:
                self.set_continuous_speed("gripper", 0.0)
            except Exception:
                pass

        # Frame pre-processing
        if hsv_frame is None or hsv_frame.size == 0:
            LOGGER.debug("Auto: empty frame, skipping step.")
            return
        hsv_proc = cv2.GaussianBlur(hsv_frame, (3, 3), 0)

        # Periodic auto-calibration
        if self.auto_threshold_enabled and now - self._last_auto_calib >= 5.0:
            self.auto_calibrate_from_frame(hsv_frame)
            self._last_auto_calib = now
            LOGGER.debug("Auto: periodic color calibration applied.")

        best_target = self._find_best_target(hsv_proc, frame_size)
        _blue_mask, blue_candidate = self._mask_target(hsv_proc, "blue")
        if best_target:
            self.last_target = best_target
            self.last_target_color = best_target[5] if len(best_target) > 5 else None
        else:
            self.last_target = None
            self.last_target_color = None

        # --- State Machine ---
        state = self.autopilot_state

        if state == AutopilotState.IDLE:
            self.stop_all_motion()
            if blue_candidate and self._trigger_blue_release(now):
                return
            if best_target:
                LOGGER.info("Auto: Target found while idle, switching to TRACKING.")
                self.autopilot_state = AutopilotState.TRACKING
            else:
                LOGGER.info("Auto: No target, switching to SCANNING.")
                self.autopilot_state = AutopilotState.SCANNING
                self.auto_state_started_at = now
                self.auto_scan_step = 0
            return

        if state == AutopilotState.SCANNING:
            cfg = self.autopilot_config
            if blue_candidate and self._trigger_blue_release(now):
                return
            if self.auto_scan_step == 0 and now - self.auto_state_started_at < cfg["scan_initial_wait"]:
                # Initial wait before starting the scan
                self.stop_all_motion()
                return

            if best_target:
                LOGGER.info("Auto: Target found during scan, switching to TRACKING.")
                self.stop_all_motion()
                self._reset_autopilot() # Reset timers and smoothers
                self.autopilot_state = AutopilotState.TRACKING
                return

            time_in_state = now - self.auto_state_started_at
            
            # Rotate -> Hold cycle
            if time_in_state < cfg["scan_rotate_duration"]:
                self._set_drive(turn=cfg["scan_turn_speed"], forward=0.0)
            elif time_in_state < cfg["scan_rotate_duration"] + cfg["scan_hold_duration"]:
                self.stop_wheels()
            else:
                # End of hold, start next step
                self.auto_scan_step += 1
                self.auto_state_started_at = now
                if self.auto_scan_step >= cfg["scan_steps"]:
                    # Devamli tarama: adimi sifirla, donusleri surdur
                    self.auto_scan_step = 0
                    LOGGER.info("Auto: Scan loop restarting (continuous).")
                else:
                    LOGGER.info(f"Auto: Scan step {self.auto_scan_step}/{cfg['scan_steps']}.")
            return

        if state == AutopilotState.TRACKING:
            if not best_target:
                LOGGER.info("Auto: Target lost during tracking. Returning to IDLE.")
                self.autopilot_state = AutopilotState.IDLE
                self.stop_all_motion()
                return
            if blue_candidate and self._trigger_blue_release(now):
                self.autopilot_state = AutopilotState.IDLE
                return

            cx, cy, area, depth_norm, mask_ratio, color = best_target
            
            # Smooth the target position
            alpha = 0.4
            if self._cx_smooth == 0: self._cx_smooth = cx
            else: self._cx_smooth = alpha * self._cx_smooth + (1 - alpha) * cx

            center_x = frame_size[0] // 2
            error_x = self._cx_smooth - center_x
            norm_err = max(-1.0, min(1.0, error_x / (frame_size[0] / 2)))

            # --- Motion control ---
            cfg = self.autopilot_config
            if abs(norm_err) < cfg["tracking_deadband"]:
                norm_err = 0.0

            turn_cmd = max(-cfg["tracking_turn_speed_max"], min(cfg["tracking_turn_speed_max"], -norm_err * cfg["tracking_turn_gain"]))
            
            # Forward/backward control based on area
            frame_area = frame_size[0] * frame_size[1]
            normalized_area = area / frame_area

            fwd_cmd = 0.0
            if normalized_area > cfg["backward_area_threshold"]:
                fwd_cmd = cfg["backward_speed"]
                LOGGER.info("Auto: Target too close, moving backward.")
            elif normalized_area > cfg["stop_area_threshold"]:
                fwd_cmd = 0.0
                LOGGER.info("Auto: Target at stop distance, holding position (no grasp).")
                # Gripper yok: yaklaşınca sadece dur, TRACKING'de kal
                self._set_drive(turn=0.0, forward=0.0)
                return
            elif normalized_area < cfg["approach_area_threshold"]:
                 fwd_cmd = min(cfg["tracking_forward_speed_max"], max(5.0, (cfg["approach_area_threshold"] - normalized_area) * cfg["tracking_forward_gain"]))

            self._drive_turn_smooth = 0.7 * self._drive_turn_smooth + 0.3 * turn_cmd
            self._drive_fwd_smooth = 0.8 * self._drive_fwd_smooth + 0.2 * fwd_cmd
            
            self._set_drive(turn=self._drive_turn_smooth, forward=self._drive_fwd_smooth)
            LOGGER.debug(f"Auto TRACKING: Target={color}, Area={normalized_area:.3f}, Fwd={self._drive_fwd_smooth:.1f}, Turn={self._drive_turn_smooth:.1f}")
            return

        if state in (AutopilotState.APPROACHING, AutopilotState.GRASPING):
            self.stop_all_motion()
            self.autopilot_state = AutopilotState.TRACKING
            return


        if state == AutopilotState.SEARCH_FAILED:
            # Do nothing, wait for a reset
            self.stop_all_motion()
            return

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
        # Sadece mevcut eklemleri kullanarak basit bir kapama profili
        if "joint" in self.arm_state:
            self.set_joint_angle("joint", 60)
        if "gripper" in self.arm_state:
            self.set_continuous_speed("gripper", -40.0)  # kapama yonu (ayarlanabilir)

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
            "run_state": self.run_state,
            "auto_target_color": self.auto_target_color,
            "auto_threshold": self.auto_threshold_enabled,
            "auto_grasped": self.auto_grasped,
            "supply_voltage": self.supply_voltage,
            "supply_current": self.supply_current,
            "power_w": self.power_consumption_w,
            "pwm_frequency_hz": self.pwm_frequency_hz,
            "pwm_outputs": dict(self.pwm_output),
        }
