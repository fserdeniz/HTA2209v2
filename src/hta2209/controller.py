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
        self.last_target: Optional[Tuple[int, int, int]] = None  # (cx, cy, area)
        self.supply_voltage: float = 5.0
        self.supply_current: float = 0.0
        self.power_consumption_w: float = 0.0
        self.pwm_frequency_hz: float = 50.0
        self.pwm_output: Dict[str, float] = {wheel: 0.0 for wheel in WHEEL_CHANNELS}

        # Once config yüklensin, sonra donanim baglansin (pinler config'ten gelsin)
        self.load_config()
        self._connect_to_hardware()
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
        if self.hbridge_ready and not self.is_simulation():  # pragma: no cover
            self._apply_wheels_to_hbridge()
        self._recompute_power()

    def _apply_wheels_to_hbridge(self) -> None:
        """L298N uzerinden sol/sağ taraf hizlarini uygula."""
        if not self.hbridge_ready:
            return
        # Ortalama ile sol/sag hizini cikar
        left_vals = [self.wheel_state[w] for w in self.wheel_state if "left" in w]
        right_vals = [self.wheel_state[w] for w in self.wheel_state if "right" in w]
        left = sum(left_vals) / max(1, len(left_vals))
        right = sum(right_vals) / max(1, len(right_vals))
        # Run_state kontrolu: stopped/paused ise hizlari sifirla
        if not self.is_running():
            left = 0.0
            right = 0.0

        def drive_side(speed: float, pwm, pin_a: int, pin_b: int, side_name: str, invert: bool = False) -> None:
            duty = max(-100.0, min(100.0, speed))
            if duty >= 0:
                GPIO.output(pin_a, GPIO.LOW if invert else GPIO.HIGH)
                GPIO.output(pin_b, GPIO.HIGH if invert else GPIO.LOW)
            else:
                GPIO.output(pin_a, GPIO.HIGH if invert else GPIO.LOW)
                GPIO.output(pin_b, GPIO.LOW if invert else GPIO.HIGH)
            pwm.ChangeDutyCycle(abs(duty))
            LOGGER.info(
                "L298N %s duty %.1f -> pins (%s=%s, %s=%s)",
                side_name,
                duty,
                pin_a,
                int(duty >= 0),
                pin_b,
                int(duty < 0),
            )

        try:
            drive_side(left, self._ena_pwm, self.l298n_pins["in1"], self.l298n_pins["in2"], "SOL", invert=False)
            drive_side(right, self._enb_pwm, self.l298n_pins["in3"], self.l298n_pins["in4"], "SAG", invert=True)
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
        self._recompute_power()

    def cleanup(self) -> None:
        """PWM'leri kapat, GPIO'yu temizle."""
        try:
            self.stop_all_motion()
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
    def _mask_target(self, hsv_frame: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, float]]]:
        thr = self.color_thresholds.get(self.auto_target_color)
        if thr is None:
            return np.zeros(hsv_frame.shape[:2], dtype=np.uint8), None
        lower = np.array([thr.hue_min, thr.sat_min, thr.val_min], dtype=np.uint8)
        upper = np.array([thr.hue_max, thr.sat_max, thr.val_max], dtype=np.uint8)
        mask = cv2.inRange(hsv_frame, lower, upper)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask, None
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        min_area = max(150.0, mask.shape[0] * mask.shape[1] * 0.002)
        if area < min_area:
            return mask, None
        M = cv2.moments(largest)
        if M["m00"] == 0:
            x, y, w, h = cv2.boundingRect(largest)
            cx = int(x + w / 2)
            cy = int(y + h / 2)
        else:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        # Basit derinlik tahmini: normalize alan tersine orantili (0-1 arasi)
        depth_norm = 1.0 / max(1.0, min(area, mask.shape[0] * mask.shape[1]))
        depth_norm = max(0.0, min(1.0, depth_norm * 1000))  # basit ölçekleme
        return mask, (cx, cy, int(area), depth_norm)

    def _aim_arm(self, target: Tuple[int, int, int, float], frame_size: Tuple[int, int]) -> None:
        """
        Move single positional joint toward target (horizontal pan); continuous gripper is not auto-driven here.
        """
        if "joint" not in self.arm_state:
            return
        cx, _cy, _area, _depth = target
        width, _height = frame_size
        center_x = width // 2
        error_x = cx - center_x
        norm_error = max(-1.0, min(1.0, error_x / (width / 2)))

        joint_target = 90 + norm_error * 60  # ±60 deg
        self.set_joint_angle("joint", joint_target)

    def autopilot_step(self, hsv_frame: np.ndarray, frame_size: Tuple[int, int]) -> None:
        """
        Basic color-follow and grasp routine.
        - Rotate/drive to search if no target.
        - Aim arm joints toward target; keep tracking even in simulation.
        - Approach when target centered.
        - Trigger grasp when close enough (area threshold).
        """
        if self.is_manual() or not self.is_running():
            return
        width, height = frame_size
        mask, target = self._mask_target(hsv_frame)
        if target and self.last_target:
            prev_cx, prev_cy, _, prev_depth = self.last_target
            cx, cy, area, depth = target
            cx = int(0.6 * prev_cx + 0.4 * cx)
            cy = int(0.6 * prev_cy + 0.4 * cy)
            depth = 0.6 * prev_depth + 0.4 * depth
            target = (cx, cy, area, depth)
        self.last_target = target
        if target is None:
            # search: slow rotate left
            self._set_drive(turn=20.0, forward=0.0)
            return

        cx, cy, area, _depth = target
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
