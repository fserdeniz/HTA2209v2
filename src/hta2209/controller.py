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
class RobotController:
    """
    PCA9685 channel abstraction and YOLO-driven autopilot control.
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
        self.l298n_swap_sides: bool = False
        self.drive_invert: bool = False
        self.camera_swap_rb: bool = False
        self.camera_source: str = "picamera2"
        self.camera_index: int = 0
        self.camera_frame_width: int = 640
        self.camera_frame_height: int = 640
        self.camera_controls: Dict[str, int] = {
            "brightness": 10,
            "contrast": 0,
            "saturation": 60,
            "hue": 0,
            "gain": 122,
        }

        self.wheel_state: Dict[str, float] = {wheel: 0.0 for wheel in WHEEL_CHANNELS}
        self.wheel_polarity: Dict[str, int] = {wheel: 1 for wheel in WHEEL_CHANNELS}
        # Gripper continuous-servo olarak % hiz, digerleri derece
        self.continuous_joints = {"gripper"}
        self.servo_channels: Dict[str, int] = dict(DEFAULT_ARM_CHANNELS)
        self.arm_state: Dict[str, float] = {
            joint: 0.0 if joint in self.continuous_joints else 90.0 for joint in self.servo_channels
        }
        self.auto_target_color: str = "red"
        self.auto_forward_invert: bool = False
        self.mode: str = "manual"
        self.simulation_enabled: bool = False
        self.run_state: str = "stopped"  # started, paused, stopped
        self.last_target: Optional[Tuple[int, int, int, float, float, str]] = None  # (cx, cy, area, depth, mask_ratio, color)
        self.last_target_color: Optional[str] = None
        # Autopilot state machine
        self.autopilot_state: AutopilotState = AutopilotState.IDLE
        self.auto_state_started_at: float = 0.0
        self.auto_scan_step: int = 0
        self.supply_voltage: float = 5.0
        self.supply_current: float = 0.0
        self.power_consumption_w: float = 0.0
        self.pwm_frequency_hz: float = 50.0
        self.pwm_output: Dict[str, float] = {wheel: 0.0 for wheel in WHEEL_CHANNELS}
        self._area_smooth: float = 0.0
        self._cx_smooth: float = 0.0
        self._cy_smooth: float = 0.0
        self._drive_turn_smooth: float = 0.0
        self._drive_fwd_smooth: float = 0.0
        self._target_hold_until: float = 0.0
        self._target_smooth_color: Optional[str] = None
        self._target_raw_pos: Optional[Tuple[int, int]] = None
        self._target_raw_color: Optional[str] = None
        self._tracking_zone: int = 0
        self._step_phase: str = "idle"
        self._step_until: float = 0.0
        self._step_direction: int = 0
        self._turn_error_prev: float = 0.0
        self._align_hold_started: float = 0.0
        self._turn_phase: str = "idle"
        self._turn_until: float = 0.0
        self._turn_direction: int = 0
        self._detect_hold_until: float = 0.0
        self._last_target_seen_at: float = 0.0
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
            left_cmd = left
            right_cmd = -right if self.l298n_invert_right else right
            if self.l298n_swap_sides:
                left_cmd, right_cmd = right_cmd, left_cmd
            drive_side(left_cmd, self._ena_pwm, self.l298n_pins["in1"], self.l298n_pins["in2"], "SOL")
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

    # ------------------------------------------------------------------ #
    # Persistence helpers
    # ------------------------------------------------------------------ #
    def serialize_state(self) -> Dict[str, Dict[str, float]]:
        return {
            "mode": self.mode,
            "simulation": self.simulation_enabled,
            "run_state": self.run_state,
            "auto_target_color": self.auto_target_color,
            "auto_forward_invert": self.auto_forward_invert,
            "wheel_polarity": self.wheel_polarity,
            "wheels": self.wheel_state,
            "arm": self.arm_state,
            "servo_channels": self.servo_channels,
            "l298n_pins": dict(self.l298n_pins),
            "l298n_invert_right": self.l298n_invert_right,
            "l298n_swap_sides": self.l298n_swap_sides,
            "drive_invert": self.drive_invert,
            "camera_swap_rb": self.camera_swap_rb,
            "camera": {
                "source": self.camera_source,
                "index": self.camera_index,
                "frame_width": self.camera_frame_width,
                "frame_height": self.camera_frame_height,
                "controls": dict(self.camera_controls),
            },
            "autopilot": dict(self.autopilot_config),
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
        if "l298n_swap_sides" in raw:
            self.l298n_swap_sides = bool(raw.get("l298n_swap_sides"))
        if "drive_invert" in raw:
            self.drive_invert = bool(raw.get("drive_invert"))
        if "camera_swap_rb" in raw:
            self.camera_swap_rb = bool(raw.get("camera_swap_rb"))
        camera_conf = raw.get("camera", {})
        if isinstance(camera_conf, dict):
            source = camera_conf.get("source")
            if isinstance(source, str) and source.lower() in ("picamera2", "opencv"):
                self.camera_source = source.lower()
            if "index" in camera_conf:
                try:
                    self.camera_index = int(camera_conf.get("index"))
                except (TypeError, ValueError):
                    pass
            if "frame_width" in camera_conf:
                try:
                    self.camera_frame_width = int(camera_conf.get("frame_width"))
                except (TypeError, ValueError):
                    pass
            if "frame_height" in camera_conf:
                try:
                    self.camera_frame_height = int(camera_conf.get("frame_height"))
                except (TypeError, ValueError):
                    pass
            controls = camera_conf.get("controls", {})
            if isinstance(controls, dict):
                for key in self.camera_controls:
                    if key in controls:
                        try:
                            self.camera_controls[key] = int(controls[key])
                        except (TypeError, ValueError):
                            pass
        # Opsiyonel servo kanal konfigurasyonu
        servo_ch = raw.get("servo_channels", {})
        for joint, ch in servo_ch.items():
            if joint in self.continuous_joints or joint == "joint":
                try:
                    self.servo_channels[joint] = int(ch)
                except Exception:
                    pass
        mode_name = raw.get("mode", "manual")
        # eski konfigurasyonlarda simulation bir mode olarak gelebilir
        if mode_name.lower() == "simulation":
            self.simulation_enabled = True
            mode_name = "manual"
        try:
            self.set_mode(mode_name)
        except ValueError:
            LOGGER.warning("Bilinmeyen kontrol modu %s, manuel mod kullanilacak.", mode_name)
        target = raw.get("auto_target_color", "red")
        if isinstance(target, str) and target:
            self.auto_target_color = target
        self.auto_forward_invert = bool(raw.get("auto_forward_invert", self.auto_forward_invert))
        self.simulation_enabled = bool(raw.get("simulation", self.simulation_enabled))
        # Kaydedilen calisma durumu okunur ama guvenlik icin acilis her zaman durdurulmus olsun
        self.run_state = raw.get("run_state", "stopped")
        self.set_run_state("stopped")

        # Autopilot ayarlarını yükle
        self.autopilot_config = {
            "scan_initial_wait": 3.0,
            "scan_rotate_duration": 0.6,   # ~45° adim (sureye bagli)
            "scan_hold_duration": 3.0,
            "scan_steps": 9999,            # surekli tarama
            "scan_turn_speed": 30.0,
            "approach_area_threshold": 0.35,
            "stop_area_threshold": 0.35,
            "target_area_ratio": 0.35,
            "area_tolerance_ratio": 0.0,
            "tracking_turn_speed_max": 10.0,
            "tracking_forward_speed_max": 15.0,
            "target_speed_scale": 0.02,
            "target_turn_scale": 0.01,
            "target_forward_min": 4.0,
            "align_center_tol_ratio": 0.08,
            "align_hold_sec": 0.3,
            "detect_hold_sec": 0.2,
            "lost_hold_sec": 1.5,
            "turn_kp": 1.0,
            "turn_kd": 0.3,
            "turn_smooth_alpha": 0.6,
            "turn_step_sec": 0.2,
            "turn_pause_sec": 0.2,
            "close_stop_ratio": 0.6,
            "step_move_sec": 0.4,
            "step_pause_sec": 0.35,
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
        LOGGER.info("Calisma durumu %s olarak ayarlandi.", normalized)

    def is_running(self) -> bool:
        return self.run_state == "started"

    def is_paused(self) -> bool:
        return self.run_state == "paused"

    def stop_all_motion(self) -> None:
        self.stop_wheels()
        for joint in self.continuous_joints:
            self.set_continuous_speed(joint, 0)
        self._area_smooth = 0.0
        self._cx_smooth = 0.0
        self._cy_smooth = 0.0
        self._drive_turn_smooth = 0.0
        self._drive_fwd_smooth = 0.0
        self._target_hold_until = 0.0
        self._target_smooth_color = None
        self._target_raw_pos = None
        self._target_raw_color = None
        self._step_phase = "idle"
        self._step_until = 0.0
        self._step_direction = 0
        self._turn_error_prev = 0.0
        self._align_hold_started = 0.0
        self._turn_phase = "idle"
        self._turn_until = 0.0
        self._turn_direction = 0
        self._detect_hold_until = 0.0
        self._last_target_seen_at = 0.0
        self._tracking_zone = 0
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

    def _reset_autopilot(self) -> None:
        self.autopilot_state = AutopilotState.IDLE
        self.auto_state_started_at = 0.0
        self.auto_scan_step = 0
        self.last_target = None
        self.last_target_color = None
        self._area_smooth = 0.0
        self._cx_smooth = 0.0
        self._cy_smooth = 0.0
        self._drive_turn_smooth = 0.0
        self._drive_fwd_smooth = 0.0
        self._target_hold_until = 0.0
        self._target_smooth_color = None
        self._target_raw_pos = None
        self._target_raw_color = None

    def autopilot_step(
        self,
        target: Optional[Tuple[int, int, int, float, float, str]],
        frame_size: Tuple[int, int],
    ) -> None:
        """Main autopilot state machine using only external (YOLO) targets."""
        if self.is_manual() or not self.is_running():
            if self.autopilot_state != AutopilotState.IDLE:
                self._reset_autopilot()
                self.stop_all_motion()
            self.last_target = None
            self.last_target_color = None
            return

        now = time.monotonic()
        raw_target = target
        best_target = target
        if raw_target is not None:
            self._last_target_seen_at = now
        if best_target:
            cx, cy, area, depth_norm, mask_ratio, color = best_target
            self._target_raw_pos = (cx, cy)
            self._target_raw_color = color
            if self._target_smooth_color != color:
                self._cx_smooth = float(cx)
                self._cy_smooth = float(cy)
                self._target_smooth_color = color
            else:
                smooth_alpha = 0.6
                self._cx_smooth = smooth_alpha * self._cx_smooth + (1 - smooth_alpha) * cx
                self._cy_smooth = smooth_alpha * self._cy_smooth + (1 - smooth_alpha) * cy
            smoothed_target = (
                int(round(self._cx_smooth)),
                int(round(self._cy_smooth)),
                area,
                depth_norm,
                mask_ratio,
                color,
            )
            best_target = smoothed_target
            self.last_target = smoothed_target
            self.last_target_color = color
            self._target_hold_until = now + 0.6
        else:
            if self.last_target is not None and now <= self._target_hold_until:
                best_target = self.last_target
            else:
                self.last_target = None
                self.last_target_color = None
                self._target_smooth_color = None
                self._target_raw_pos = None
                self._target_raw_color = None

        # --- State Machine ---
        state = self.autopilot_state
        cfg = self.autopilot_config

        if state == AutopilotState.IDLE:
            if self.auto_state_started_at == 0.0:
                self.auto_state_started_at = now
            self.stop_all_motion()
            if now - self.auto_state_started_at < cfg["scan_initial_wait"]:
                return
            self._step_phase = "idle"
            self._step_direction = 0
            if best_target:
                LOGGER.info("Auto: Initial wait done, target found -> TRACKING.")
                self.autopilot_state = AutopilotState.TRACKING
                hold = max(0.0, float(cfg.get("detect_hold_sec", 0.0)))
                self._detect_hold_until = now + hold if hold > 0.0 else 0.0
            else:
                LOGGER.info("Auto: Initial wait done, target yok -> SCANNING.")
                self.autopilot_state = AutopilotState.SCANNING
            self.auto_state_started_at = now
            return

        if state == AutopilotState.SCANNING:
            if best_target:
                LOGGER.info("Auto: Target found during scan -> TRACKING.")
                self.autopilot_state = AutopilotState.TRACKING
                self.auto_state_started_at = now
                self._step_phase = "idle"
                self._step_direction = 0
                hold = max(0.0, float(cfg.get("detect_hold_sec", 0.0)))
                self._detect_hold_until = now + hold if hold > 0.0 else 0.0
                self.stop_wheels()
                return

            rotate_dur = max(0.05, float(cfg["scan_rotate_duration"]))
            hold_dur = max(0.0, float(cfg["scan_hold_duration"]))
            cycle = rotate_dur + hold_dur
            if cycle <= 0.0:
                self.stop_wheels()
                return
            phase = (now - self.auto_state_started_at) % cycle
            if phase < rotate_dur:
                self._set_drive(turn=cfg["scan_turn_speed"], forward=0.0)
            else:
                self.stop_wheels()
            return

        if state == AutopilotState.TRACKING:
            if raw_target is None:
                lost_hold = max(0.0, float(cfg.get("lost_hold_sec", 0.0)))
                if self._last_target_seen_at and now - self._last_target_seen_at <= lost_hold:
                    self.stop_wheels()
                    return
                LOGGER.info("Auto: Target lost -> SCANNING.")
                self.autopilot_state = AutopilotState.SCANNING
                self.auto_state_started_at = now
                self._step_phase = "idle"
                self._step_direction = 0
                self._turn_error_prev = 0.0
                self._drive_turn_smooth = 0.0
                self._align_hold_started = 0.0
                self._turn_phase = "idle"
                self._turn_direction = 0
                self._detect_hold_until = 0.0
                self.stop_all_motion()
                return

            if self._detect_hold_until and now < self._detect_hold_until:
                self.stop_wheels()
                return
            self._detect_hold_until = 0.0

            cx, _cy, area, _depth_norm, mask_ratio, color = best_target
            width = frame_size[0]
            if width <= 0:
                self.stop_wheels()
                return
            center_x = width / 2.0
            height = frame_size[1]
            frame_area = float(width * height)
            if frame_area > 0:
                normalized_area = area / frame_area
                ratio_value = mask_ratio if mask_ratio is not None else normalized_area
                if self._area_smooth == 0.0:
                    self._area_smooth = ratio_value
                else:
                    if ratio_value > self._area_smooth:
                        self._area_smooth = 0.4 * self._area_smooth + 0.6 * ratio_value
                    else:
                        self._area_smooth = 0.7 * self._area_smooth + 0.3 * ratio_value
                ratio_for_ctrl = self._area_smooth
                close_stop = float(cfg.get("close_stop_ratio", 0.0))
                if close_stop > 0.0 and ratio_for_ctrl >= close_stop:
                    self.stop_wheels()
                    return
            raw_cx = cx
            if self._target_raw_pos is not None:
                raw_cx = self._target_raw_pos[0]
            align_cx = raw_cx
            if self._cx_smooth:
                align_cx = self._cx_smooth
            tol_ratio = float(cfg.get("align_center_tol_ratio", 0.08))
            tol_px = max(6.0, width * tol_ratio)
            error = align_cx - center_x
            if abs(error) <= tol_px:
                align_hold = max(0.0, float(cfg.get("align_hold_sec", 0.0)))
                if align_hold > 0.0:
                    if self._align_hold_started == 0.0:
                        self._align_hold_started = now
                    if now - self._align_hold_started < align_hold:
                        self._turn_error_prev = 0.0
                        self._drive_turn_smooth = 0.0
                        self._turn_phase = "idle"
                        self._turn_direction = 0
                        self.stop_wheels()
                        return
                self._align_hold_started = 0.0
                self._turn_error_prev = 0.0
                self._drive_turn_smooth = 0.0
                self._turn_phase = "idle"
                self._turn_direction = 0
                self.stop_wheels()
                self.autopilot_state = AutopilotState.APPROACHING
                self.auto_state_started_at = now
                self._step_phase = "idle"
                self._step_direction = 0
                LOGGER.debug("Auto: Aligned -> APPROACHING.")
                return
            self._align_hold_started = 0.0

            error_norm = error / max(1.0, center_x)
            error_norm = max(-1.0, min(1.0, error_norm))
            turn_dir = 1 if error_norm > 0 else -1
            if turn_dir != self._turn_direction:
                self._turn_phase = "idle"
                self._turn_direction = turn_dir
            deriv = error_norm - self._turn_error_prev
            self._turn_error_prev = error_norm
            kp = float(cfg.get("turn_kp", 1.0))
            kd = float(cfg.get("turn_kd", 0.0))
            control = kp * error_norm + kd * deriv
            control = max(-1.0, min(1.0, control))
            speed_scale = max(0.0, float(cfg.get("target_speed_scale", 1.0)))
            turn_scale = max(0.0, float(cfg.get("target_turn_scale", speed_scale)))
            turn_cmd = control * cfg["tracking_turn_speed_max"] * turn_scale
            smooth_alpha = float(cfg.get("turn_smooth_alpha", 0.6))
            self._drive_turn_smooth = smooth_alpha * self._drive_turn_smooth + (1.0 - smooth_alpha) * turn_cmd
            step_turn = max(0.0, float(cfg.get("turn_step_sec", 0.0)))
            step_pause = max(0.0, float(cfg.get("turn_pause_sec", 0.0)))
            if step_turn > 0.0:
                if self._turn_phase == "idle":
                    self._turn_phase = "move"
                    self._turn_until = now + step_turn
                if self._turn_phase == "pause":
                    self.stop_wheels()
                    if now >= self._turn_until:
                        self._turn_phase = "idle"
                    return
                if now >= self._turn_until:
                    self._turn_phase = "pause"
                    self._turn_until = now + step_pause
                    self.stop_wheels()
                    return
            self._set_drive(turn=self._drive_turn_smooth, forward=0.0)
            LOGGER.debug("Auto TRACKING: Target=%s, Turn=%.2f", color, self._drive_turn_smooth)
            return

        if state == AutopilotState.APPROACHING:
            if raw_target is None:
                lost_hold = max(0.0, float(cfg.get("lost_hold_sec", 0.0)))
                if self._last_target_seen_at and now - self._last_target_seen_at <= lost_hold:
                    self.stop_wheels()
                    return
                LOGGER.info("Auto: Target lost -> SCANNING.")
                self.autopilot_state = AutopilotState.SCANNING
                self.auto_state_started_at = now
                self._step_phase = "idle"
                self._step_direction = 0
                self.stop_all_motion()
                return

            cx, _cy, area, _depth_norm, mask_ratio, _color = best_target
            width, height = frame_size
            if width <= 0 or height <= 0:
                self.stop_wheels()
                return
            center_x = width / 2.0
            raw_cx = cx if self._target_raw_pos is None else self._target_raw_pos[0]
            align_cx = raw_cx
            if self._cx_smooth:
                align_cx = self._cx_smooth
            tol_ratio = float(cfg.get("align_center_tol_ratio", 0.08))
            tol_px = max(6.0, width * tol_ratio)
            if abs(align_cx - center_x) > tol_px:
                self.autopilot_state = AutopilotState.TRACKING
                self._step_phase = "idle"
                self._step_direction = 0
                self._align_hold_started = 0.0
                self._turn_phase = "idle"
                self._turn_direction = 0
                self.stop_wheels()
                return

            frame_area = float(width * height)
            normalized_area = area / frame_area
            ratio_value = mask_ratio if mask_ratio is not None else normalized_area
            if self._area_smooth == 0.0:
                self._area_smooth = ratio_value
            else:
                if ratio_value > self._area_smooth:
                    self._area_smooth = 0.4 * self._area_smooth + 0.6 * ratio_value
                else:
                    self._area_smooth = 0.7 * self._area_smooth + 0.3 * ratio_value
            ratio_for_ctrl = self._area_smooth

            target_area = cfg.get("target_area_ratio")
            tol_area = float(cfg.get("area_tolerance_ratio", 0.0))
            if isinstance(target_area, (int, float)):
                min_area = max(0.0, float(target_area) - tol_area)
            else:
                min_area = cfg["approach_area_threshold"]

            if ratio_for_ctrl >= min_area:
                self._step_phase = "idle"
                self._step_direction = 0
                self.stop_wheels()
                return

            close_stop = float(cfg.get("close_stop_ratio", 0.0))
            if close_stop > 0.0 and ratio_for_ctrl >= close_stop:
                self._step_phase = "idle"
                self._step_direction = 0
                self.stop_wheels()
                return

            direction = 1
            if direction != self._step_direction:
                self._step_phase = "idle"
                self._step_direction = direction

            step_move = max(0.05, float(cfg.get("step_move_sec", 0.4)))
            step_pause = max(0.0, float(cfg.get("step_pause_sec", 0.35)))
            speed_scale = max(0.0, float(cfg.get("target_speed_scale", 1.0)))
            base_forward = float(cfg["tracking_forward_speed_max"])
            min_forward = max(0.0, float(cfg.get("target_forward_min", 0.0)))
            forward_speed = max(base_forward * speed_scale, min_forward)

            if self._step_phase == "idle":
                self._step_phase = "move"
                self._step_until = now + step_move

            if self._step_phase == "move":
                if now >= self._step_until:
                    self._step_phase = "pause"
                    self._step_until = now + step_pause
                    self.stop_wheels()
                    return
                speed = forward_speed if direction > 0 else forward_speed
                fwd_cmd = speed if direction > 0 else -speed
                if self.auto_forward_invert:
                    fwd_cmd = -fwd_cmd
                self._set_drive(turn=0.0, forward=fwd_cmd)
                return

            if self._step_phase == "pause":
                self.stop_wheels()
                if now >= self._step_until:
                    self._step_phase = "idle"
                return

        if state == AutopilotState.GRASPING:
            self.stop_all_motion()
            self.autopilot_state = AutopilotState.TRACKING
            return

        if state == AutopilotState.SEARCH_FAILED:
            self.stop_all_motion()
            return

    def _set_drive(self, turn: float, forward: float) -> None:
        # simple differential mix
        forward_cmd = -forward if self.drive_invert else forward
        left = max(-100.0, min(100.0, forward_cmd + turn))
        right = max(-100.0, min(100.0, forward_cmd - turn))
        for wheel in self.wheels():
            val = left if "left" in wheel else right
            self.set_wheel_speed(wheel, val)

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
            "supply_voltage": self.supply_voltage,
            "supply_current": self.supply_current,
            "power_w": self.power_consumption_w,
            "pwm_frequency_hz": self.pwm_frequency_hz,
            "pwm_outputs": dict(self.pwm_output),
        }
