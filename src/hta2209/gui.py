"""
Tkinter GUI for HTA2209 with camera preview, manual/auto control, and test runners.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter import scrolledtext
from pathlib import Path
from typing import Dict

import cv2
from PIL import Image, ImageTk
try:
    from picamera2 import Picamera2
except ImportError:  # Picamera2 yoksa alt tarafta yeniden deneyecegiz
    Picamera2 = None  # type: ignore

from .controller import RobotController
from . import detector

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
# GUI log paneline controller loglarini aktaracak handler
from .controller import GuiLogHandler

CAMERA_CONTROL_FIELDS = (
    ("brightness", 0, 255),
    ("contrast", 0, 255),
    ("saturation", 0, 255),
    ("hue", 0, 255),
    ("gain", 0, 255),
)

MANUAL_DRIVE_INCREMENT = 10
MANUAL_TURN_INCREMENT = 10
JOINT_INCREMENT = 5
JOINT_SPEED_INCREMENT = 10
JOINT_KEY_BINDINGS = {
    "q": ("joint", JOINT_INCREMENT),
    "a": ("joint", -JOINT_INCREMENT),
    "t": ("gripper", JOINT_SPEED_INCREMENT),
    "g": ("gripper", -JOINT_SPEED_INCREMENT),
}

CAMERA_FRAME_INTERVAL_MS = 50
CAMERA_PREVIEW_SIZE = (640, 360)


class HTAControlGUI:
    def __init__(self, controller: RobotController) -> None:
        self.controller = controller
        self.root = tk.Tk()
        self.root.title("HTA2209 - YOLO Bazli Mobil Manipulator")
        self.root.geometry("1000x700")
        self._updating = False
        # Controller loglarini GUI paneline aktar
        try:
            handler = GuiLogHandler(self._append_log)
            handler.setLevel(logging.INFO)
            logging.getLogger("hta2209.controller").addHandler(handler)
        except Exception:
            pass

        self.status_var = tk.StringVar()
        self.mode_var = tk.StringVar(value=self.controller.mode)
        self.simulation_var = tk.BooleanVar(value=self.controller.is_simulation())
        self.run_state_var = tk.StringVar(value=self.controller.run_state)
        self.manual_drive_level = 0.0
        self.manual_turn_level = 0.0

        self.picam_supported = False
        self.picam: Picamera2 | None = None
        self._picam_color_order = "rgb"
        self._ensure_picamera2()

        # Varsayilan kaynak Picamera2; istenirse OpenCV (USB) ile de calisir.
        self._camera_failures = 0
        self.camera_status_var = tk.StringVar(value="Kamera kapali")
        self.camera_source_var = tk.StringVar(value=self.controller.camera_source)
        self.camera_index_var = tk.IntVar(value=self.controller.camera_index)
        self.target_color_var = tk.StringVar(value="red")
        self.camera_label: ttk.Label | None = None
        self.camera_running = False
        self.camera_loop_id: str | None = None
        self._camera_photo = None
        self._last_frame = None
        self.cv2_cap = None
        self.metric_labels: Dict[str, tk.StringVar] = {}

        self.camera_control_vars: Dict[str, tk.IntVar] = {}
        self.wheel_scales: Dict[str, ttk.Scale] = {}
        self.wheel_value_labels: Dict[str, tk.StringVar] = {}
        self.joint_scales: Dict[str, ttk.Scale] = {}
        self.joint_labels: Dict[str, tk.StringVar] = {}
        self.stop_button: ttk.Button | None = None
        self.drive_buttons: list[ttk.Button] = []
        self.require_hardware_var = tk.BooleanVar(value=False)
        self.test_output: scrolledtext.ScrolledText | None = None
        self.repo_root = Path(__file__).resolve().parents[2]
        self.yolo_detector = detector.YoloDetector(self.repo_root / "hta2209.pt")
        self.yolo_target_class = "red"
        self._yolo_last_detections: list[dict] = []
        self._yolo_last_target = None
        self._yolo_last_error = None
        self._yolo_last_infer = 0.0
        self._yolo_interval_s = 0.25
        self._yolo_disabled = False
        self.controller.auto_target_color = self.yolo_target_class
        self.target_color_var.set(self.yolo_target_class)

        self._create_widgets()
        self.refresh_from_controller()
        self.root.bind_all("<KeyPress>", self._on_key_press)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _ensure_picamera2(self) -> None:
        """Try to import Picamera2 even if not in virtualenv site-packages."""
        global Picamera2
        if self.picam_supported:
            return
        if Picamera2 is not None:
            self.picam_supported = True
            return
        candidates = [
            f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages",
            "/usr/lib/python3/dist-packages",  # system python default (3.x)
        ]
        for candidate in candidates:
            if candidate not in sys.path:
                sys.path.append(candidate)
        try:
            from picamera2 import Picamera2 as _Picamera2
            Picamera2 = _Picamera2  # type: ignore
            self.picam_supported = True
        except Exception:
            self.picam_supported = False

    # ------------------------------------------------------------------ #
    def _create_widgets(self) -> None:
        self._create_header()
        root_paned = ttk.Panedwindow(self.root, orient=tk.VERTICAL)
        root_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        top_paned = ttk.Panedwindow(root_paned, orient=tk.HORIZONTAL)
        root_paned.add(top_paned, weight=5)

        sidebar_frame = ttk.Frame(top_paned)
        camera_frame = ttk.Frame(top_paned)
        tabs_frame = ttk.Frame(top_paned)

        top_paned.add(sidebar_frame, weight=1)
        top_paned.add(camera_frame, weight=4)
        top_paned.add(tabs_frame, weight=2)

        self._create_sidebar(sidebar_frame)
        self._create_camera_panel(camera_frame)

        notebook = ttk.Notebook(tabs_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=(10, 0), pady=(0, 5))
        notebook.add(self._connection_tab(notebook), text="Baglanti")
        notebook.add(self._yolo_tab(notebook), text="YOLO")
        notebook.add(self._drive_tab(notebook), text="Surus Kontrol")
        notebook.add(self._arm_tab(notebook), text="Robot Kol")
        notebook.add(self._metrics_tab(notebook), text="Grafikler")

        log_frame = ttk.Frame(root_paned)
        root_paned.add(log_frame, weight=1)
        self._create_log_panel(log_frame)

        # Varsayilan kamerayi otomatik tara ve varsa onizlemeyi hazirla
        self._refresh_camera_devices()
        # Kullanici isterse "Baslat" ile acacak; otomatik baslatma yok

    def _create_header(self) -> None:
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X)

        status_label = ttk.Label(top_frame, textvariable=self.status_var, font=("Segoe UI", 10, "bold"))
        status_label.pack(side=tk.LEFT)

        ttk.Button(top_frame, text="Kaydet", command=self._save_state).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(top_frame, text="Yeniden Yukle", command=self._reload_state).pack(side=tk.RIGHT)

    # ------------------------------------------------------------------ #
    def _connection_tab(self, parent: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=10)

        hw = "Hazir" if self.controller.hardware_ready else "Simulasyon"
        ttk.Label(frame, text=f"Donanim modu: {hw}").pack(anchor=tk.W, pady=5)
        ttk.Label(frame, text=f"Konfigurasyon dosyasi: {self.controller.config_path}").pack(anchor=tk.W, pady=5)

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        description = (
            "GUI, SSH X11/VNC veya yerelde calisir. Kamera onizlemesi orta panelde sabit kalir."
        )
        ttk.Label(frame, text=description, wraplength=800, justify=tk.LEFT).pack(anchor=tk.W)

        mode_info = (
            "Manual: Tekerlek/eklem kontrolleri ve klavye acik.\n"
            "Auto: Manuel girdiler kilitlenir, otonom davranis icin hazirlanir."
        )
        ttk.Label(frame, text=mode_info, wraplength=800, justify=tk.LEFT).pack(anchor=tk.W, pady=10)

        keyboard_info = (
            "Klavye (Manual): Yon tuslari hiz/yon, Space durdurur; Q/A, W/S, E/D, R/F, T/G eklemleri 5 derece oynatir."
        )
        ttk.Label(frame, text=keyboard_info, justify=tk.LEFT).pack(anchor=tk.W)
        return frame

    # ------------------------------------------------------------------ #
    def _yolo_tab(self, parent: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=10)
        ttk.Label(frame, text="Auto modda yalnizca YOLOv8 kullanilir.", font=("Segoe UI", 11, "bold")).pack(
            anchor="w", pady=(0, 8)
        )
        ttk.Label(frame, text=f"Model: {self.yolo_detector.model_path}").pack(anchor="w", pady=2)
        ttk.Label(frame, text=f"Sinif: {self.yolo_target_class}").pack(anchor="w", pady=2)
        ttk.Label(
            frame,
            text="Not: Model bulunamazsa auto mod hedef tespiti yapmaz.",
            wraplength=700,
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(8, 0))
        return frame

    def _make_camera_control_callback(self, field_name: str, var: tk.IntVar):
        def callback() -> None:
            self._on_camera_control_change(field_name, var.get())

        return callback

    def _make_camera_control_trace(self, field_name: str, var: tk.IntVar):
        def trace(*_args) -> None:
            if self._updating:
                return
            self._on_camera_control_change(field_name, var.get())

        return trace

    def _on_camera_control_change(self, field_name: str, value: int) -> None:
        if field_name in self.controller.camera_controls:
            self.controller.camera_controls[field_name] = int(value)
        if self.camera_running:
            self._apply_camera_controls()

    def _on_camera_source_change(self) -> None:
        self.controller.camera_source = self.camera_source_var.get()
        self._update_camera_source_ui()
        if self.camera_running:
            self._stop_stream()
            self._start_stream()

    def _make_camera_index_trace(self):
        def trace(*_args) -> None:
            if self._updating:
                return
            self._on_camera_index_change()

        return trace

    def _on_camera_index_change(self, *_args) -> None:
        if self._updating:
            return
        try:
            self.controller.camera_index = int(self.camera_index_var.get())
        except (TypeError, ValueError):
            return
        if self.camera_running and self.camera_source_var.get() == "opencv":
            self._stop_stream()
            self._start_stream()

    def _update_camera_source_ui(self) -> None:
        use_opencv = self.camera_source_var.get() == "opencv"
        try:
            self.camera_index_spin.configure(state="normal" if use_opencv else "disabled")
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def _drive_tab(self, parent: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=10)
        for idx, wheel in enumerate(self.controller.wheels()):
            lf = ttk.LabelFrame(frame, text=wheel.replace("_", " ").title(), padding=10)
            lf.grid(row=idx // 2, column=idx % 2, padx=10, pady=10, sticky="nsew")
            frame.grid_columnconfigure(idx % 2, weight=1)
            var = tk.StringVar(value="0%")
            self.wheel_value_labels[wheel] = var
            ttk.Label(lf, textvariable=var, font=("Segoe UI", 11, "bold")).pack(anchor=tk.W)
            scale = ttk.Scale(
                lf,
                from_=-100,
                to=100,
                orient=tk.HORIZONTAL,
                command=lambda val, w=wheel: self._on_wheel_change(w, float(val)),
                length=300,
            )
            scale.pack(fill=tk.X, padx=5, pady=10)
            self.wheel_scales[wheel] = scale
        self.stop_button = ttk.Button(frame, text="Tekerlekleri Durdur", command=self._stop_wheels)
        self.stop_button.grid(
            row=(len(self.controller.wheels()) + 1) // 2 + 1, column=0, columnspan=2, pady=10
        )

        controls = ttk.LabelFrame(frame, text="Hizli Kontrol (Klavye gibi)", padding=10)
        controls.grid(row=(len(self.controller.wheels()) + 1) // 2 + 2, column=0, columnspan=2, pady=5, sticky="ew")
        btn_forward = ttk.Button(controls, text="▲", command=lambda: self._drive_button_action("forward"))
        btn_back = ttk.Button(controls, text="▼", command=lambda: self._drive_button_action("backward"))
        btn_left = ttk.Button(controls, text="◀", command=lambda: self._drive_button_action("left"))
        btn_right = ttk.Button(controls, text="▶", command=lambda: self._drive_button_action("right"))
        btn_stop = ttk.Button(controls, text="■", command=self._stop_wheels)
        btn_forward.grid(row=0, column=1, padx=5, pady=2)
        btn_left.grid(row=1, column=0, padx=5, pady=2)
        btn_stop.grid(row=1, column=1, padx=5, pady=2)
        btn_right.grid(row=1, column=2, padx=5, pady=2)
        btn_back.grid(row=2, column=1, padx=5, pady=2)
        self.drive_buttons = [btn_forward, btn_back, btn_left, btn_right, btn_stop]

        return frame

    def _on_wheel_change(self, wheel: str, value: float) -> None:
        if self._updating:
            return
        # Kullanici hareket verdiğinde yalnizca stopped ise otomatik baslat; paused ise kilitli kalsin
        if (not self.controller.is_running()) and (not self.controller.is_paused()):
            try:
                self.controller.set_run_state("started")
                self.run_state_var.set(f"Durum: {self.controller.run_state}")
            except Exception:
                pass
        self.controller.set_wheel_speed(wheel, value)
        self.wheel_value_labels[wheel].set(f"{value:.0f}%")

    def _stop_wheels(self) -> None:
        self.manual_drive_level = 0.0
        self.manual_turn_level = 0.0
        self.controller.stop_wheels()
        self.refresh_from_controller()

    def _reset_manual_motion(self) -> None:
        self.manual_drive_level = 0.0
        self.manual_turn_level = 0.0
        self.controller.stop_all_motion()
        self.refresh_from_controller()

    def _drive_button_action(self, action: str) -> None:
        if action == "forward":
            self._adjust_drive(MANUAL_DRIVE_INCREMENT)
        elif action == "backward":
            self._adjust_drive(-MANUAL_DRIVE_INCREMENT)
        elif action == "left":
            self._adjust_turn(-MANUAL_TURN_INCREMENT)
        elif action == "right":
            self._adjust_turn(MANUAL_TURN_INCREMENT)

    def _adjust_drive(self, delta: float) -> None:
        if (not self.controller.is_running()) and (not self.controller.is_paused()):
            try:
                self.controller.set_run_state("started")
                self.run_state_var.set(f"Durum: {self.controller.run_state}")
            except Exception:
                pass
        next_level = self.manual_drive_level + delta
        if self.manual_drive_level != 0 and (self.manual_drive_level * next_level) < 0:
            self.manual_drive_level = 0.0
        else:
            self.manual_drive_level = self._clamp_speed(next_level)
        self._apply_drive_turn()

    def _adjust_turn(self, delta: float) -> None:
        if (not self.controller.is_running()) and (not self.controller.is_paused()):
            try:
                self.controller.set_run_state("started")
                self.run_state_var.set(f"Durum: {self.controller.run_state}")
            except Exception:
                pass
        self.manual_turn_level = self._clamp_speed(self.manual_turn_level + delta)
        self._apply_drive_turn()

    def _apply_drive_turn(self) -> None:
        self.controller._set_drive(turn=self.manual_turn_level, forward=self.manual_drive_level)
        self.refresh_from_controller()

    @staticmethod
    def _clamp_speed(value: float) -> float:
        return max(-100.0, min(100.0, value))

    # ------------------------------------------------------------------ #
    def _arm_tab(self, parent: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=10)
        for idx, joint in enumerate(self.controller.joints()):
            lf = ttk.LabelFrame(frame, text=joint.title(), padding=10)
            lf.grid(row=idx // 2, column=idx % 2, padx=10, pady=10, sticky="nsew")
            frame.grid_columnconfigure(idx % 2, weight=1)
            var = tk.StringVar(value="0" if joint in self.controller.continuous() else "90")
            self.joint_labels[joint] = var
            ttk.Label(lf, textvariable=var, font=("Segoe UI", 11, "bold")).pack(anchor=tk.W)
            if joint in self.controller.continuous():
                scale = ttk.Scale(
                    lf,
                    from_=-100,
                    to=100,
                    orient=tk.HORIZONTAL,
                    command=lambda val, j=joint: self._on_joint_change(j, float(val)),
                    length=300,
                )
                ttk.Label(lf, text="Hiz (%)").pack(anchor=tk.W)
            else:
                scale = ttk.Scale(
                    lf,
                    from_=0,
                    to=180,
                    orient=tk.HORIZONTAL,
                    command=lambda val, j=joint: self._on_joint_change(j, float(val)),
                    length=300,
                )
                ttk.Label(lf, text="Aci (derece)").pack(anchor=tk.W)
            scale.pack(fill=tk.X, padx=5, pady=10)
            self.joint_scales[joint] = scale
        return frame

    def _metrics_tab(self, parent: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=10)
        grid = ttk.Frame(frame)
        grid.pack(fill=tk.BOTH, expand=True)

        rows = [
            ("Mod", "mode"),
            ("Simulasyon", "simulation"),
            ("Calisma", "run_state"),
            ("Auto hedef sinif", "auto_target_color"),
            ("Kaynak Voltaji (V)", "supply_voltage"),
            ("Anlik Akim (A)", "supply_current"),
            ("Guc Tuketimi (W)", "power_w"),
            ("PWM Frekansi (Hz)", "pwm_freq"),
        ]
        for i, (label, key) in enumerate(rows):
            ttk.Label(grid, text=label + ":").grid(row=i, column=0, sticky="w", padx=5, pady=2)
            var = tk.StringVar(value="")
            self.metric_labels[key] = var
            ttk.Label(grid, textvariable=var, font=("Segoe UI", 10, "bold")).grid(
                row=i, column=1, sticky="w", padx=5, pady=2
            )

        ttk.Separator(grid, orient=tk.HORIZONTAL).grid(row=len(rows), column=0, columnspan=2, sticky="ew", pady=8)
        start = len(rows) + 1
        for idx, wheel in enumerate(self.controller.wheels()):
            ttk.Label(grid, text=wheel.replace("_", " ").title() + ":").grid(
                row=start + idx, column=0, sticky="w", padx=5, pady=2
            )
            var = tk.StringVar(value="")
            self.metric_labels[f"wheel_{wheel}"] = var
            ttk.Label(grid, textvariable=var).grid(row=start + idx, column=1, sticky="w", padx=5, pady=2)

        start = start + len(self.controller.wheels())
        for idx, wheel in enumerate(self.controller.wheels()):
            ttk.Label(grid, text="PWM " + wheel.replace("_", " ").title() + ":").grid(
                row=start + idx, column=0, sticky="w", padx=5, pady=2
            )
            var = tk.StringVar(value="")
            self.metric_labels[f"pwm_{wheel}"] = var
            ttk.Label(grid, textvariable=var).grid(row=start + idx, column=1, sticky="w", padx=5, pady=2)

        start = start + len(self.controller.wheels())
        for idx, joint in enumerate(self.controller.joints()):
            ttk.Label(grid, text=joint.title() + ":").grid(row=start + idx, column=0, sticky="w", padx=5, pady=2)
            var = tk.StringVar(value="")
            self.metric_labels[f"joint_{joint}"] = var
            ttk.Label(grid, textvariable=var).grid(row=start + idx, column=1, sticky="w", padx=5, pady=2)

        return frame

    def _on_joint_change(self, joint: str, value: float) -> None:
        if self._updating:
            return
        if joint in self.controller.continuous():
            self.controller.set_continuous_speed(joint, value)
            self.joint_labels[joint].set(f"{value:.0f}%")
        else:
            self.controller.set_joint_angle(joint, value)
            self.joint_labels[joint].set(f"{value:.0f}°")

    def _refresh_camera_devices(self) -> None:
        if self.picam_supported:
            self.camera_status_var.set("Picamera2 hazir")
        else:
            self.camera_status_var.set("Picamera2 bulunamadi (OpenCV kullanilabilir)")

    def _apply_camera_controls(self) -> None:
        controls = self.controller.camera_controls
        source = self.camera_source_var.get()
        if source == "opencv" and self.cv2_cap is not None:
            self.cv2_cap.set(cv2.CAP_PROP_BRIGHTNESS, controls.get("brightness", 0))
            self.cv2_cap.set(cv2.CAP_PROP_CONTRAST, controls.get("contrast", 0))
            self.cv2_cap.set(cv2.CAP_PROP_SATURATION, controls.get("saturation", 0))
            self.cv2_cap.set(cv2.CAP_PROP_HUE, controls.get("hue", 0))
            self.cv2_cap.set(cv2.CAP_PROP_GAIN, controls.get("gain", 0))
            return
        if source == "picamera2" and self.picam is not None:
            control_map = {
                "brightness": "Brightness",
                "contrast": "Contrast",
                "saturation": "Saturation",
                "gain": "AnalogueGain",
                "hue": "Hue",
            }
            payload = {}
            try:
                available = getattr(self.picam, "camera_controls", None)
            except Exception:
                available = None
            if not isinstance(available, dict) or not available:
                return
            for key, name in control_map.items():
                if key not in controls:
                    continue
                if name not in available:
                    continue
                value = controls.get(key)
                if value is None:
                    continue
                ctrl_info = available.get(name)
                if ctrl_info is None:
                    continue
                try:
                    min_val = float(ctrl_info.min)
                    max_val = float(ctrl_info.max)
                    value = float(value)
                    value = min_val + (max_val - min_val) * (value / 255.0)
                    value = max(min_val, min(max_val, value))
                except Exception:
                    continue
                payload[name] = value
            if payload:
                try:
                    self.picam.set_controls(payload)
                except Exception as exc:
                    self._append_log(f"Kamera ayarlari uygulanamadi: {exc}")

    def _start_stream(self) -> None:
        self._stop_stream()
        source = self.camera_source_var.get()
        self.controller.camera_source = source
        self._on_camera_index_change()

        if source == "opencv":
            try:
                self.cv2_cap = cv2.VideoCapture(self.controller.camera_index)
                if not self.cv2_cap.isOpened():
                    raise RuntimeError("OpenCV kamera acilamadi")
                self.cv2_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.controller.camera_frame_width)
                self.cv2_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.controller.camera_frame_height)
                self._apply_camera_controls()
                self.camera_running = True
                self._camera_failures = 0
                self.camera_status_var.set(f"Kamera acik (OpenCV index {self.controller.camera_index})")
                self._update_camera_frame()
                return
            except Exception as exc:
                self.camera_status_var.set(f"OpenCV kamera acilamadi: {exc}")
                messagebox.showerror("Kamera", f"OpenCV kamera acilamadi: {exc}")
                return

        if not self.picam_supported:
            self.camera_status_var.set("Picamera2 bulunamadi; OpenCV kullanabilirsiniz.")
            messagebox.showerror("Kamera", "Picamera2 modulu bulunamadi veya yuklenemedi.")
            return
        try:
            if self.picam is None:
                self.picam = Picamera2()
            preferred_size = (
                int(self.controller.camera_frame_width),
                int(self.controller.camera_frame_height),
            )
            fallback_size = (1280, 720)
            actual_size = preferred_size
            self._picam_color_order = "rgb"
            try:
                config = self.picam.create_video_configuration(main={"size": preferred_size, "format": "RGB888"})
                self.picam.configure(config)
            except Exception:
                try:
                    config = self.picam.create_video_configuration(main={"size": preferred_size, "format": "BGR888"})
                    self.picam.configure(config)
                    self._picam_color_order = "bgr"
                except Exception:
                    actual_size = fallback_size
                    try:
                        config = self.picam.create_video_configuration(main={"size": fallback_size, "format": "RGB888"})
                        self.picam.configure(config)
                    except Exception:
                        config = self.picam.create_video_configuration(main={"size": fallback_size, "format": "BGR888"})
                        self.picam.configure(config)
                        self._picam_color_order = "bgr"
            self._sync_picam_color_order()
            self.picam.start()
            self._apply_camera_controls()
            self.camera_running = True
            self._camera_failures = 0
            self.camera_status_var.set(f"Kamera acik (Picamera2 {actual_size[0]}x{actual_size[1]})")
            self._update_camera_frame()
            return
        except Exception as exc:
            self.camera_status_var.set(f"Picamera2 acilamadi: {exc}")
            messagebox.showerror("Kamera", f"Picamera2 acilamadi: {exc}")

    def _stop_stream(self) -> None:
        self.camera_running = False
        if self.camera_loop_id is not None:
            self.root.after_cancel(self.camera_loop_id)
            self.camera_loop_id = None
        if self.picam is not None:
            try:
                self.picam.stop()
            except Exception:
                pass
        if self.cv2_cap is not None:
            try:
                self.cv2_cap.release()
            except Exception:
                pass
            self.cv2_cap = None
        self.camera_status_var.set("Kaynak kapali")
        if self.camera_label is not None:
            self.camera_label.configure(image="", text="Onizleme yok")
        self._camera_photo = None
        self._last_frame = None
        self._camera_failures = 0

    def _apply_preview_guides(self, frame):
        if frame is None or frame.size == 0:
            return frame
        if self._last_frame is not None and frame is self._last_frame:
            frame = frame.copy()
        height, width = frame.shape[:2]
        x1 = width // 3
        x2 = (2 * width) // 3
        guide_color = (255, 255, 255)
        cv2.line(frame, (x1, 0), (x1, height - 1), guide_color, 1)
        cv2.line(frame, (x2, 0), (x2, height - 1), guide_color, 1)
        return frame

    def _run_yolo_detection(self, frame_rgb):
        if self._yolo_disabled:
            return [], None
        if frame_rgb is None or frame_rgb.size == 0:
            return [], None
        now = time.monotonic()
        if now - self._yolo_last_infer < self._yolo_interval_s:
            return self._yolo_last_detections, self._yolo_last_target

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        detections = self.yolo_detector.detect(frame_bgr, class_name=self.yolo_target_class)
        self._yolo_last_infer = now
        err = self.yolo_detector.last_error
        if err:
            if err != self._yolo_last_error:
                self._append_log(f"YOLO: {err}")
                self._yolo_last_error = err
            fatal_markers = ("not available", "not found", "load failed")
            if any(marker in err for marker in fatal_markers):
                self._yolo_disabled = True
            self._yolo_last_detections = []
            self._yolo_last_target = None
            return [], None

        self._yolo_last_error = None
        self._yolo_last_detections = detections
        self._yolo_last_target = detector.yolo_best_target(
            detections,
            frame_rgb.shape[:2],
            self.yolo_target_class,
        )
        return self._yolo_last_detections, self._yolo_last_target

    def _draw_yolo_detections(self, frame_rgb, detections):
        if frame_rgb is None or frame_rgb.size == 0 or not detections:
            return frame_rgb
        overlay = frame_rgb.copy()
        for det in detections:
            coords = det.get("xyxy")
            if not isinstance(coords, (list, tuple)) or len(coords) != 4:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in coords]
            color = (255, 0, 0)
            label = f"{det.get('class_name', '')} {float(det.get('conf', 0.0)):.2f}"
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                overlay,
                label.strip(),
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        return overlay

    def _update_camera_frame(self) -> None:
        if not self.camera_running:
            return
        source = self.camera_source_var.get()
        source_label = "OpenCV" if source == "opencv" else "Picamera2"
        try:
            if source == "opencv":
                if self.cv2_cap is None:
                    raise RuntimeError("OpenCV kamera hazir degil")
                ok, frame = self.cv2_cap.read()
                if not ok or frame is None:
                    raise RuntimeError("OpenCV frame is None")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                if self.picam is None:
                    return
                frame = self.picam.capture_array()
                if frame is None:
                    raise RuntimeError("Picamera2 frame is None")
                # Kamera ters takili, 180 derece cevir
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                if frame.ndim == 3 and frame.shape[2] == 4:
                    if self._picam_color_order == "rgb":
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                elif self._picam_color_order == "bgr":
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.controller.camera_swap_rb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            self._camera_failures += 1
            if self._camera_failures >= 5:
                self.camera_status_var.set(f"{source_label} kare okunamadi ({exc}), durduruluyor.")
                self._stop_stream()
                return
            self.camera_loop_id = self.root.after(CAMERA_FRAME_INTERVAL_MS, self._update_camera_frame)
            return

        self._camera_failures = 0
        self._last_frame = frame
        display_frame = self._last_frame

        if not self.controller.is_manual():
            yolo_detections, yolo_target = self._run_yolo_detection(self._last_frame)
            if yolo_detections:
                display_frame = self._draw_yolo_detections(display_frame, yolo_detections)
            if self._yolo_disabled:
                self.controller.stop_all_motion()
                self.controller.last_target = None
                self.controller.last_target_color = None
            else:
                self.controller.autopilot_step(
                    yolo_target,
                    (display_frame.shape[1], display_frame.shape[0]),
                )
            if self.controller.last_target:
                lt = self.controller.last_target
                cx, cy, area = lt[0], lt[1], lt[2]
                depth = lt[3] if len(lt) > 3 else None
                mask_ratio = lt[4] if len(lt) > 4 else None
                target_color = self.controller.last_target_color or (lt[5] if len(lt) > 5 else None)
                color_map = {
                    "red": (255, 0, 0),
                    "green": (0, 255, 0),
                    "blue": (0, 0, 255),
                    "yellow": (255, 255, 0),
                    "orange": (255, 165, 0),
                    "purple": (255, 0, 255),
                    "cyan": (0, 255, 255),
                }
                rgb = color_map.get(target_color or "yellow", (0, 255, 0))
                cv2.circle(display_frame, (cx, cy), 8, rgb, 2)
                label = f"X={cx}, Y={cy}"
                if depth is not None:
                    label += f", Z~{depth:.2f}"
                if mask_ratio is not None:
                    label += f", oran={mask_ratio:.2f}"
                cv2.putText(
                    display_frame,
                    label,
                    (cx + 10, max(20, cy - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    rgb,
                    2,
                )
            self.refresh_from_controller()

        display_frame = self._apply_preview_guides(display_frame)
        image = Image.fromarray(display_frame)
        image = image.resize(CAMERA_PREVIEW_SIZE, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image=image)
        if self.camera_label is not None:
            self.camera_label.configure(image=photo, text="")
            self.camera_label.image = photo  # type: ignore[attr-defined]
        self._camera_photo = photo

        self.camera_loop_id = self.root.after(CAMERA_FRAME_INTERVAL_MS, self._update_camera_frame)

    # ------------------------------------------------------------------ #
    # Test helpers
    # ------------------------------------------------------------------ #
    def _run_software_test(self) -> None:
        self._run_test("tests/software_test.py", require_hardware=False)

    def _run_hardware_test(self) -> None:
        self._run_test("tests/hardware_test.py", require_hardware=self.require_hardware_var.get())

    def _run_test(self, script_rel: str, require_hardware: bool = False) -> None:
        script_path = self.repo_root / script_rel
        env = dict(**os.environ)
        if require_hardware:
            env["REQUIRE_HARDWARE"] = "1"
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.repo_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except Exception as exc:
            self._append_test_output(f"[ERROR] Test calistirilamadi: {exc}")
            messagebox.showerror("Test", f"Test calistirilamadi: {exc}")
            return

        output = (result.stdout or "") + (result.stderr or "")
        self._append_test_output(output.strip() or "<bos cikti>")
        if result.returncode == 0:
            messagebox.showinfo("Test", f"{script_rel} basarili.")
        else:
            messagebox.showerror("Test", f"{script_rel} basarisiz (exit {result.returncode}).")

    def _append_test_output(self, text: str) -> None:
        if self.test_output is not None:
            self.test_output.configure(state="normal")
            if self.test_output.index("end-1c") != "1.0":
                self.test_output.insert(tk.END, "\n" + "-" * 40 + "\n")
            self.test_output.insert(tk.END, text + "\n")
            self.test_output.see(tk.END)
            self.test_output.configure(state="disabled")
        # Her durumda Log paneline yaz
        self._append_log(text)

    def _on_wheel_invert_toggle(self, wheel: str, inverted: bool) -> None:
        self.controller.wheel_polarity[wheel] = -1 if inverted else 1
        self.controller.save_config()
        self._append_log(f"{wheel} polaritesi {'-1' if inverted else '1'} olarak ayarlandi")

    def _append_log(self, text: str) -> None:
        if not hasattr(self, "log_output") or self.log_output is None:
            return
        self.log_output.insert(tk.END, text + "\n")
        self.log_output.see(tk.END)
        # Ayrica logging'e de yaz
        try:
            LOGGER.info(text)
        except Exception:
            pass

    def _clear_log(self) -> None:
        if not hasattr(self, "log_output") or self.log_output is None:
            return
        self.log_output.delete("1.0", tk.END)

    def _copy_log_selection(self, event=None):
        if not hasattr(self, "log_output") or self.log_output is None:
            return "break"
        try:
            text = self.log_output.get("sel.first", "sel.last")
        except tk.TclError:
            text = self.log_output.get("1.0", tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        return "break"

    def _show_log_menu(self, event) -> None:
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Kopyala", command=lambda: self._copy_log_selection())
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _block_log_edit(self, event) -> str | None:
        # Klavye ile duzenlemeyi engelle; sadece kopyaya izin ver
        if event.keysym.lower() == "c" and (event.state & 0x4 or event.state & 0x100000):
            return None
        return "break"

    # ------------------------------------------------------------------ #
    def _create_sidebar(self, parent: ttk.Frame) -> None:
        sidebar = ttk.Frame(parent, padding=10)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.rowconfigure(99, weight=1)

        settings = ttk.LabelFrame(sidebar, text="Ayarlar", padding=10)
        settings.pack(fill=tk.X, pady=(0, 10))

        mode_frame = ttk.LabelFrame(settings, text="Kontrol Modu", padding=5)
        mode_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Radiobutton(
            mode_frame, text="Manual", value="manual", variable=self.mode_var, command=self._on_mode_change
        ).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(
            mode_frame, text="Auto", value="auto", variable=self.mode_var, command=self._on_mode_change
        ).pack(side=tk.LEFT, padx=5)

        sim_check = ttk.Checkbutton(
            settings,
            text="Simulasyon (donanim yerine sanal calistir)",
            variable=self.simulation_var,
            command=self._on_simulation_toggle,
        )
        sim_check.pack(anchor="w", pady=(4, 6))

        run_frame = ttk.LabelFrame(settings, text="Calisma Durumu", padding=5)
        run_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(run_frame, text="Baslat", command=lambda: self._set_run_state("started")).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(run_frame, text="Duraklat", command=lambda: self._set_run_state("paused")).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(run_frame, text="Durdur", command=lambda: self._set_run_state("stopped")).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Label(run_frame, textvariable=self.run_state_var, foreground="blue").pack(side=tk.LEFT, padx=6)

        color_frame = ttk.Frame(settings)
        color_frame.pack(fill=tk.X, pady=(5, 5))
        ttk.Label(color_frame, text="Auto hedef sinif (YOLO: red):").pack(side=tk.LEFT)
        color_combo = ttk.Combobox(
            color_frame,
            values=[self.yolo_target_class],
            textvariable=self.target_color_var,
            state="disabled",
            width=10,
        )
        color_combo.pack(side=tk.LEFT, padx=5)
        color_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_target_color_change())

        ttk.Button(settings, text="Kaydet", command=self._save_state).pack(fill=tk.X, pady=2)
        ttk.Button(settings, text="Yeniden Yukle", command=self._reload_state).pack(fill=tk.X, pady=2)

        cam_box = ttk.LabelFrame(sidebar, text="Kamera Kontrolu", padding=10)
        cam_box.pack(fill=tk.X, pady=(0, 10))
        row = 0
        source_frame = ttk.Frame(cam_box)
        source_frame.grid(row=row, column=0, columnspan=4, sticky="w")
        ttk.Label(source_frame, text="Kaynak:").pack(side=tk.LEFT)
        ttk.Radiobutton(
            source_frame,
            text="Picamera2 (CSI)",
            value="picamera2",
            variable=self.camera_source_var,
            command=self._on_camera_source_change,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(
            source_frame,
            text="OpenCV (USB)",
            value="opencv",
            variable=self.camera_source_var,
            command=self._on_camera_source_change,
        ).pack(side=tk.LEFT, padx=4)
        row += 1
        index_frame = ttk.Frame(cam_box)
        index_frame.grid(row=row, column=0, columnspan=4, sticky="w", pady=(2, 4))
        ttk.Label(index_frame, text="OpenCV indeks:").pack(side=tk.LEFT)
        self.camera_index_spin = tk.Spinbox(
            index_frame,
            from_=0,
            to=10,
            textvariable=self.camera_index_var,
            width=5,
            command=self._on_camera_index_change,
        )
        self.camera_index_spin.pack(side=tk.LEFT, padx=4)
        self.camera_index_var.trace_add("write", self._make_camera_index_trace())
        row += 1

        ttk.Button(cam_box, text="Baslat", command=self._start_stream).grid(row=row, column=0, columnspan=2, padx=5, pady=2, sticky="ew")
        ttk.Button(cam_box, text="Durdur", command=self._stop_stream).grid(row=row, column=2, columnspan=2, padx=5, pady=2, sticky="ew")
        row += 1

        ttk.Label(cam_box, textvariable=self.camera_status_var).grid(row=row, column=0, columnspan=4, sticky="w")
        row += 1

        controls_box = ttk.LabelFrame(cam_box, text="Goruntu Ayarlari", padding=6)
        controls_box.grid(row=row, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        for idx, (field_name, min_val, max_val) in enumerate(CAMERA_CONTROL_FIELDS):
            ttk.Label(controls_box, text=field_name.replace("_", " ").title()).grid(row=idx, column=0, sticky="w")
            var = tk.IntVar(value=self.controller.camera_controls.get(field_name, 0))
            spin = tk.Spinbox(
                controls_box,
                from_=min_val,
                to=max_val,
                textvariable=var,
                width=6,
                command=self._make_camera_control_callback(field_name, var),
            )
            spin.grid(row=idx, column=1, padx=5, pady=2, sticky="e")
            var.trace_add("write", self._make_camera_control_trace(field_name, var))
            self.camera_control_vars[field_name] = var
        self._update_camera_source_ui()

        ttk.Label(
            sidebar,
            text="Not: Kamera onizlemesi orta panelde sabit kalir; sekmeler sagda kalir.",
            wraplength=200,
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(0, 10))

        tests_box = ttk.LabelFrame(sidebar, text="Testler", padding=10)
        tests_box.pack(fill=tk.X, expand=False)
        ttk.Checkbutton(
            tests_box,
            text="Donanim zorunlu (REQUIRE_HARDWARE=1)",
            variable=self.require_hardware_var,
        ).pack(anchor="w", pady=(0, 5))

        invert_box = ttk.LabelFrame(tests_box, text="Tekerlek Yonu", padding=5)
        invert_box.pack(fill=tk.X, pady=(5, 5))
        for wheel in self.controller.wheels():
            var = tk.BooleanVar(value=self.controller.wheel_polarity.get(wheel, 1) == -1)
            ttk.Checkbutton(
                invert_box,
                text=wheel.replace("_", " "),
                variable=var,
                command=lambda w=wheel, v=var: self._on_wheel_invert_toggle(w, v.get()),
            ).pack(anchor="w")

        ttk.Button(tests_box, text="SOFTWARE_TEST", command=self._run_software_test).pack(fill=tk.X, pady=2)
        ttk.Button(tests_box, text="HARDWARE_TEST", command=self._run_hardware_test).pack(fill=tk.X, pady=2)
        # Test çıktıları alttaki Loglar sekmesine yönlendiriliyor.
        self.test_output = None

    def _create_camera_panel(self, parent: ttk.Frame) -> None:
        preview_frame = ttk.LabelFrame(parent, text="Kamera Onizleme", padding=6)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        self.camera_label = ttk.Label(preview_frame, text="Onizleme yok", anchor="center")
        self.camera_label.grid(row=0, column=0, sticky="nsew")

    def _create_log_panel(self, parent: ttk.Frame) -> None:
        log_frame = ttk.LabelFrame(parent, text="Loglar", padding=6)
        log_frame.pack(fill=tk.BOTH, expand=True)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_output = scrolledtext.ScrolledText(log_frame, height=6, state="normal", wrap="word")
        self.log_output.grid(row=0, column=0, sticky="nsew")
        # Kopyalama / menü / düzenleme engeli
        self.log_output.bind("<Control-c>", self._copy_log_selection)
        self.log_output.bind("<Command-c>", self._copy_log_selection)  # macOS
        self.log_output.bind("<Button-3>", self._show_log_menu)
        self.log_output.bind("<Key>", self._block_log_edit)
        ttk.Button(log_frame, text="Temizle", command=self._clear_log).grid(row=0, column=1, padx=5, pady=2, sticky="ns")

    # ------------------------------------------------------------------ #
    # Mode & keyboard helpers
    # ------------------------------------------------------------------ #
    def _on_key_press(self, event) -> None:  # type: ignore[override]
        if not self.controller.is_manual():
            return
        keysym = event.keysym.lower()
        key_map = {
            "up": "forward",
            "down": "backward",
            "left": "left",
            "right": "right",
        }
        action = key_map.get(keysym)
        if action:
            self._drive_button_action(action)
            return
        if keysym == "space":
            self._reset_manual_motion()
            return
        joint_cmd = JOINT_KEY_BINDINGS.get(keysym)
        if joint_cmd:
            joint, delta = joint_cmd
            current = self.controller.arm_state[joint]
            if joint in self.controller.continuous():
                self.controller.set_continuous_speed(joint, current + (JOINT_SPEED_INCREMENT if delta > 0 else -JOINT_SPEED_INCREMENT))
            else:
                self.controller.set_joint_angle(joint, current + delta)
            self.refresh_from_controller()

    def _on_mode_change(self) -> None:
        if self._updating:
            return
        selected = self.mode_var.get()
        try:
            self.controller.set_mode(selected)
        except ValueError as exc:
            messagebox.showerror("Hata", f"Gecersiz kontrol modu: {exc}")
            self.mode_var.set(self.controller.mode)
            return
        if selected == "auto":
            self._reset_manual_motion()
        else:
            self.refresh_from_controller()
        self.controller.auto_target_color = self.target_color_var.get()
        self._append_log(f"Mod {self.controller.mode} olarak ayarlandi")

    def _on_simulation_toggle(self) -> None:
        self.controller.set_simulation(self.simulation_var.get())
        self.controller.save_config()
        self._append_log(f"Simulasyon modu {'acik' if self.controller.is_simulation() else 'kapali'}")
        self.refresh_from_controller()

    def _set_run_state(self, state: str) -> None:
        try:
            self.controller.set_run_state(state)
        except ValueError as exc:
            messagebox.showerror("Hata", str(exc))
            return
        self.controller.save_config()
        self._append_log(f"Calisma durumu: {state}")
        self.refresh_from_controller()

    # ------------------------------------------------------------------ #
    def _sync_picam_color_order(self) -> None:
        if self.picam is None:
            return
        try:
            cfg = self.picam.camera_configuration()
        except Exception:
            return
        fmt = str(cfg.get("main", {}).get("format", "")).upper()
        if "BGR" in fmt:
            self._picam_color_order = "bgr"
        elif "RGB" in fmt:
            self._picam_color_order = "rgb"

    def _on_target_color_change(self) -> None:
        self.controller.auto_target_color = self.target_color_var.get()

    def _set_manual_controls_state(self, enabled: bool) -> None:
        allow = enabled and self.controller.is_running()
        for scale in self.wheel_scales.values():
            self._set_widget_state(scale, allow)
        for scale in self.joint_scales.values():
            self._set_widget_state(scale, allow)
        if self.stop_button is not None:
            self._set_widget_state(self.stop_button, allow)
        for btn in self.drive_buttons:
            self._set_widget_state(btn, allow)

    @staticmethod
    def _set_widget_state(widget, enabled: bool) -> None:
        try:
            if enabled:
                widget.state(["!disabled"])
            else:
                widget.state(["disabled"])
        except (tk.TclError, AttributeError):
            widget.configure(state="normal" if enabled else "disabled")

    # ------------------------------------------------------------------ #
    def refresh_from_controller(self) -> None:
        self._updating = True
        mode_label = f"Mod: {self.controller.mode.title()}"
        sim = self.controller.is_simulation()
        hw_label = (
            "Simulasyon modu (donanim yazmiyor)"
            if sim
            else "Donanim hazir"
            if self.controller.hardware_ready
            else "Donanim bagli degil"
        )
        self.status_var.set(f"{mode_label} | {hw_label}")
        self.mode_var.set(self.controller.mode)
        self.simulation_var.set(sim)
        self.run_state_var.set(f"Durum: {self.controller.run_state}")
        self.target_color_var.set(self.controller.auto_target_color)
        self.camera_source_var.set(self.controller.camera_source)
        self.camera_index_var.set(self.controller.camera_index)
        for name, var in self.camera_control_vars.items():
            var.set(self.controller.camera_controls.get(name, var.get()))
        self._update_camera_source_ui()

        for wheel, scale in self.wheel_scales.items():
            value = self.controller.wheel_state[wheel]
            scale.set(value)
            self.wheel_value_labels[wheel].set(f"{value:.0f}%")

        left_values = [self.controller.wheel_state[w] for w in self.controller.wheels() if "left" in w]
        right_values = [self.controller.wheel_state[w] for w in self.controller.wheels() if "right" in w]
        if left_values and right_values:
            left_avg = sum(left_values) / len(left_values)
            right_avg = sum(right_values) / len(right_values)
            self.manual_drive_level = (left_avg + right_avg) / 2
            self.manual_turn_level = (left_avg - right_avg) / 2

        for joint, scale in self.joint_scales.items():
            val = self.controller.arm_state[joint]
            scale.set(val)
            if joint in self.controller.continuous():
                self.joint_labels[joint].set(f"{val:.0f}%")
            else:
                self.joint_labels[joint].set(f"{val:.0f}°")

        self._set_manual_controls_state(self.controller.is_manual())
        # update metrics
        if self.metric_labels:
            metrics = self.controller.get_metrics()
            self.metric_labels.get("mode", tk.StringVar()).set(metrics.get("mode", ""))
            self.metric_labels.get("simulation", tk.StringVar()).set(
                "acik" if metrics.get("simulation") else "kapali"
            )
            rs = metrics.get("run_state", "")
            self.metric_labels.get("run_state", tk.StringVar()).set(rs)
            self.metric_labels.get("auto_target_color", tk.StringVar()).set(metrics.get("auto_target_color", ""))
            sv = metrics.get("supply_voltage", 0.0)
            sc = metrics.get("supply_current", 0.0)
            pw = metrics.get("power_w", 0.0)
            self.metric_labels.get("supply_voltage", tk.StringVar()).set(f"{sv:.2f} V")
            self.metric_labels.get("supply_current", tk.StringVar()).set(f"{sc:.2f} A")
            self.metric_labels.get("power_w", tk.StringVar()).set(f"{pw:.2f} W")
            freq = metrics.get("pwm_frequency_hz", 0.0)
            self.metric_labels.get("pwm_freq", tk.StringVar()).set(f"{freq:.0f} Hz")
            for wheel in self.controller.wheels():
                var = self.metric_labels.get(f"wheel_{wheel}")
                if var:
                    var.set(f"{self.controller.wheel_state[wheel]:.0f}%")
                pwm_var = self.metric_labels.get(f"pwm_{wheel}")
                if pwm_var:
                    pwm_outputs = metrics.get("pwm_outputs", {}) or {}
                    pwm_val = pwm_outputs.get(wheel, 0.0)
                    pwm_var.set(f"{pwm_val:.2f}")
            for joint in self.controller.joints():
                var = self.metric_labels.get(f"joint_{joint}")
                if var:
                    val = self.controller.arm_state[joint]
                    if joint in self.controller.continuous():
                        var.set(f"{val:.0f}%")
                    else:
                        var.set(f"{val:.0f}°")
        self._updating = False

    # ------------------------------------------------------------------ #
    def _save_state(self) -> None:
        try:
            self.controller.save_config()
            messagebox.showinfo("Basarili", "Yapilandirma kaydedildi.")
        except OSError as exc:
            LOGGER.error("Konfigurasyon kaydedilemedi: %s", exc)
            messagebox.showerror("Hata", f"Konfigurasyon kaydedilemedi: {exc}")

    def _reload_state(self) -> None:
        self.controller.load_config()
        self.refresh_from_controller()
        messagebox.showinfo("Yuklendi", "Yapilandirma yeniden yuklendi.")

    def _on_close(self) -> None:
        self._stop_stream()
        try:
            self.controller.save_config()
        except Exception:
            pass
        try:
            self.controller.cleanup()
        except Exception:
            pass
        self.root.destroy()

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="HTA2209 GUI")
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.json",
        help="Konfigurasyon dosyasi (varsayilan: config/settings.json)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO", help="Log seviyesi (DEBUG, INFO, WARNING...)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    controller = RobotController(config_path=args.config)
    controller.set_mode("manual")
    app = HTAControlGUI(controller)
    app.run()


if __name__ == "__main__":  # pragma: no cover
    main()
