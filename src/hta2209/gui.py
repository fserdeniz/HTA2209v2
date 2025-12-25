"""
Tkinter GUI for HTA2209 with camera preview, manual/auto control, and test runners.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
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

THRESHOLD_FIELDS = (
    ("hue_min", 0, 179),
    ("hue_max", 0, 179),
    ("sat_min", 0, 255),
    ("sat_max", 0, 255),
    ("val_min", 0, 255),
    ("val_max", 0, 255),
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
        self.root.title("HTA2209 - Renk Bazli Mobil Manipulator")
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

        # Bu sistemde yalnızca CSI portundaki Picamera2 kullanılacak
        self.source_type_var = tk.StringVar(value="picam")
        self.camera_index_var = tk.IntVar(value=0)
        self.camera_choice_var = tk.StringVar(value="")
        self._camera_failures = 0
        self._current_camera_index: int | None = None
        self.camera_status_var = tk.StringVar(value="Kamera kapali")
        self.folder_path_var = tk.StringVar(value="")
        self.video_path_var = tk.StringVar(value="")
        self.target_color_var = tk.StringVar(value="red")
        self.camera_label: ttk.Label | None = None
        self.camera_running = False
        self.camera_capture: cv2.VideoCapture | None = None
        self.camera_loop_id: str | None = None
        self.camera_combo: ttk.Combobox | None = None
        self._camera_photo = None
        self._last_frame = None
        self._auto_frame_counter = 0
        self._folder_images: list[Path] = []
        self._folder_index: int = 0
        self._camera_devices: list[tuple[int, str]] = []
        self.ai_detection_var = tk.BooleanVar(value=False)
        self.ai_colors_var = tk.StringVar(value="")
        self.metric_labels: Dict[str, tk.StringVar] = {}

        self.threshold_vars: Dict[str, Dict[str, tk.IntVar]] = {}
        self.wheel_scales: Dict[str, ttk.Scale] = {}
        self.wheel_value_labels: Dict[str, tk.StringVar] = {}
        self.joint_scales: Dict[str, ttk.Scale] = {}
        self.joint_labels: Dict[str, tk.StringVar] = {}
        self.stop_button: ttk.Button | None = None
        self.drive_buttons: list[ttk.Button] = []
        self.require_hardware_var = tk.BooleanVar(value=False)
        self.auto_threshold_var = tk.BooleanVar(value=False)
        self.test_output: scrolledtext.ScrolledText | None = None
        self.repo_root = Path(__file__).resolve().parents[2]

        self._create_widgets()
        if not self.picam_supported:
            self._refresh_camera_devices()
        else:
            # V4L2 taramasi yapma, dogrudan Picamera2 kullan
            self.camera_choice_var.set("Picamera2")
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
        notebook.add(self._threshold_tab(notebook), text="Renk Esikleri")
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
    def _threshold_tab(self, parent: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=10)
        for idx, color in enumerate(self.controller.colors()):
            lf = ttk.LabelFrame(frame, text=color.title(), padding=10)
            lf.grid(row=idx // 2, column=idx % 2, padx=10, pady=10, sticky="nsew")
            frame.grid_columnconfigure(idx % 2, weight=1)
            self.threshold_vars[color] = {}
            for row, (field_name, min_val, max_val) in enumerate(THRESHOLD_FIELDS):
                ttk.Label(lf, text=field_name.replace("_", " ").title()).grid(row=row, column=0, sticky="w")
                var = tk.IntVar(value=0)
                spin = tk.Spinbox(
                    lf,
                    from_=min_val,
                    to=max_val,
                    textvariable=var,
                    width=5,
                    command=self._make_threshold_callback(color, field_name, var),
                )
                spin.grid(row=row, column=1, padx=5, pady=2)
                var.trace_add("write", self._make_threshold_trace(color, field_name, var))
                self.threshold_vars[color][field_name] = var

        auto_box = ttk.LabelFrame(frame, text="Otomatik Esikleme", padding=10)
        auto_box.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        ttk.Checkbutton(
            auto_box,
            text="Ortama gore S/V otomatik uyarla (kamera karelerinden)",
            variable=self.auto_threshold_var,
            command=self._on_auto_threshold_toggle,
        ).pack(anchor="w", pady=(0, 5))
        ttk.Button(auto_box, text="Anlik Kalibre Et (guncel kare)", command=self._manual_auto_calibrate).pack(
            anchor="w"
        )
        ttk.Label(
            auto_box,
            text="Kamera acik olmalidir. Manual esikler her zaman düzenlenebilir; otomatik mod aktifken S/V esikleri karelere gore guncellenir.",
            wraplength=700,
            justify=tk.LEFT,
        ).pack(anchor="w", pady=5)

        ai_box = ttk.LabelFrame(frame, text="Yapay Zeka Kenar/Renk", padding=10)
        ai_box.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        self.ai_checkbox = ttk.Checkbutton(
            ai_box,
            text="AI destekli kenar + renk tespiti (kamera onizlemesinde)",
            variable=self.ai_detection_var,
        )
        self.ai_checkbox.pack(anchor="w", pady=(0, 5))
        ttk.Label(ai_box, textvariable=self.ai_colors_var, wraplength=700, justify=tk.LEFT).pack(anchor="w")
        return frame

    def _make_threshold_callback(self, color: str, field_name: str, var: tk.IntVar):
        def callback() -> None:
            self._on_threshold_change(color, field_name, var.get())

        return callback

    def _make_threshold_trace(self, color: str, field_name: str, var: tk.IntVar):
        def trace(*_args) -> None:
            if self._updating:
                return
            self._on_threshold_change(color, field_name, var.get())

        return trace

    def _on_threshold_change(self, color: str, field_name: str, value: int) -> None:
        self.controller.set_color_threshold(color, field_name, value)

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
        self.manual_drive_level = self._clamp_speed(self.manual_drive_level + delta)
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
        left = self._clamp_speed(self.manual_drive_level + self.manual_turn_level)
        right = self._clamp_speed(self.manual_drive_level - self.manual_turn_level)
        for wheel in self.controller.wheels():
            value = left if "left" in wheel else right
            self.controller.set_wheel_speed(wheel, value)
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
            ("Auto hedef rengi", "auto_target_color"),
            ("Otomatik esikleme", "auto_threshold"),
            ("Auto kavrama", "auto_grasped"),
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

    # ------------------------------------------------------------------ #
    def _select_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Video sec",
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov"), ("All Files", "*.*")],
        )
        if path:
            self.video_path_var.set(path)
            self.source_type_var.set("video")

    def _select_folder(self) -> None:
        path = filedialog.askdirectory(title="Resim klasoru sec")
        if path:
            self.folder_path_var.set(path)
            self.source_type_var.set("folder")

    # Camera discovery helpers ---------------------------------------- #
    def _format_camera_label(self, index: int, name: str) -> str:
        if sys.platform.startswith("win"):
            return f"Kamera {index} - {name}"
        elif sys.platform == "darwin":
            return f"AVF {index} - {name}"
        return f"/dev/video{index} - {name}"

    def _is_capture_device(self, index: int) -> bool:
        """Return True if device is a real capture node (skip ISP/codec/meta)."""
        sys_path = Path(f"/sys/class/video4linux/video{index}")
        driver = ""
        driver_link = sys_path / "device/driver"
        if driver_link.exists():
            try:
                driver = driver_link.resolve().name
            except OSError:
                driver = ""

        # Known non-capture nodes to skip
        skip_drivers = {
            "bcm2835-codec",
            "bcm2835-isp",
            "rpi-hevc-dec",
            # Unicam düğümlerini libcamera üzerinden kullanacağız; V4L2 capture denemeyelim
            "unicam",
        }
        # Allow UVC (USB) capture drivers
        allow_drivers = {"uvcvideo"}

        try:
            proc = subprocess.run(
                ["v4l2-ctl", "-d", f"/dev/video{index}", "--info"],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            # v4l2-ctl yoksa sadece driver filtresine göre karar ver
            if driver and driver in skip_drivers:
                return False
            return not driver or driver in allow_drivers

        if proc.returncode != 0:
            return False
        stdout = proc.stdout

        # Filter by driver name from v4l2-ctl if available
        if "Driver name" in stdout:
            driver_name = stdout.split("Driver name", 1)[1]
            if any(skip in driver_name for skip in skip_drivers):
                return False

        has_capture = "Device Caps" in stdout and "Video Capture" in stdout.split("Device Caps", 1)[1]
        return has_capture

    def _refresh_camera_devices(self) -> None:
        # Yalnızca Picamera2 destekleniyor; USB taraması yapılmaz.
        self._camera_devices = []
        if self.picam_supported:
            self.camera_choice_var.set("Picamera2")
            self.camera_status_var.set("Picamera2 hazir")
        else:
            self.camera_choice_var.set("Picamera2 bulunamadi")
            self.camera_status_var.set("Picamera2 bulunamadi")
        self._update_camera_combo()

    def _update_camera_combo(self) -> None:
        # Picamera2 haric kaynak yok; combobox kullanılmıyor.
        if self.camera_combo is None:
            return
        self.camera_combo["values"] = []

    def _on_camera_choice(self, _event=None) -> None:
        choice = self.camera_choice_var.get()
        for idx, name in self._camera_devices:
            if choice.startswith(f"/dev/video{idx}") or choice.startswith(f"Kamera {idx}") or choice.startswith(
                f"AVF {idx}"
            ):
                self.camera_index_var.set(idx)
                return

    def _start_stream(self) -> None:
        self._stop_stream()
        self._auto_frame_counter = 0

        # Yalnızca Picamera2 kullanılacak
        if not self.picam_supported:
            self.camera_status_var.set("Picamera2 bulunamadı; yalnızca CSI destekleniyor.")
            messagebox.showerror("Kamera", "Picamera2 modulu bulunamadı veya yüklenemedi.")
            return
        try:
            if self.picam is None:
                self.picam = Picamera2()
            self._picam_color_order = "rgb"
            try:
                config = self.picam.create_video_configuration(main={"size": (1280, 720), "format": "RGB888"})
                self.picam.configure(config)
            except Exception:
                config = self.picam.create_video_configuration(main={"size": (1280, 720), "format": "BGR888"})
                self.picam.configure(config)
                self._picam_color_order = "bgr"
            self._sync_picam_color_order()
            self.picam.start()
            self.source_type_var.set("picam")
            self.camera_running = True
            self._camera_failures = 0
            self._current_camera_index = None
            self.camera_status_var.set("Kamera açık (Picamera2 1280x720)")
            self._update_picam_frame()
            return
        except Exception as exc:
            self.camera_status_var.set(f"Picamera2 açılamadı: {exc}")
            messagebox.showerror("Kamera", f"Picamera2 açılamadı: {exc}")

    def _stop_stream(self) -> None:
        self.camera_running = False
        if self.camera_loop_id is not None:
            self.root.after_cancel(self.camera_loop_id)
            self.camera_loop_id = None
        if self.camera_capture is not None:
            self.camera_capture.release()
            self.camera_capture = None
        if self.picam is not None:
            try:
                self.picam.stop()
            except Exception:
                pass
        self.camera_status_var.set("Kaynak kapali")
        if self.camera_label is not None:
            self.camera_label.configure(image="", text="Onizleme yok")
        self._camera_photo = None
        self._last_frame = None
        self._folder_images = []
        self._folder_index = 0
        self._camera_failures = 0
        self._current_camera_index = None

    def _update_camera_frame(self) -> None:
        if not self.camera_running:
            return

        source = self.source_type_var.get()
        if source == "picam":
            self._update_picam_frame()
            return
        if source == "folder":
            self._update_folder_frame()
            return
        if self.camera_capture is None:
            return

        ret, frame = self.camera_capture.read()
        if not ret:
            if source == "video":
                self.camera_status_var.set("Video bitti")
                self._stop_stream()
            else:
                self._camera_failures += 1
                # Sık aç/kapa yapmamak için sadece bekle ve yeniden dene
                self.camera_status_var.set(
                    f"Kare okunamadi, tekrar denenecek... (deneme {self._camera_failures})"
                )
                self.camera_loop_id = self.root.after(CAMERA_FRAME_INTERVAL_MS, self._update_camera_frame)
            return
        self._camera_failures = 0
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._last_frame = frame_rgb
        display_frame = frame_rgb

        if self.ai_detection_var.get():
            overlay, colors = detector.analyze_frame(self._last_frame)
            display_frame = overlay
            if colors:
                color_names = ", ".join([c[0] for c in colors])
                self.ai_colors_var.set(f"Dominant renkler: {color_names}")
            else:
                self.ai_colors_var.set("Dominant renk bulunamadi")
        else:
            self.ai_colors_var.set("")

        dominant_list = None
        if self.ai_detection_var.get() and self.ai_colors_var.get():
            names = self.ai_colors_var.get().replace("Dominant renkler:","").strip()
            if names:
                dominant_list = [n.strip() for n in names.split(",") if n.strip()]

        if not self.controller.is_manual():
            hsv = cv2.cvtColor(self._last_frame, cv2.COLOR_RGB2HSV)
            self.controller.autopilot_step(hsv, (frame.shape[1], frame.shape[0]))
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
            # Simulasyon modunda da PWM/matrixleri canlı görmek için yenile
            self.refresh_from_controller()

        image = Image.fromarray(display_frame)
        image = image.resize(CAMERA_PREVIEW_SIZE, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image=image)
        if self.camera_label is not None:
            self.camera_label.configure(image=photo, text="")
            self.camera_label.image = photo  # type: ignore[attr-defined]
        self._camera_photo = photo

        self._auto_frame_counter += 1
        if self.controller.auto_threshold_enabled and self._auto_frame_counter % 10 == 0:
            hsv = cv2.cvtColor(self._last_frame, cv2.COLOR_RGB2HSV)
            self.controller.auto_calibrate_from_frame(hsv)
            self._refresh_threshold_fields()

        self.camera_loop_id = self.root.after(CAMERA_FRAME_INTERVAL_MS, self._update_camera_frame)

    def _update_folder_frame(self) -> None:
        if not self.camera_running or not self._folder_images:
            return
        image_path = self._folder_images[self._folder_index]
        frame = cv2.imread(str(image_path))
        if frame is None:
            self.camera_status_var.set(f"Resim okunamadi: {image_path.name}")
            self.ai_colors_var.set("")
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._last_frame = frame_rgb
            display_frame = frame_rgb
            if self.ai_detection_var.get():
                overlay, colors = detector.analyze_frame(self._last_frame)
                display_frame = overlay
                if colors:
                    color_names = ", ".join([c[0] for c in colors])
                    self.ai_colors_var.set(f"Dominant renkler: {color_names}")
                else:
                    self.ai_colors_var.set("Dominant renk bulunamadi")
            else:
                self.ai_colors_var.set("")

            dominant_list = None
            if self.ai_detection_var.get() and self.ai_colors_var.get():
                names = self.ai_colors_var.get().replace("Dominant renkler:","").strip()
                if names:
                    dominant_list = [n.strip() for n in names.split(",") if n.strip()]

            if not self.controller.is_manual():
                hsv = cv2.cvtColor(self._last_frame, cv2.COLOR_RGB2HSV)
                self.controller.autopilot_step(hsv, (frame.shape[1], frame.shape[0]))
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

            image = Image.fromarray(display_frame)
            image = image.resize(CAMERA_PREVIEW_SIZE, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image=image)
            if self.camera_label is not None:
                self.camera_label.configure(image=photo, text="")
                self.camera_label.image = photo  # type: ignore[attr-defined]
            self._camera_photo = photo

            if self.controller.auto_threshold_enabled and self._auto_frame_counter % 2 == 0:
                hsv = cv2.cvtColor(self._last_frame, cv2.COLOR_RGB2HSV)
                self.controller.auto_calibrate_from_frame(hsv)
                self._refresh_threshold_fields()
        self._auto_frame_counter += 1
        self._folder_index = (self._folder_index + 1) % len(self._folder_images)
        self.camera_loop_id = self.root.after(CAMERA_FRAME_INTERVAL_MS, self._update_folder_frame)

    def _update_picam_frame(self) -> None:
        if not self.camera_running or self.picam is None:
            return
        try:
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
        except Exception as exc:
            self._camera_failures += 1
            if self._camera_failures >= 5:
                self.camera_status_var.set(f"Picamera2 kare okunamadi ({exc}), durduruluyor.")
                self._stop_stream()
                return
            self.camera_loop_id = self.root.after(CAMERA_FRAME_INTERVAL_MS, self._update_picam_frame)
            return

        self._camera_failures = 0
        # Normalize Picamera2 output to RGB for the OpenCV pipeline.
        self._last_frame = frame
        display_frame = self._last_frame

        if self.ai_detection_var.get():
            overlay, colors = detector.analyze_frame(self._last_frame)
            display_frame = overlay
            if colors:
                color_names = ", ".join([c[0] for c in colors])
                self.ai_colors_var.set(f"Dominant renkler: {color_names}")
            else:
                self.ai_colors_var.set("Dominant renk bulunamadi")
        else:
            self.ai_colors_var.set("")

        dominant_list = None
        if self.ai_detection_var.get() and self.ai_colors_var.get():
            names = self.ai_colors_var.get().replace("Dominant renkler:","").strip()
            if names:
                dominant_list = [n.strip() for n in names.split(",") if n.strip()]

        if not self.controller.is_manual():
            hsv = cv2.cvtColor(self._last_frame, cv2.COLOR_RGB2HSV)
            self.controller.autopilot_step(hsv, (display_frame.shape[1], display_frame.shape[0]))
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

        image = Image.fromarray(display_frame)
        image = image.resize(CAMERA_PREVIEW_SIZE, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image=image)
        if self.camera_label is not None:
            self.camera_label.configure(image=photo, text="")
            self.camera_label.image = photo  # type: ignore[attr-defined]
        self._camera_photo = photo

        self._auto_frame_counter += 1
        if self.controller.auto_threshold_enabled and self._auto_frame_counter % 10 == 0:
            hsv = cv2.cvtColor(self._last_frame, cv2.COLOR_RGB2HSV)
            self.controller.auto_calibrate_from_frame(hsv)
            self._refresh_threshold_fields()

        self.camera_loop_id = self.root.after(CAMERA_FRAME_INTERVAL_MS, self._update_picam_frame)

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
        ttk.Label(color_frame, text="Auto hedef rengi (kilitli: sari/kirmizi):").pack(side=tk.LEFT)
        color_combo = ttk.Combobox(
            color_frame,
            values=list(self.controller.colors()),
            textvariable=self.target_color_var,
            state="disabled",
            width=10,
        )
        color_combo.pack(side=tk.LEFT, padx=5)
        color_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_target_color_change())

        ttk.Button(settings, text="Kaydet", command=self._save_state).pack(fill=tk.X, pady=2)
        ttk.Button(settings, text="Yeniden Yukle", command=self._reload_state).pack(fill=tk.X, pady=2)

        cam_box = ttk.LabelFrame(sidebar, text="Kamera Kontrolu (Picamera2 - CSI)", padding=10)
        cam_box.pack(fill=tk.X, pady=(0, 10))
        row = 0
        ttk.Label(cam_box, text="Kaynak: Raspberry Pi Camera Module v3 (Picamera2)").grid(
            row=row, column=0, columnspan=4, sticky="w"
        )
        row += 1

        ttk.Button(cam_box, text="Baslat", command=self._start_stream).grid(row=row, column=0, columnspan=2, padx=5, pady=2, sticky="ew")
        ttk.Button(cam_box, text="Durdur", command=self._stop_stream).grid(row=row, column=2, columnspan=2, padx=5, pady=2, sticky="ew")
        row += 1

        ttk.Label(cam_box, textvariable=self.camera_status_var).grid(row=row, column=0, columnspan=4, sticky="w")

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
        if keysym == "up":
            self._adjust_drive(MANUAL_DRIVE_INCREMENT)
            return
        if keysym == "down":
            self._adjust_drive(-MANUAL_DRIVE_INCREMENT)
            return
        if keysym == "left":
            self._adjust_turn(-MANUAL_TURN_INCREMENT)
            return
        if keysym == "right":
            self._adjust_turn(MANUAL_TURN_INCREMENT)
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
            # Auto modda AI destekli tespit otomatik acilsin
            self.ai_detection_var.set(True)
            try:
                self.ai_checkbox.state(["disabled"])
            except Exception:
                pass
        else:
            self.refresh_from_controller()
            try:
                self.ai_checkbox.state(["!disabled"])
            except Exception:
                pass
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

    def _on_auto_threshold_toggle(self) -> None:
        enabled = self.auto_threshold_var.get()
        self.controller.set_auto_threshold_enabled(enabled)

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

    def _camera_backends(self) -> list[int]:
        if sys.platform.startswith("win"):
            return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        if sys.platform == "darwin":
            return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
        return [cv2.CAP_V4L2, cv2.CAP_ANY]

    def _check_camera_available(self, index: int) -> bool:
        cap = self._open_camera_capture(index)
        if cap is None:
            return False
        cap.release()
        return True

    def _open_camera_capture(self, index: int) -> cv2.VideoCapture | None:
        """Open camera with a safe default format and verify frame read."""
        for backend in self._camera_backends():
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            # Safer defaults for USB2 cams (Linux). Windows/mac'te varsayilan format kullan
            if not sys.platform.startswith("win") and sys.platform != "darwin":
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 10)
            for _ in range(3):
                ret, frame = cap.read()
                if ret and frame is not None:
                    return cap
            cap.release()
        return None

    def _manual_auto_calibrate(self) -> None:
        if self._last_frame is None:
            messagebox.showerror("Kamera", "Kamera acik degil veya kare yok.")
            return
        hsv = cv2.cvtColor(self._last_frame, cv2.COLOR_RGB2HSV)
        self.controller.auto_calibrate_from_frame(hsv)
        self._refresh_threshold_fields()
        messagebox.showinfo("Otomatik", "S/V esikleri guncellendi (guncel kare).")

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
    def _refresh_threshold_fields(self) -> None:
        self._updating = True
        for color, fields in self.threshold_vars.items():
            threshold = self.controller.color_thresholds[color]
            for field_name, var in fields.items():
                var.set(getattr(threshold, field_name))
        self._updating = False

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
        self.auto_threshold_var.set(self.controller.auto_threshold_enabled)
        self.target_color_var.set(self.controller.auto_target_color)

        self._refresh_threshold_fields()

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
            self.metric_labels.get("auto_threshold", tk.StringVar()).set(
                "acik" if metrics.get("auto_threshold") else "kapali"
            )
            self.metric_labels.get("auto_grasped", tk.StringVar()).set(
                "tamamlandi" if metrics.get("auto_grasped") else "bekleniyor"
            )
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
    app = HTAControlGUI(controller)
    app.run()


if __name__ == "__main__":  # pragma: no cover
    main()
