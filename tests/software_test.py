"""
Lightweight software sanity checks for HTA2209.
- Validates core Python dependencies are importable.
- Instantiates RobotController to verify configuration parsing (donanım gerektirmeden).

Exit code:
0 -> tüm temel kontroller geçti (donanım bağımlı uyarılar serbest)
1 -> bir zorunlu bağımlılık veya denetim başarısız
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

MODULES = [
    "cv2",
    "PIL",  # pillow
    "numpy",
    "adafruit_servokit",  # may warn on unsupported platforms
    "dotenv",
    "ultralytics",
]


def check_import(name: str) -> tuple[bool, Exception | None]:
    try:
        importlib.import_module(name)
        return True, None
    except Exception as exc:
        return False, exc


def main() -> None:
    failures: list[str] = []
    warnings: list[str] = []
    ok_count = 0
    require_hw = os.getenv("REQUIRE_HARDWARE", "0").lower() in ("1", "true", "yes", "on")

    for name in MODULES:
        ok, exc = check_import(name)
        if ok:
            ok_count += 1
            print(f"[OK] import {name}")
            continue
        if name == "adafruit_servokit" and isinstance(exc, NotImplementedError):
            warnings.append(f"adafruit_servokit not usable on this platform: {exc}")
            print(f"[WARN] adafruit_servokit import skipped (platform not supported): {exc}")
            continue
        failures.append(f"{name} import failed: {exc}")
        print(f"[FAIL] import {name}: {exc}")

    try:
        from hta2209.controller import RobotController

        RobotController(config_path=Path("config/test_settings.json"))
        ok_count += 1
        print("[OK] RobotController instantiate edildi")
    except Exception as exc:  # pragma: no cover - runtime safety
        msg = f"RobotController init failed: {exc}"
        if require_hw:
            failures.append(msg)
            print(f"[FAIL] {msg}")
        else:
            warnings.append(msg)
            print(f"[WARN] {msg} (REQUIRE_HARDWARE=0)")

    if warnings:
        print(f"[WARN] {len(warnings)} warning(s):")
        for item in warnings:
            print(f"       - {item}")

    if failures:
        print(f"[FAIL] Software test failed with {len(failures)} error(s).")
        sys.exit(1)

    print(f"[OK] Software test passed. Checks={ok_count}, warnings={len(warnings)}")
    sys.exit(0)


if __name__ == "__main__":
    main()
