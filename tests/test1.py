#!/usr/bin/env python3
# L298N 4WD (tek L298N, 4 motor paralel 2 kanal) - ENA/ENB jumperlı (PWM yok)
# RPi4B -> IN1-IN4 ile yön kontrol hareket testi (güncel pinlerinle)

import time
import sys

try:
    from gpiozero import DigitalOutputDevice
except ImportError:
    print("gpiozero yüklü değil. Kurmak için:")
    print("  sudo apt update && sudo apt install -y python3-gpiozero")
    sys.exit(1)

# --- PIN MAP (BCM) / SENİN BAĞLANTIN ---
IN1_PIN = 23  # (senin dediğin) IN1 -> GPIO23
IN2_PIN = 24  # IN2 -> GPIO24
IN3_PIN = 27  # IN3 -> GPIO27
IN4_PIN = 22  # IN4 -> GPIO22

# --- Setup outputs ---
in1 = DigitalOutputDevice(IN1_PIN)
in2 = DigitalOutputDevice(IN2_PIN)
in3 = DigitalOutputDevice(IN3_PIN)
in4 = DigitalOutputDevice(IN4_PIN)

def stop():
    """Motorları durdur (coast)."""
    in1.off(); in2.off(); in3.off(); in4.off()

def forward():
    """İleri: Kanal A ileri, Kanal B ileri"""
    # A: IN1=1 IN2=0, B: IN3=0 IN4=1
    in1.on();  in2.off()
    in3.off();  in4.on()

def backward():
    """Geri: Kanal A geri, Kanal B geri"""
    # A: IN1=0 IN2=1, B: IN3=1 IN4=0
    in1.off(); in2.on()
    in3.on();  in4.off()

def turn_left():
    """Sola dön (pivot): sol geri, sağ ileri"""
    # Sol tarafı Kanal A varsayıyoruz, sağ taraf Kanal B
    in1.off(); in2.on()   # A geri
    in3.on();  in4.off()  # B ileri

def turn_right():
    """Sağa dön (pivot): sol ileri, sağ geri"""
    in1.on();  in2.off()  # A ileri
    in3.off(); in4.on()   # B geri

def step(name, fn, duration=1.5, pause=0.7):
    print(f"[TEST] {name}  ({duration:.2f}s)")
    fn()
    time.sleep(duration)
    print("[TEST] STOP")
    stop()
    time.sleep(pause)

def main():
    try:
        print("L298N hareket testi basliyor (ENA/ENB jumperli, PWM yok).")
        print("Pinler (BCM): IN1=23 IN2=24 IN3=27 IN4=22")
        time.sleep(1.0)

        step("FORWARD", forward, 2.0)
        step("BACKWARD", backward, 2.0)
        step("TURN LEFT (pivot)", turn_left, 1.5)
        step("TURN RIGHT (pivot)", turn_right, 1.5)

        # Kablolama/yon kontrolü için kısa darbeler
        for i in range(3):
            step(f"PULSE FORWARD #{i+1}", forward, 0.35, 0.35)

        print("Test bitti. Motorlar durduruldu.")
        stop()

    except KeyboardInterrupt:
        print("\nCTRL+C alindi. Motorlar durduruluyor...")
        stop()

if __name__ == "__main__":
    main()
