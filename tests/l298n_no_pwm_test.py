import time

try:
    import RPi.GPIO as GPIO
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"RPi.GPIO import edilemedi: {exc}")

# BCM pinleri
IN1, IN2, IN3, IN4 = 23, 24, 27, 22
# ENA/ENB jumperlari takili kabul ediliyor; bu nedenle PWM pini kullanmiyoruz.
# 5V_EN jumper'i takili olmali veya kart +5V lojik besleme almalidir.


def main() -> None:
    print("L298N jumper (PWM'siz) test: ileri 2sn, geri 2sn, sonra dur.")
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in (IN1, IN2, IN3, IN4):
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)
    try:
        print("Ileri...")
        GPIO.output(IN1, GPIO.HIGH)  # sol ileri
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.LOW)   # sag ileri
        GPIO.output(IN4, GPIO.HIGH)
        time.sleep(2)

        print("Geri...")
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)
        time.sleep(2)

        print("Durdur")
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.LOW)
    finally:
        GPIO.cleanup()
        print("GPIO temizlendi, bitti.")


if __name__ == "__main__":
    main()
