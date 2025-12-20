import time
from gpiozero import DigitalOutputDevice

pins = [23,24,27,22]
devs = [DigitalOutputDevice(p) for p in pins]

print("Pins ON (3s)...")
for d in devs: d.on()
time.sleep(3)

print("Pins OFF (3s)...")
for d in devs: d.off()
time.sleep(3)

print("Done.")