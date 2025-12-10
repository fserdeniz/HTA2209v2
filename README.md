# RPi Renk Tabanli Mobil Manipulator

Bu depo, Raspberry Pi 4B (8 GB) uzerinde calisan, renkleri algilayarak hedef nesnelere yaklasan ve PCA9685 PWM surucu karti uzerinden hem mobil platformu hem de robotik kolu yoneten sistem icin baslangic altyapisini sunar.

## Donanim Hedefi
- Raspberry Pi 4B 8 GB
- PCA9685 tabanli PWM motor/servo surucu
- 4 tekerlekli diferansiyel surus tabani
- Servo tabanli robotik kol
- CSI veya USB kamera (renk algilama icin)

## Yazilim Bilesenleri
- `src/hta2209/controller.py`: PCA9685 kanallarini soyutlayan, tekerlek hizlari, eklem acilari ve HSV esiklerini saklayip kaydeden denetleyici. Donanim bagli degilse otomatik olarak simulasyon moduna gecer.
- `src/hta2209/gui.py`: Tkinter ile yazilmis GUI; renk esikleri, surus ve kol kontrolu, Manual/Auto mod secimi ve OpenCV + Pillow kullanarak kamera onizlemesi sunar.
- `requirements.txt`: OpenCV, NumPy, Adafruit suruculeri ve Pillow dahil gerekli Python paketleri.

## Kurulum
1. **Depoyu klonlayin**
   ```powershell
   git clone <repo-url>
   cd HTA2209
   ```
2. **Sanal ortami olusturun (ilk sefer)**
   ```powershell
   python -m venv 2209_env
   ```
3. **Sanal ortami etkinlestirin**
   - PowerShell
     ```powershell
     .\2209_env\Scripts\Activate.ps1
     ```
   - Linux/macOS
     ```bash
     source 2209_env/bin/activate
     ```
4. **Bagimliliklari kurun**
   ```powershell
   pip install -r requirements.txt
   ```
5. **Paketi editable modda kurun** (modulun `python -m hta2209.gui` ile bulunabilmesi icin)
   ```powershell
   pip install -e .
   ```

## GUI Kullanim
GUI hem Raspberry Pi uzerinde hem de gelistirme bilgisayarinda calisir. SSH ile baglaniyorsaniz X11 forwarding (Linux/macOS) veya VcXsrv/Xming (Windows) ile goruntuleyebilirsiniz.

```bash
python -m hta2209.gui --config config/settings.json --log-level INFO
```

- **Kısayol komutu (her OS)**: Paketi `pip install -e .` ile kurduktan sonra sanal ortam acikken
  ```bash
  hta2209-gui --config config/settings.json --log-level INFO
  ```
  komutu GUI'yi dogrudan baslatir; Windows/Mac/Linux hepsinde gecerli bir console script olusturuldu.

- **Orta panel (Kamera)**: Kamera onizlemesi her zaman merkezde kalir, ortam aydinlatmasi icin otomatik S/V uyarlama yapabilir. Sol panelden kamera indexini secip Baslat/Durdur yapabilirsiniz.
- **Sol panel (Ayarlar)**: Manual/Auto mod secimi, konfig kaydet/yukle ve kamera kontrol butonlari.
- **Sag sekmeler**:
  - **Baglanti**: Donanimin hazir olup olmadigini ve konfig dosyasini gosterir.
  - **Renk Esikleri**: HSV bandlarini (H min/max, S min/max, V min/max) ayarlayin.
  - **Surus Kontrol**: Her tekerlek icin -100/+100 arasi hiz; "Tekerlekleri Durdur" tum throttle'i sifirlar.
  - **Robot Kol**: Baz, omuz, dirsek, bilek ve gripper servo acilarini 0-180 derece arasinda degistirin.

GUI ust kismindaki "Kaydet" mevcut durumu `config/settings.json` dosyasina yazar; "Yeniden Yukle" ayni dosyadan geri alir. Dosya yoksa otomatik olusturulur.

## Kontrol Modlari ve Kisayollar
- **Manual**: Tum slider'lar aktif, klavye kontrolu acik.
- **Auto**: Manuel girdiler kilitlenir; otonom algoritmalarinizi calistirmak icin kullanin.

Klavye (Manual mod):
- Yon tuslari: Ileri/geri hiz ve donus bilesenlerini +/-10 degistirir (`Space` = acil durdur).
- Q/A: Base, W/S: Shoulder, E/D: Elbow, R/F: Wrist, T/G: Gripper acilarini +/-5 degistirir.

## Testler
- Yazilim testi (bagimlilik ve kontrolcu baslangic sagligi):
  ```bash
  python tests/software_test.py
  ```
- Donanim testi (PCA9685 ve kamera yoksa uyari verir; zorunlu kilmak icin `REQUIRE_HARDWARE=1`):
  ```bash
  python tests/hardware_test.py
  # veya donanim zorunlu
  $env:REQUIRE_HARDWARE=1; python tests/hardware_test.py
  ```
- GUI uzerinden: Sol paneldeki **Testler** bolumunden SOFTWARE_TEST/HARDWARE_TEST butonlariyla calistirabilir, ciktiyi ayni panelde gorebilirsiniz.

## SSH Uzerinden Calistirma
- **Linux/macOS**: `ssh -X pi@raspberrypi.local` ardindan sanal ortami etkinlestirip `python -m hta2209.gui`.
- **Windows**: VcXsrv/Xming'i baslatin, `ssh -X` ile baglanin. Isterseniz GUI'yi tamamen yerelde de calistirabilirsiniz; donanim yoksa simulasyon modunda kalir.

## Sonraki Adimlar
- OpenCV tabanli renk tespiti ve hedef takibini `src/` altinda yeni modullerle ekleyin.
- Auto moduna otonom surus/kol stratejilerini baglayin.
- PCA9685 kanal konfiglerini `.env` veya ek JSON dosyalariyla parametrik hale getirin.
