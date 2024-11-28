## Deteksi Larangan Parkir Menggunakan YOLOv8

### Instalasi Requirements
```bash
pip install -r requirements.txt
```
### Menggambar Area Terlarang
1. Jalankan script line_drawer.py 
```bash
python line_drawer.py
```
2. Pilih gambar yang ingin digunakan
3. Gambar area yang ingin dijadikan area terlarang parkir, kemudian simpan dengan menekan "s"

### Deteksi Kendaraan
1. Ubah script deteksi-larangan-parkir.py
```bash
video_path = os.path.join(current_dir, "areabaru.mp4") #sesuaikan dengan path video yang digunakan
output_path = os.path.join(current_dir, "output_video_detection7.mp4") #pilih nama output
model_path = os.path.join(current_dir, "bestv8.pt") #sesuaikan dengan model yang anda gunakan
areas_path = os.path.join(current_dir, "restricted_areas.json") #sesuaikan dengan file json anda
logo_path = os.path.join(current_dir, "augenio.png") #opsional
```

2. jalankan script 
```bash
python deteksi-larangan-parkir.py
```

