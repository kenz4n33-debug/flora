# Flora Backend API - AI Vision TFLite 🌿🚀

Backend REST API berbasis **FastAPI** dan **TensorFlow Lite (LiteRT)** untuk inferensi klasifikasi penyakit daun tanaman secara *real-time*. Siap langsung dideploy ke **Railway**.

---

## 📁 Struktur Repositori

```text
backendflora/
├── app.py                 # FastAPI backend server & inferensi TFLite
├── requirements.txt       # Python dependencies
├── Procfile               # Perintah start untuk deployment
├── railway.json           # Konfigurasi Railway deployment (Nixpacks)
├── nixpacks.toml          # Pin Python ke versi 3.11
├── .python-version        # Versi Python (3.11.9)
├── Dockerfile             # Container image (opsional jika menggunakan Docker di Railway)
├── models/
│   └── AI_VISION.tflite   # Model TFLite klasifikasi penyakit daun
└── README.md
```

---

## 🚀 Endpoint API

| Method | Endpoint | Deskripsi |
|---|---|---|
| `GET` | `/` | Cek status API & metadata |
| `GET` | `/health` | Healthcheck (verifikasi interpreter TFLite aktif) |
| `GET` | `/docs` | Dokumentasi interaktif Swagger UI |
| `POST` | `/predict` | Inferensi gambar (Multipart file, Base64 JSON, atau raw binary) |

### Format Output `/predict`:
```json
{
  "status": "success",
  "prediction": "Healthy",
  "confidence": 98.45,
  "healthy": 98.45,
  "powdery": 1.12,
  "rust": 0.43,
  "probabilities": {
    "Healthy": 98.45,
    "Powdery": 1.12,
    "Rust": 0.43
  }
}
```

---

## 🛠️ Deploy ke Railway

1. Buka [Railway.app](https://railway.app/).
2. Buat project baru -> **Deploy from GitHub repo**.
3. Pilih repositori **`kenz4n33-debug/flora`**.
4. Railway akan otomatis mendeteksi file konfigurasi (`railway.json`, `Procfile`, `nixpacks.toml`, dan `requirements.txt`).
5. Tunggu proses build selesai dan URL publik akan otomatis digenerate.

---

## 💻 Menjalankan Secara Lokal

```bash
# 1. Buat virtual environment
python -m venv .venv
source .venv/bin/activate  # atau .venv\Scripts\activate di Windows

# 2. Install dependensi
pip install -r requirements.txt

# 3. Jalankan server FastAPI
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Akses di browser: `http://localhost:8000/docs`
