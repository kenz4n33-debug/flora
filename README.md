# AI Vision TFLite Testing Web Application 🌱🤖

Website pengujian untuk menguji model **AI Computer Vision TensorFlow Lite (`AI_VISION.tflite`)** secara langsung melalui inferensi *real-time* (bukan dummy/simulasi/hardcoded).

Website ini digunakan untuk memverifikasi konsistensi hasil prediksi model `AI_VISION.tflite` sebelum dideploy ke perangkat IoT / ESP32-CAM.

---

## 🛠️ Tech Stack

### Frontend
- **React** (v18)
- **Vite**
- **JavaScript**
- **Modern & Responsive Vanilla CSS** (AI + Smart Agriculture dark theme, glassmorphism, glow indicators, progress bars)
- **Lucide React** (Modern Icons)

### Backend
- **Python** (3.12+)
- **FastAPI**
- **Uvicorn**
- **TensorFlow Lite / LiteRT Interpreter** (`ai-edge-litert`)
- **Pillow** (Image Processing)
- **NumPy**

---

## 📁 Struktur Project

```text
AI_VISION/
├── backend/
│   ├── app.py                 # FastAPI backend server with TFLite model loader & /predict endpoint
│   ├── requirements.txt       # Dependencies Python
│   └── models/
│       └── AI_VISION.tflite   # Model TensorFlow Lite
├── frontend/
│   ├── index.html             # HTML entry point
│   ├── package.json           # Frontend dependencies & scripts
│   ├── vite.config.js         # Vite configuration with proxy to http://localhost:8000
│   └── src/
│       ├── App.jsx            # Modern AI Vision Tester UI component
│       ├── index.css          # Modern AI + Smart Agriculture styling
│       └── main.jsx           # React mount entry point
└── README.md                  # Dokumentasi project
```

---

## ⚙️ Model Information & Pipeline Specs

- **Nama Model**: `AI_VISION.tflite`
- **Input Shape**: `(1, 224, 224, 3)`
- **Input Format**: Float32 RGB image
- **Internal Preprocessing**: Model sudah memiliki **Rescaling layer** internal (`Rescaling(1./255)`). Preprocessing backend **TIDAK melakukan `img / 255.0`**.
- **Urutan Kelas Output**:
  - `0` = **Healthy**
  - `1` = **Powdery**
  - `2` = **Rust**

---

## 🚀 Cara Menjalankan Project

### 1. Jalankan Backend (FastAPI)

Buka terminal di folder `backend`:

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload
```

Backend akan berjalan di: `http://localhost:8000`

Endpoints:
- `GET /`: Health status & API info
- `GET /health`: Model status & input/output tensor details
- `POST /predict`: Endpoint inferensi citra daun

### 2. Jalankan Frontend (React + Vite)

Buka terminal baru di folder `frontend`:

```bash
cd frontend
npm install
npm run dev
```

Frontend akan berjalan di: `http://localhost:5173`

---

## 📊 Format Output Predictions

Endpoint `POST /predict` mengembalikan JSON dengan struktur:

```json
{
  "prediction": "Rust",
  "confidence": 92.41,
  "probabilities": {
    "Healthy": 3.12,
    "Powdery": 4.47,
    "Rust": 92.41
  }
}
```

---

## ⚠️ Disclaimer

> "AI predictions are experimental and should not be considered a definitive diagnosis of plant disease."
