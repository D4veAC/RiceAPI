# Rice Disease Classifier API

FastAPI backend untuk klasifikasi penyakit daun padi menggunakan Keras model.

## Classes
- Bacterial Leaf Blight
- Brown Spot
- Healthy Rice Leaf
- Leaf Blast
- Leaf Scald
- Sheath Blight

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check + info model |
| POST | `/predict` | Upload gambar → prediksi penyakit |
| GET | `/docs` | Swagger UI (auto-generated) |

## Cara pakai `/predict`

```bash
curl -X POST "https://your-app.railway.app/predict" \
  -F "file=@foto_padi.jpg"
```

Response:
```json
{
  "prediction": "Leaf Blast",
  "confidence": 94.21,
  "all_scores": [
    {"class": "Leaf Blast", "confidence": 94.21},
    {"class": "Brown Spot", "confidence": 3.12},
    ...
  ]
}
```

## Deploy ke Railway

1. Push repo ini ke GitHub
2. Connect repo di [railway.app](https://railway.app)
3. Railway auto-detect Procfile dan deploy
4. Done — endpoint aktif dalam ~3 menit

## Run lokal

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```
