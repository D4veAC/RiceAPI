import json
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = FastAPI(title="Rice Disease Classifier", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
class_names = []

@app.on_event("startup")
def load_model():
    global model, class_names
    model = tf.keras.models.load_model("rice_model.keras", safe_mode=False)
    with open("class_names.json") as f:
        class_names = json.load(f)
    print(f"Model loaded. Classes: {class_names}")

@app.get("/")
def root():
    return {"status": "ok", "model": "Rice Disease Classifier", "classes": class_names}

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")

    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)

    preds = model.predict(img_array)[0]
    top_idx = int(np.argmax(preds))
    top_conf = float(preds[top_idx])

    results = [
        {"class": class_names[i], "confidence": round(float(preds[i]) * 100, 2)}
        for i in range(len(class_names))
    ]
    results.sort(key=lambda x: x["confidence"], reverse=True)

    if top_conf < 0.6:
        return {
            "prediction": "Not a rice leaf",
            "confidence": round(top_conf * 100, 2),
            "all_scores": results,
            "message": "Gambar tidak dikenali sebagai daun padi. Pastikan foto menampilkan daun padi secara jelas."
        }

    return {
        "prediction": class_names[top_idx],
        "confidence": round(top_conf * 100, 2),
        "all_scores": results,
    }