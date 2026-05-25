import json
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import tensorflow as tf

app = FastAPI(title="Rice Disease Classifier", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model & class names on startup
model = None
class_names = []

@app.on_event("startup")
def load_model():
    global model, class_names
    model = tf.keras.models.load_model("rice_model.keras",safe_mode=False )
    with open("class_names.json") as f:
        class_names = json.load(f)
    print(f"Model loaded. Classes: {class_names}")


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)


@app.get("/")
def root():
    return {"status": "ok", "model": "Rice Disease Classifier", "classes": class_names}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")

    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)

    preds = model.predict(img_array)[0]  # shape: (6,)
    top_idx = int(np.argmax(preds))

    results = [
        {"class": class_names[i], "confidence": round(float(preds[i]) * 100, 2)}
        for i in range(len(class_names))
    ]
    results.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "prediction": class_names[top_idx],
        "confidence": round(float(preds[top_idx]) * 100, 2),
        "all_scores": results,
    }
