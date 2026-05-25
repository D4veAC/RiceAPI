import json
import os
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import google.generativeai as genai

app = FastAPI(title="Rice Disease Classifier", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
class_names = []
gemini_model = None

@app.on_event("startup")
def load_model():
    global model, class_names, gemini_model

    # Load TF model
    model = tf.keras.models.load_model("rice_model.keras", safe_mode=False)
    with open("class_names.json") as f:
        class_names = json.load(f)
    print(f"Model loaded. Classes: {class_names}")

    # Load Gemini
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel("gemini-2.0-flash-lite")
        print("Gemini loaded.")
    else:
        print("WARNING: GEMINI_API_KEY not set. Running without Gemini validation.")


@app.get("/")
def root():
    return {
        "status": "ok",
        "model": "Rice Disease Classifier",
        "classes": class_names,
        "gemini_enabled": gemini_model is not None
    }


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def validate_with_gemini(image_bytes: bytes) -> tuple[bool, str]:
    """
    Returns (is_rice_leaf, reason)
    """
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        response = gemini_model.generate_content([
            pil_img,
            (
                "Look at this image carefully. "
                "Is this a close-up photo of a rice plant leaf? "
                "Answer with ONLY 'yes' or 'no', nothing else."
            )
        ])
        answer = response.text.strip().lower()
        return ("yes" in answer), answer
    except Exception as e:
        print(f"Gemini error: {e}")
        return True, "gemini_error"  # fallback: lanjut ke model


def recheck_with_gemini(image_bytes: bytes, model_prediction: str, confidence: float) -> dict:
    """
    Recheck kalau confidence rendah (0.6-0.75).
    Returns dict dengan gemini_says dan final_prediction.
    """
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        class_list = ", ".join(class_names)
        response = gemini_model.generate_content([
            pil_img,
            (
                f"This is a rice leaf image. My model predicted '{model_prediction}' "
                f"with {confidence:.1f}% confidence but I'm not sure. "
                f"The possible diseases are: {class_list}. "
                f"What disease does this rice leaf most likely have? "
                f"Answer with ONLY the exact class name from the list, nothing else."
            )
        ])
        gemini_answer = response.text.strip()
        # Cek apakah jawaban Gemini ada di class_names
        matched = next((c for c in class_names if c.lower() in gemini_answer.lower()), None)
        return {
            "gemini_says": gemini_answer,
            "matched_class": matched
        }
    except Exception as e:
        print(f"Gemini recheck error: {e}")
        return {"gemini_says": None, "matched_class": None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")

    image_bytes = await file.read()

    # ── Step 1: Gemini gatekeeper ─────────────────────────────────────────
    if gemini_model is not None:
        is_leaf, gemini_raw = validate_with_gemini(image_bytes)
        if not is_leaf:
            return {
                "prediction": "Not a rice leaf",
                "confidence": 0.0,
                "all_scores": [],
                "message": "Gambar tidak dikenali sebagai daun padi. Pastikan foto menampilkan daun padi secara jelas.",
                "validated_by": "gemini"
            }

    # ── Step 2: Model predict ─────────────────────────────────────────────
    img_array = preprocess_image(image_bytes)
    preds = model.predict(img_array)[0]
    top_idx = int(np.argmax(preds))
    top_conf = float(preds[top_idx])

    results = [
        {"class": class_names[i], "confidence": round(float(preds[i]) * 100, 2)}
        for i in range(len(class_names))
    ]
    results.sort(key=lambda x: x["confidence"], reverse=True)

    # ── Step 3: Confidence check ──────────────────────────────────────────
    # Kalau sangat rendah → tolak
    if top_conf < 0.5:
        return {
            "prediction": "Not a rice leaf",
            "confidence": round(top_conf * 100, 2),
            "all_scores": results,
            "message": "Gambar tidak dikenali sebagai daun padi. Pastikan foto menampilkan daun padi secara jelas.",
            "validated_by": "threshold"
        }

    # ── Step 4: Gemini recheck kalau confidence 0.5-0.75 ─────────────────
    final_prediction = class_names[top_idx]
    validated_by = "model"

    if gemini_model is not None and top_conf < 0.75:
        recheck = recheck_with_gemini(image_bytes, final_prediction, top_conf * 100)
        if recheck["matched_class"]:
            final_prediction = recheck["matched_class"]
            validated_by = "gemini_recheck"

    return {
        "prediction": final_prediction,
        "confidence": round(top_conf * 100, 2),
        "all_scores": results,
        "validated_by": validated_by
    }
