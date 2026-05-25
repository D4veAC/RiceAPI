import json
import os
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from google import genai
from google.genai import types

app = FastAPI(title="Rice Disease Classifier", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
class_names = []
gemini_client = None

@app.on_event("startup")
def load_model():
    global model, class_names, gemini_client

    model = tf.keras.models.load_model("rice_model.keras", safe_mode=False)
    with open("class_names.json") as f:
        class_names = json.load(f)
    print(f"Model loaded. Classes: {class_names}")

    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        gemini_client = genai.Client(api_key=api_key)
        print("Gemini 2.5 Flash loaded.")
    else:
        print("WARNING: GEMINI_API_KEY not set. Running without Gemini validation.")


@app.get("/")
def root():
    return {
        "status": "ok",
        "model": "Rice Disease Classifier",
        "classes": class_names,
        "gemini_enabled": gemini_client is not None
    }


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def resize_for_gemini(image_bytes: bytes) -> bytes:
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pil_img = pil_img.resize((224, 224))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def validate_with_gemini(image_bytes: bytes) -> tuple[bool, str]:
    try:
        img_bytes = resize_for_gemini(image_bytes)
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                types.Part.from_text(text="Is this a close-up photo of a rice plant leaf? Answer with ONLY 'yes' or 'no', nothing else.")
            ]
        )
        answer = response.text.strip().lower()
        return ("yes" in answer), answer
    except Exception as e:
        print(f"Gemini error: {e}")
        return True, "gemini_error"


def recheck_with_gemini(image_bytes: bytes, model_prediction: str, confidence: float) -> dict:
    try:
        img_bytes = resize_for_gemini(image_bytes)
        class_list = ", ".join(class_names)
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                types.Part.from_text(
                    text=(
                        f"This is a rice leaf image. My model predicted '{model_prediction}' "
                        f"with {confidence:.1f}% confidence but I'm not sure. "
                        f"The possible diseases are: {class_list}. "
                        f"What disease does this rice leaf most likely have? "
                        f"Answer with ONLY the exact class name from the list, nothing else."
                    )
                )
            ]
        )
        gemini_answer = response.text.strip()
        matched = next((c for c in class_names if c.lower() in gemini_answer.lower()), None)
        return {"gemini_says": gemini_answer, "matched_class": matched}
    except Exception as e:
        print(f"Gemini recheck error: {e}")
        return {"gemini_says": None, "matched_class": None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None or not class_names:
        raise HTTPException(status_code=503, detail="Model belum siap, coba lagi.")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")
    ...

    image_bytes = await file.read()

    # Step 1: Gemini gatekeeper
    if gemini_client is not None:
        is_leaf, gemini_raw = validate_with_gemini(image_bytes)
        if not is_leaf:
            return {
                "prediction": "Not a rice leaf",
                "confidence": 0.0,
                "all_scores": [],
                "message": "Gambar tidak dikenali sebagai daun padi. Pastikan foto menampilkan daun padi secara jelas.",
                "validated_by": "gemini"
            }

    # Step 2: Model predict
    img_array = preprocess_image(image_bytes)
    preds = model.predict(img_array)[0]
    top_idx = int(np.argmax(preds))
    top_conf = float(preds[top_idx])

    results = [
        {"class": class_names[i], "confidence": round(float(preds[i]) * 100, 2)}
        for i in range(len(class_names))
    ]
    results.sort(key=lambda x: x["confidence"], reverse=True)

    # Step 3: Terlalu rendah → tolak
    if top_conf < 0.5:
        return {
            "prediction": "Not a rice leaf",
            "confidence": round(top_conf * 100, 2),
            "all_scores": results,
            "message": "Gambar tidak dikenali sebagai daun padi. Pastikan foto menampilkan daun padi secara jelas.",
            "validated_by": "threshold"
        }

    # Step 4: Gemini recheck kalau confidence 0.5-0.75
    final_prediction = class_names[top_idx]
    validated_by = "model"

    if gemini_client is not None and top_conf < 0.75:
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