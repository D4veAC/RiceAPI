import json
import os
import asyncio
from contextlib import asynccontextmanager
from functools import partial
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from google import genai
from google.genai import types

model = None
class_names = []
gemini_client = None


def load_model():
    global model, class_names, gemini_client

    model = tf.keras.models.load_model("rice_model.keras", safe_mode=False)
    
    with open("class_names.json") as f:
        loaded = json.load(f)
    class_names.extend(loaded)  # ← pakai extend bukan assignment

    output_units = model.output_shape[-1]
    if len(class_names) < output_units:
        missing = output_units - len(class_names)
        print(
            f"WARNING: class_names.json has {len(class_names)} labels but model output has {output_units} units. "
            f"Adding {missing} placeholder label(s)."
        )
        class_names.extend([f"Unknown Label {i+1}" for i in range(missing)])
    elif len(class_names) > output_units:
        print(
            f"WARNING: class_names.json has {len(class_names)} labels but model output has {output_units} units. "
            f"Truncating extra label(s)."
        )
        del class_names[output_units:]

    print(f"Model loaded. Classes: {class_names}")

    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        gemini_client = genai.Client(api_key=api_key)
        print("Gemini 2.5 Flash loaded.")
    else:
        print("WARNING: GEMINI_API_KEY not set. Running without Gemini validation.")


# Use lifespan context manager instead of deprecated @app.on_event("startup")
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    # Shutdown cleanup (if needed in the future)


app = FastAPI(title="Rice Disease Classifier", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def analyze_with_gemini(image_bytes: bytes) -> dict | None:
    prompt = """
        Analyze this image of a paddy plant. 
        Identify if it has one of these 6 conditions: 
        1. Blast
        2. HDB (Bacterial Leaf Blight)
        3. Tungro
        4. Brown Planthopper (Wereng Cokelat)
        5. Golden Apple Snail (Keong Mas)
        6. Nitrogen Deficiency
        
        Or if it looks Healthy.
        
        Return a JSON object with:
        - condition: string (one of the above or "Healthy" or "Unknown")
        - confidence: number (0-100)
        - treatment: string (specific advice in Indonesian)
        - description: string (brief description of what is seen)
    """
    try:
        img_bytes = resize_for_gemini(image_bytes)
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            ],
            config={"responseMimeType": "application/json"}
        )
        response_text = response.text
        if not response_text:
            raise ValueError("Empty Gemini response")
        result = json.loads(response_text)
        if not isinstance(result, dict):
            raise ValueError("Gemini returned non-JSON object")
        return result
    except Exception as e:
        print(f"Gemini analysis error: {e}")
        return None




@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None or not class_names:
        raise HTTPException(status_code=503, detail="Model belum siap, coba lagi.")
    
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar (image/*)")

    image_bytes = await file.read()

    if gemini_client is not None:
        gemini_result = analyze_with_gemini(image_bytes)
        if gemini_result is not None:
            gemini_result["source"] = "gemini"
            return gemini_result
        print("Gemini analysis failed or returned invalid output, falling back to RiceAPI model.")

    # Step 2: Model predict
    img_array = preprocess_image(image_bytes)
    # Run CPU-intensive prediction in a thread to avoid blocking the async event loop
    loop = asyncio.get_event_loop()
    preds = await loop.run_in_executor(None, partial(model.predict, img_array))
    preds = preds[0]
    top_idx = int(np.argmax(preds))
    top_conf = float(preds[top_idx])

    label_count = min(len(class_names), len(preds))
    results = [
        {"class": class_names[i], "confidence": round(float(preds[i]) * 100, 2)}
        for i in range(label_count)
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

    final_prediction = class_names[top_idx]
    validated_by = "riceapi"

    return {
        "prediction": final_prediction,
        "confidence": round(top_conf * 100, 2),
        "all_scores": results,
        "validated_by": validated_by
    }
