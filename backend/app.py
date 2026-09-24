import io
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image

# Import TFLite interpreter with fallbacks for maximum environment compatibility
try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite

# Class mapping order required by model specification:
# 0 = Healthy, 1 = Powdery, 2 = Rust
CLASS_NAMES = ["Healthy", "Powdery", "Rust"]

# Global references for TFLite interpreter and details
interpreter = None
input_details = None
output_details = None


def get_model_path():
    """Find the location of AI_VISION.tflite file."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "models", "AI_VISION.tflite"),
        os.path.join(os.path.dirname(__file__), "..", "models", "AI_VISION.tflite"),
        os.path.abspath("backend/models/AI_VISION.tflite"),
        os.path.abspath("models/AI_VISION.tflite"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Model file 'AI_VISION.tflite' not found in expected directories.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to load model once at startup."""
    global interpreter, input_details, output_details
    try:
        model_path = get_model_path()
        print(f"Loading TFLite model from: {model_path}")
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("TFLite Model loaded successfully!")
        print("Input details:", input_details)
        print("Output details:", output_details)
    except Exception as e:
        print(f"Error loading TFLite model: {e}")
        raise e
    yield
    interpreter = None


app = FastAPI(
    title="AI Vision TFLite Plant Disease Classifier",
    description="Backend API for TensorFlow Lite plant disease model inference.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS: allow all origins so Web Dashboard, Railway, and localhost can access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    """Welcome endpoint."""
    return {"name": "AI_VISION API", "status": "online", "model": "AI_VISION.tflite"}


@app.get("/health")
def health_check():
    """Health check endpoint confirming model status."""
    if interpreter is None or input_details is None or output_details is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TFLite model interpreter is not initialized.",
        )
    return {
        "status": "online",
        "model": "AI_VISION.tflite",
        "input_shape": input_details[0]["shape"].tolist(),
        "classes": CLASS_NAMES,
    }


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(None)):
    """
    Predict plant leaf condition using AI_VISION.tflite.
    Supports:
    1. Multipart file upload: `file: UploadFile`
    2. JSON payload: `{"image": "<base64>"}` or `{"image_base64": "<base64>"}`
    3. Raw binary image in request body (Content-Type: image/jpeg or application/octet-stream)
    """
    if interpreter is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="TFLite interpreter is not loaded.",
        )

    contents = None
    content_type = request.headers.get("content-type", "")

    # 1. Multipart file upload
    if file is not None:
        contents = await file.read()
    # 2. JSON Base64 upload
    elif "application/json" in content_type:
        try:
            body = await request.json()
            raw_b64 = body.get("image") or body.get("image_base64") or body.get("data")
            if raw_b64:
                import base64
                if "base64," in raw_b64:
                    raw_b64 = raw_b64.split("base64,")[1]
                contents = base64.b64decode(raw_b64)
        except Exception as ex:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON base64 payload: {str(ex)}",
            )
    # 3. Direct raw binary (e.g. sent directly from ESP32-CAM HTTPClient)
    else:
        body = await request.body()
        if body and len(body) > 0:
            contents = body

    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image data provided. Send a file, JSON with base64, or raw image bytes.",
        )

    try:
        image = Image.open(io.BytesIO(contents))
        image = image.convert("RGB")
        image = image.resize((224, 224))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupted or unsupported image file: {str(e)}",
        )

    try:
        # Preprocessing: convert to float32 without scaling (model contains internal Rescaling layer)
        img_array = np.array(image, dtype=np.float32)
        img_batch = np.expand_dims(img_array, axis=0)  # Shape (1, 224, 224, 3)

        # Set tensor & perform inference
        interpreter.set_tensor(input_details[0]["index"], img_batch)
        interpreter.invoke()

        # Retrieve output probabilities
        output_data = interpreter.get_tensor(output_details[0]["index"])[0]

        # Ensure probabilities
        if np.all(output_data >= 0) and np.isclose(np.sum(output_data), 1.0, atol=1e-2):
            probs = output_data
        else:
            exp_scores = np.exp(output_data - np.max(output_data))
            probs = exp_scores / np.sum(exp_scores)

        # Map to classes: 0 = Healthy, 1 = Powdery, 2 = Rust
        healthy_prob = round(float(probs[0]) * 100, 2)
        powdery_prob = round(float(probs[1]) * 100, 2)
        rust_prob = round(float(probs[2]) * 100, 2)

        # Find predicted class index
        top_idx = int(np.argmax(probs))
        top_prediction = CLASS_NAMES[top_idx]
        confidence = round(float(probs[top_idx]) * 100, 2)

        return {
            "status": "success",
            "prediction": top_prediction,
            "confidence": confidence,
            "healthy": healthy_prob,
            "powdery": powdery_prob,
            "rust": rust_prob,
            "probabilities": {
                "Healthy": healthy_prob,
                "Powdery": powdery_prob,
                "Rust": rust_prob,
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}",
        )
