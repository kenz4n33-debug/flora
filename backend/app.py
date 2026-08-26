import io
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, status
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
    # Cleanup if needed
    interpreter = None


app = FastAPI(
    title="AI Vision TFLite Plant Disease Classifier",
    description="Backend API for testing TensorFlow Lite plant disease model inference.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS so React frontend can make API requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ai-vision-tau.vercel.app",
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    """Welcome endpoint."""
    return {"name": "AI_VISION API", "status": "online"}


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
async def predict(file: UploadFile = File(...)):
    """
    Predict plant leaf condition using AI_VISION.tflite.
    Preprocessing steps:
    1. Read uploaded image bytes.
    2. Convert to RGB image.
    3. Resize to 224x224 pixels.
    4. Convert to float32 NumPy array with shape (1, 224, 224, 3).
    5. Note: Rescaling layer is INSIDE the model, so NO division by 255.0 is applied.
    """
    if interpreter is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="TFLite interpreter is not loaded.",
        )

    # Validate file type extension/mime
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File provided is not a valid image. Please upload JPG, JPEG, or PNG.",
        )

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        image = Image.open(io.BytesIO(contents))
        image = image.convert("RGB")
        image = image.resize((224, 224))
    except HTTPException:
        raise
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

        # Ensure probabilities (apply softmax if raw logits are returned)
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
            "prediction": top_prediction,
            "confidence": confidence,
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
