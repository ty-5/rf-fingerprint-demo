#!/usr/bin/env python3
"""
RF Classifier Demo - FastAPI Backend Server (ONNX Version)
===========================================================

This is a lightweight version using ONNX Runtime instead of PyTorch.
Optimized for AWS Lambda deployment with smaller dependencies.

Key Changes from PyTorch version:
- Uses ONNX Runtime instead of PyTorch (much smaller)
- Same API endpoints and functionality
- Suitable for AWS Lambda deployment
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import sys
import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import time

# =============================================================================
# FASTAPI APPLICATION SETUP
# =============================================================================

app = FastAPI(
    title="RF Classifier API (ONNX)",
    description="API for classifying RF signals using ONNX model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:5174",
        "*"  # Allow all origins for deployment
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

ort_session = None  # ONNX Runtime inference session
window_size = 128   # Signal window size

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class SignalData(BaseModel):
    signal: List[List[float]]
    get_layers: Optional[bool] = False
    metadata: Optional[Dict[str, Any]] = {}

class ClassificationResponse(BaseModel):
    status: str
    predicted_transmitter: int
    confidence: float
    all_probabilities: List[float]
    processing_time_ms: float
    layer_outputs: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = {}

# =============================================================================
# MODEL LOADING
# =============================================================================

@app.on_event("startup")
async def load_model():
    """Load the ONNX model when the server starts"""
    global ort_session

    print("RF Classifier API Server Starting (ONNX Version)...")
    print("Loading ONNX model...")

    try:
        # Find the ONNX model file
        model_path = Path(__file__).parent / "models" / "rf_classifier_alt.onnx"

        print(f"Loading model from: {model_path}")
        print(f"File exists: {model_path.exists()}")

        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        print(f"File size: {model_path.stat().st_size / (1024*1024):.2f} MB")

        # Create ONNX Runtime inference session
        ort_session = ort.InferenceSession(str(model_path))

        # Get model info
        input_name = ort_session.get_inputs()[0].name
        input_shape = ort_session.get_inputs()[0].shape
        output_name = ort_session.get_outputs()[0].name

        print("Model loaded successfully!")
        print(f"Input name: {input_name}")
        print(f"Input shape: {input_shape}")
        print(f"Output name: {output_name}")
        print(f"Ready to classify RF signals!")

    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def preprocess_signal(signal_data: List[List[float]]) -> np.ndarray:
    """Convert raw signal data to numpy array format for ONNX"""
    try:
        print(f"Preprocessing signal_data length: {len(signal_data) if signal_data else 'None'}")

        # Convert to numpy array with correct data type
        signal_array = np.array(signal_data, dtype=np.float64)
        print(f"Converted to numpy shape: {signal_array.shape}")

        # Ensure correct shape: [channels, time_samples]
        if signal_array.shape[0] != 2:
            raise ValueError(f"Expected 2 channels (I/Q), got {signal_array.shape[0]}")

        if signal_array.shape[1] != window_size:
            # Pad or truncate to correct size
            if signal_array.shape[1] < window_size:
                padding = window_size - signal_array.shape[1]
                signal_array = np.pad(signal_array, ((0, 0), (0, padding)), mode='constant')
            else:
                signal_array = signal_array[:, :window_size]

        # Add batch dimension: [1, channels, time_samples]
        signal_array = np.expand_dims(signal_array, axis=0)

        # Normalize (same as training)
        mean = signal_array.mean()
        std = signal_array.std()
        signal_array = (signal_array - mean) / (std + 1e-8)

        print(f"After normalization - mean: {signal_array.mean():.6f}, std: {signal_array.std():.6f}")
        print(f"Final shape for ONNX: {signal_array.shape}")

        return signal_array.astype(np.float32)  # ONNX typically uses float32

    except Exception as e:
        print(f"Preprocessing error: {e}")
        raise ValueError(f"Signal preprocessing failed: {str(e)}")

def softmax(x):
    """Compute softmax values for array x"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Basic health check endpoint"""
    return {
        "message": "RF Classifier API is running! (ONNX Version)",
        "status": "healthy",
        "model_loaded": ort_session is not None,
        "version": "1.0.0-onnx"
    }

@app.get("/model/info")
async def model_info():
    """Get detailed information about the loaded model"""
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    input_info = ort_session.get_inputs()[0]
    output_info = ort_session.get_outputs()[0]

    return {
        "model_type": "ONNX CNNFingerprinter",
        "runtime": "ONNX Runtime",
        "accuracy": "98%+ distance-invariant",
        "dataset": "ORACLE RF Dataset",
        "num_transmitters": 16,
        "input_name": input_info.name,
        "input_shape": input_info.shape,
        "output_name": output_info.name,
        "output_shape": output_info.shape,
        "distances_tested": "2ft to 62ft (11 distances)",
        "transmitter_hardware": "Bit-identical USRP X310"
    }

@app.post("/classify", response_model=ClassificationResponse)
async def classify_signal(signal_request: SignalData):
    """Main classification endpoint"""
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_time = time.time()

        print(f"Received signal request")
        print(f"Signal shape: {len(signal_request.signal)}x{len(signal_request.signal[0]) if signal_request.signal else 0}")

        # Step 1: Preprocess the signal
        signal_array = preprocess_signal(signal_request.signal)

        # Step 2: Run ONNX inference
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name

        predictions = ort_session.run([output_name], {input_name: signal_array})[0]

        print(f"Raw model outputs (logits): {predictions[0]}")
        print(f"Min/Max logit: {predictions.min():.3f}/{predictions.max():.3f}")

        # Step 3: Convert logits to probabilities
        probabilities = softmax(predictions)

        # Step 4: Extract results
        predicted_class = int(np.argmax(probabilities, axis=1)[0])
        confidence = float(np.max(probabilities, axis=1)[0])
        all_probs = probabilities[0].tolist()

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000

        print(f"Predicted: Transmitter {predicted_class} with {confidence:.2%} confidence")

        return ClassificationResponse(
            status="success",
            predicted_transmitter=predicted_class,
            confidence=confidence,
            all_probabilities=all_probs,
            processing_time_ms=processing_time,
            layer_outputs=None,  # Not implemented for ONNX version
            metadata=signal_request.metadata
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/classify/upload")
async def classify_uploaded_file(file: UploadFile = File(...)):
    """Handle file uploads for classification"""
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        content = await file.read()

        if file.filename.endswith('.json'):
            signal_data = json.loads(content.decode('utf-8'))
            signal_request = SignalData(**signal_data)
            return await classify_signal(signal_request)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use JSON format.")

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

@app.get("/test")
async def test_classification():
    """Simple test endpoint with random data"""
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Generate random test signal
        test_signal = np.random.randn(1, 2, window_size).astype(np.float32)

        # Run inference
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        predictions = ort_session.run([output_name], {input_name: test_signal})[0]
        probabilities = softmax(predictions)

        predicted_class = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))

        return {
            "status": "success",
            "predicted_transmitter": predicted_class,
            "confidence": confidence,
            "message": f"Test successful! Predicted transmitter {predicted_class} with {confidence:.1%} confidence"
        }

    except Exception as e:
        return {"error": f"Test failed: {str(e)}"}

# =============================================================================
# SERVER STARTUP
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("RF CLASSIFIER DEMO - ONNX VERSION")
    print("="*60)
    print("Starting FastAPI server...")
    print("Server: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("Test Endpoint: http://localhost:8000/test")
    print("="*60)

    try:
        uvicorn.run(
            "main_onnx:app",
            host="0.0.0.0",
            port=8000,
            reload=True
        )
    except Exception as e:
        print(f"Server failed to start: {e}")
        input("Press Enter to exit...")
