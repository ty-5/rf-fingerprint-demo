#!/usr/bin/env python3
"""
RF Classifier Demo - FastAPI Backend Server
===========================================

This FastAPI server provides a REST API for the RF signal classification demo.
It loads our trained PyTorch CNN model and serves predictions via HTTP endpoints.

Key Features:
- Loads trained CNNFingerprinter model (98%+ accuracy on ORACLE dataset)
- Processes I/Q signal data for classification
- Provides layer-by-layer outputs for visualization
- Supports file uploads and real-time classification
- Auto-generates API documentation

Architecture:
Frontend (React) ←→ FastAPI Server ←→ PyTorch Model
(UI)                (This file)       (CNN_Extended.py)

Purpose: RF Fingerprinting Demo for Executive Presentation
Dataset: ORACLE RF Dataset (16 bit-identical USRP X310 transmitters)
"""

# =============================================================================
# IMPORTS AND SETUP
# =============================================================================

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import sys
import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import time

# Add parent directory to Python path so we can import our custom modules
# This allows us to import CNN_Extended.py and configs.py from the root directory
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

# Import our custom modules
from CNN_Extended import CNNFingerprinter  # CNN model architecture
import configs  # Configuration file with model parameters

# =============================================================================
# FASTAPI APPLICATION SETUP
# =============================================================================

# Create the FastAPI application instance
app = FastAPI(
    title="RF Classifier API",
    description="API for classifying RF signals using trained CNN model",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI at /docs
    redoc_url="/redoc"  # Alternative docs at /redoc
)

# Enable Cross-Origin Resource Sharing (CORS)
# This allows our React frontend to communicate with this backend server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server (React)
        "http://localhost:3000",  # Create React App dev server
        "http://localhost:5174"   # Alternative Vite port
    ],
    allow_credentials=True,
    allow_methods=["*"],        # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],        # Allow all headers
)

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

# Global variables to store the loaded model and device info
# These are set once when the server starts up
model = None          # Will hold our trained CNNFingerprinter model
device = None         # Will be "cuda" or "cpu" depending on availability

# =============================================================================
# PYDANTIC MODELS (REQUEST/RESPONSE SCHEMAS)
# =============================================================================

class SignalData(BaseModel):
    """
    Schema for signal classification requests
    
    Expected format:
    {
        "signal": [[I_channel_data], [Q_channel_data]],
        "get_layers": true,  # Optional: return intermediate layer outputs
        "metadata": {"distance": "32ft", "true_label": 7}  # Optional
    }
    """
    signal: List[List[float]]           # I/Q signal data [2 channels x time_samples]
    get_layers: Optional[bool] = False  # Whether to return layer-by-layer outputs
    metadata: Optional[Dict[str, Any]] = {}  # Additional info (distance, true label, etc.)

class ClassificationResponse(BaseModel):
    """
    Schema for classification results
    
    Returns:
    - Predicted transmitter (0-15)
    - Confidence score (0-1)
    - All 16 transmitter probabilities
    - Processing time
    - Optional: layer outputs for visualization
    """
    status: str                                      # "success" or "error"
    predicted_transmitter: int                       # Predicted class (0-15)
    confidence: float                               # Confidence of prediction (0-1)
    all_probabilities: List[float]                  # Probabilities for all 16 transmitters
    processing_time_ms: float                       # How long inference took
    layer_outputs: Optional[List[Dict[str, Any]]] = None  # Layer-by-layer outputs
    metadata: Optional[Dict[str, Any]] = {}         # Echo back metadata

class LayerStreamRequest(BaseModel):
    """
    Schema for layer-by-layer streaming (for animation)
    
    Used to process signals progressively through CNN layers
    for the animated demo visualization
    """
    signal: List[List[float]]  # I/Q signal data
    layer_index: int          # Which layer to process up to (0-5)
    
class LayerStreamResponse(BaseModel):
    """
    Schema for layer streaming results
    
    Returns real intermediate results for honest demo animation
    """
    layer_name: str                    # Name of current layer
    layer_index: int                  # Current layer index
    feature_maps: List[List[float]]   # Real feature map data from CNN
    current_predictions: List[float]  # Only populated at final layer
    is_final: bool                   # Whether this is the final layer
    layer_info: Optional[Dict[str, Any]] = {}  # Description of what we're seeing

# =============================================================================
# MODEL LOADING (STARTUP EVENT)
# =============================================================================

@app.on_event("startup")
async def load_model():
    """
    Load the trained PyTorch model when the server starts
    
    This function runs once when the FastAPI server starts up.
    It loads our trained CNN model and prepares it for inference.
    
    Process:
    1. Determine if CUDA (GPU) is available
    2. Create model architecture using CNNFingerprinter class
    3. Load trained weights from RF_Model_Weights_98%.pth
    4. Set model to evaluation mode (disable training features)
    """
    global model, device
    
    print("RF Classifier API Server Starting...")
    print("Loading trained CNN model...")
    
    # Determine computation device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Create model architecture
        # Parameters match our training configuration:
        # - 16 transmitters (ORACLE dataset)
        # - 2 input channels (I/Q)
        # - Window size from configs.py (128 samples)
        print("Creating CNN architecture...")
        model = CNNFingerprinter(
            num_transmitters=16,
            input_channels=2,
            input_window=configs.window_size
        ).to(device)
        
        # Load trained weights
        weights_path = ROOT_DIR / "RF_Model_Weights_98%.pth"
        print(f"Loading weights from: {weights_path}")
        
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        
        # Load state dictionary (trained parameters)
        state_dict = torch.load(str(weights_path), map_location=device)
        model.load_state_dict(state_dict)
        
        # Set to evaluation mode
        # This disables dropout, batch norm training mode, etc.
        model.eval()
        
        print("Model loaded successfully!")
        print(f"Input shape: [batch_size, 2, {configs.window_size}]")
        print(f"Output shape: [batch_size, 16]")
        print(f"Ready to classify RF signals!")
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Make sure RF_Model_Weights_98%.pth is in the root directory")
        raise

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def preprocess_signal(signal_data: List[List[float]]) -> torch.Tensor:
    """
    Convert raw signal data to PyTorch tensor format
    
    Args:
        signal_data: List of 2 lists [I_channel, Q_channel]
        
    Returns:
        torch.Tensor: Preprocessed signal ready for model inference
        
    Process:
    1. Convert to numpy array
    2. Validate shape (must be 2 channels)
    3. Pad or truncate to correct window size
    4. Convert to PyTorch tensor
    5. Add batch dimension for model
    """
    try:
        # Convert to numpy array with correct data type
        signal_array = np.array(signal_data, dtype=np.float64)
        
        # Validate input shape
        if signal_array.shape[0] != 2:
            raise ValueError(f"Expected 2 channels (I/Q), got {signal_array.shape[0]}")
        
        # Ensure correct window size
        if signal_array.shape[1] != configs.window_size:
            if signal_array.shape[1] < configs.window_size:
                # Pad with zeros if signal is too short
                padding = configs.window_size - signal_array.shape[1]
                signal_array = np.pad(signal_array, ((0, 0), (0, padding)), mode='constant')
            else:
                # Truncate if signal is too long
                signal_array = signal_array[:, :configs.window_size]
        
        # Convert to PyTorch tensor
        signal_tensor = torch.tensor(signal_array, dtype=configs.datatype, device=device)
        
        # Add batch dimension: [channels, time] → [batch=1, channels, time]
        signal_tensor = signal_tensor.unsqueeze(0)
        
        return signal_tensor
        
    except Exception as e:
        raise ValueError(f"Signal preprocessing failed: {str(e)}")

def extract_layer_outputs(signal_tensor: torch.Tensor) -> List[Dict[str, Any]]:
    """
    Extract intermediate outputs from each layer for visualization
    
    This function processes a signal through the CNN and captures
    the output from each layer. Used for the animated demo to show
    how the signal transforms through the network.
    
    Args:
        signal_tensor: Preprocessed input signal
        
    Returns:
        List of dictionaries containing layer information
    """
    layer_outputs = []
    
    with torch.no_grad():  # Disable gradient computation for faster inference
        x = signal_tensor
        
        # Capture input
        layer_outputs.append({
            "name": "Input",
            "shape": list(x.shape),
            "sample_values": x[0, :, :8].cpu().tolist()  # First 8 time samples for viz
        })
        
        # Process through each convolutional block
        # Our CNN has 4 blocks, each with 2 conv layers + pooling
        for block_idx, block in enumerate(model.ConvolutionalLayers):
            x = block(x)
            layer_outputs.append({
                "name": f"ConvBlock{block_idx + 1}_Output",
                "shape": list(x.shape),
                # Sample feature maps for visualization (first 8 channels, 8 time samples)
                "feature_maps": x[0, :8, :8].cpu().tolist() if x.dim() > 2 else x[0, :8].cpu().tolist(),
                "num_features": x.shape[1] if x.dim() > 2 else x.shape[-1]
            })
        
        # Flatten for classifier
        x_flat = x.view(x.size(0), -1)
        layer_outputs.append({
            "name": "Flattened",
            "shape": list(x_flat.shape),
            "sample_values": x_flat[0, :16].cpu().tolist()  # First 16 flattened values
        })
        
        # Final classifier output
        final_output = model.classifer(x_flat)
        layer_outputs.append({
            "name": "Classifier_Output",
            "shape": list(final_output.shape),
            "logits": final_output[0].cpu().tolist(),
            "probabilities": torch.softmax(final_output, dim=1)[0].cpu().tolist()
        })
    
    return layer_outputs

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """
    Basic health check endpoint
    
    Returns server status and basic information.
    Useful for verifying the server is running.
    """
    return {
        "message": "RF Classifier API is running!",
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown",
        "version": "1.0.0"
    }

@app.get("/model/info")
async def model_info():
    """
    Get detailed information about the loaded model
    
    Returns:
    - Model architecture details
    - Training dataset info
    - Input/output specifications
    - Performance metrics
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "CNNFingerprinter", 
        "accuracy": "98%+ distance-invariant",
        "dataset": "ORACLE RF Dataset",
        "num_transmitters": 16,
        "input_shape": [2, configs.window_size],
        "device": str(device),
        "data_type": str(configs.datatype),
        "layers": [
            "Input",
            "ConvBlock1_Conv1", "ConvBlock1_Conv2",
            "ConvBlock2_Conv1", "ConvBlock2_Conv2",
            "ConvBlock3_Conv1", "ConvBlock3_Conv2", 
            "ConvBlock4_Conv1", "ConvBlock4_Conv2",
            "Classifier"
        ],
        "distances_tested": "2ft to 62ft (11 distances)",
        "transmitter_hardware": "Bit-identical USRP X310"
    }

@app.post("/classify", response_model=ClassificationResponse)
async def classify_signal(signal_request: SignalData):
    """
    Main classification endpoint
    
    Classifies an RF signal and returns prediction results.
    This is the primary endpoint for signal classification.
    
    Args:
        signal_request: SignalData containing I/Q signal and options
        
    Returns:
        ClassificationResponse with prediction results
        
    Process:
    1. Preprocess the input signal
    2. Run inference through the CNN
    3. Convert logits to probabilities
    4. Extract prediction and confidence
    5. Optionally extract layer outputs for visualization
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        
        # Step 1: Preprocess the signal
        signal_tensor = preprocess_signal(signal_request.signal)
        
        # Step 2: Run inference
        with torch.no_grad():
            # Forward pass through the CNN
            predictions = model(signal_tensor)
            # Convert logits to probabilities using softmax
            probabilities = torch.softmax(predictions, dim=1)
        
        # Step 3: Extract results
        predicted_class = int(torch.argmax(probabilities, dim=1)[0])
        confidence = float(torch.max(probabilities, dim=1)[0])
        all_probs = probabilities[0].cpu().tolist()
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Step 4: Extract layer outputs if requested (for visualization)
        layer_outputs = None
        if signal_request.get_layers:
            layer_outputs = extract_layer_outputs(signal_tensor)
        
        return ClassificationResponse(
            status="success",
            predicted_transmitter=predicted_class,
            confidence=confidence,
            all_probabilities=all_probs,
            processing_time_ms=processing_time,
            layer_outputs=layer_outputs,
            metadata=signal_request.metadata
        )
        
    except ValueError as e:
        # Client error (bad input format)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Server error (something went wrong during processing)
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/classify/stream", response_model=LayerStreamResponse)
async def classify_layer_stream(request: LayerStreamRequest):
    """
    HONEST layer-by-layer processing for demo
    
    This endpoint processes a signal progressively through CNN layers,
    showing the REAL feature map evolution without fake confidence building.
    
    Used for the demo animation that shows:
    - How the signal actually transforms through each layer
    - Real feature maps detected by the CNN
    - Honest representation of CNN processing
    - Final classification only appears at the end (where it actually happens)
    
    Args:
        request: LayerStreamRequest with signal and target layer
        
    Returns:
        LayerStreamResponse with real layer outputs and feature maps
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        signal_tensor = preprocess_signal(request.signal)
        layer_names = ["Input", "ConvBlock1", "ConvBlock2", "ConvBlock3", "ConvBlock4", "Classifier"]
        
        if request.layer_index >= len(layer_names):
            raise HTTPException(status_code=400, detail="Layer index out of range")
        
        with torch.no_grad():
            x = signal_tensor
            
            # Process up to the requested layer
            if request.layer_index >= 1:  # ConvBlock1
                x = model.ConvolutionalLayers[0](x)
            if request.layer_index >= 2:  # ConvBlock2
                x = model.ConvolutionalLayers[1](x)
            if request.layer_index >= 3:  # ConvBlock3
                x = model.ConvolutionalLayers[2](x)
            if request.layer_index >= 4:  # ConvBlock4
                x = model.ConvolutionalLayers[3](x)
            
            # Only show predictions at final layer -- intermediate confidence estimates are fallacious
            current_predictions = []
            if request.layer_index >= 5:  # Final classifier layer only
                x_flat = x.view(x.size(0), -1)
                final_predictions = model.classifer(x_flat)
                probabilities = torch.softmax(final_predictions, dim=1)
                current_predictions = probabilities[0].cpu().tolist()
                
                # Also return detailed prediction info for final layer
                predicted_class = int(torch.argmax(probabilities, dim=1)[0])
                confidence = float(torch.max(probabilities, dim=1)[0])
            # For layers 0-4: No predictions available yet -- showing feature maps instead
            
            # Extract REAL feature maps for visualization
            feature_maps = []
            layer_info = {}
            
            if request.layer_index == 0:  # Input layer
                # Show the raw I/Q signal
                feature_maps = x[0].cpu().tolist()  # [I_channel, Q_channel]
                layer_info = {
                    "description": "Raw I/Q signal input",
                    "channels": 2,
                    "samples": x.shape[2],
                    "data_type": "Raw RF signal"
                }
                
            elif x.dim() == 3:  # Convolutional layers [batch, channels, time]
                # Show actual feature maps (what the CNN is really detecting)
                num_display_channels = min(8, x.shape[1])  # Show up to 8 feature maps
                feature_maps = x[0, :num_display_channels].cpu().tolist()
                
                layer_info = {
                    "description": f"Feature maps from {layer_names[request.layer_index]}",
                    "total_channels": x.shape[1],
                    "displayed_channels": num_display_channels,
                    "time_samples": x.shape[2],
                    "data_type": "Learned feature activations"
                }
                
            else:  # Flattened layer
                # Show flattened features
                feature_maps = [x[0, :32].cpu().tolist()]  # First 32 flattened values
                layer_info = {
                    "description": "Flattened features for classification",
                    "total_features": x.shape[1],
                    "displayed_features": min(32, x.shape[1]),
                    "data_type": "Preprocessed features for final classification"
                }
        
        return LayerStreamResponse(
            layer_name=layer_names[request.layer_index],
            layer_index=request.layer_index,
            feature_maps=feature_maps,
            current_predictions=current_predictions,  # Empty until final layer
            is_final=request.layer_index >= len(layer_names) - 1,
            layer_info=layer_info  # Additional info about what we're seeing
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Layer processing failed: {str(e)}")

@app.post("/classify/upload")
async def classify_uploaded_file(file: UploadFile = File(...)):
    """
    Handle file uploads for classification
    
    Allows users to upload signal files (JSON format) for classification.
    Useful for processing ORACLE dataset files or saved signals.
    
    Expected file format:
    {
        "signal": [[I_channel], [Q_channel]],
        "metadata": {"distance": "32ft", "true_label": 7}
    }
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read uploaded file content
        content = await file.read()
        
        # Handle different file types
        if file.filename.endswith('.json'):
            # Parse JSON signal data
            signal_data = json.loads(content.decode('utf-8'))
            signal_request = SignalData(**signal_data)
            return await classify_signal(signal_request)
            
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use JSON format.")
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

@app.post("/classify/demo")
async def classify_for_demo(signal_request: SignalData):
    """
    Two-phase demo classification for honest animation
    
    This endpoint provides a complete classification result that the frontend
    can use for honest animation. The process:
    
    1. Immediately classify the signal (get real result)
    2. Return the real result + metadata for animation
    3. Frontend can animate revealing this REAL result layer by layer
    
    This approach ensures:
    - No fake confidence building
    - Real CNN feature map evolution
    - Honest blind classification
    - Trustworthy results
    
    Args:
        signal_request: SignalData with signal to classify
        
    Returns:
        Complete classification result + animation metadata
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get the REAL classification immediately
        real_result = await classify_signal(signal_request)
        
        # Return real result + metadata for honest animation
        return {
            "status": "success",
            "real_classification": {
                "predicted_transmitter": real_result.predicted_transmitter,
                "confidence": real_result.confidence,
                "all_probabilities": real_result.all_probabilities,
                "processing_time_ms": real_result.processing_time_ms
            },
            "animation_metadata": {
                "total_layers": 6,
                "layer_names": ["Input", "ConvBlock1", "ConvBlock2", "ConvBlock3", "ConvBlock4", "Classifier"],
                "message": "Use /classify/stream to animate through layers, then reveal this real result",
                "demo_approach": "honest_cnn_processing"
            },
            "metadata": signal_request.metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demo classification failed: {str(e)}")

@app.get("/test")
async def test_classification():
    """
    Simple test endpoint with random data
    
    Generates a random signal and classifies it to verify the model is working.
    Useful for quick health checks and debugging.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Generate random test signal matching our input format
        test_signal = torch.randn(1, 2, configs.window_size, dtype=configs.datatype, device=device)
        
        # Run inference
        with torch.no_grad():
            predictions = model(test_signal)
            probabilities = torch.softmax(predictions, dim=1)
        
        # Extract results
        predicted_class = int(torch.argmax(probabilities, dim=1)[0])
        confidence = float(torch.max(probabilities, dim=1)[0])
        
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
    """
    Start the FastAPI server when this file is run directly
    
    This allows you to start the server with: python main.py
    """
    import uvicorn
    
    print("\n" + "="*60)
    print("RF CLASSIFIER DEMO - HONEST CNN VISUALIZATION")
    print("="*60)
    print("Starting FastAPI server...")
    print("Server: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("Test Endpoint: http://localhost:8000/test")
    print("Model Info: http://localhost:8000/model/info")
    print("")
    print("="*60)
    
    try:
        # Start the server with auto-reload for development
        uvicorn.run(
            "main:app",           # Module:app_instance
            host="0.0.0.0",      # Listen on all interfaces
            port=8000,           # Port number
            reload=True          # Auto-reload on code changes
        )
    except Exception as e:
        print(f"Server failed to start: {e}")
        input("Press Enter to exit...")

# =============================================================================
# HONEST DEMO ARCHITECTURE NOTES
# =============================================================================

"""
FRONTEND INTEGRATION GUIDE - HONEST CNN DEMO
============================================

For an honest, trustworthy demo that shows real CNN processing:

1. DEMO FLOW:
   • User uploads/selects signal file
   • Call /classify/demo to get REAL result (hidden from user)
   • Animate through layers 0-5 using /classify/stream
   • Show real feature map evolution
   • Reveal actual classification at the end

2. LAYER-BY-LAYER ANIMATION:
   • Layer 0: Show raw I/Q signal
   • Layers 1-4: Show evolving feature maps (real CNN processing)
   • Layer 5: Reveal real classification result

3. FRONTEND EXAMPLE:
   ```javascript
   // Step 1: Get real result (hidden)
   const realResult = await fetch('/classify/demo', {
     method: 'POST',
     body: JSON.stringify({signal: signalData})
   });
   
   // Step 2: Animate through layers
   for (let layer = 0; layer <= 5; layer++) {
     const layerResult = await fetch('/classify/stream', {
       method: 'POST', 
       body: JSON.stringify({signal: signalData, layer_index: layer})
     });
     
     // Show real feature maps
     displayFeatureMaps(layerResult.feature_maps);
     
     if (layer === 5) {
       // Reveal the real result
       displayPrediction(realResult.real_classification);
     }
   }
   ```
"""