#!/usr/bin/env python3
"""
Lightweight test using the existing rf_signals.json file
This tests a few samples to verify model predictions work correctly
"""

import torch
import json
import requests
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

from CNN_Extended import CNNFingerprinter
import configs

def test_model_directly():
    """Test the model directly with a few samples from rf_signals.json"""
    
    print("="*60)
    print("LIGHTWEIGHT MODEL TEST")
    print("="*60)
    
    # Load the JSON signals
    try:
        with open("rf_signals.json", "r") as f:
            signals_data = json.load(f)
        print(f"✓ Loaded rf_signals.json with {len(signals_data)} transmitters")
    except Exception as e:
        print(f"❌ Failed to load rf_signals.json: {e}")
        return False
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNNFingerprinter(
        num_transmitters=16,
        input_channels=2,
        input_window=configs.window_size
    ).to(device)
    
    try:
        state_dict = torch.load("RF_Model_Weights_98%.pth", map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"✓ Model loaded on {device}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Test one sample from each transmitter (first 5 transmitters)
    print(f"\n🧪 Testing model directly with JSON samples...")
    print("-" * 60)
    
    predictions_vary = False
    
    for transmitter_idx in range(min(5, len(signals_data))):
        if len(signals_data[transmitter_idx]) == 0:
            continue
            
        # Get first sample from this transmitter
        sample = signals_data[transmitter_idx][0]
        
        # Convert to tensor (same as main.py preprocessing)
        signal_array = np.array(sample, dtype=np.float64)
        signal_tensor = torch.tensor(signal_array, dtype=configs.datatype, device=device)
        signal_tensor = signal_tensor.unsqueeze(0)  # Add batch dimension
        
        # Apply same normalization as main.py
        IQ_mean = signal_tensor.mean()
        IQ_std = signal_tensor.std()
        signal_tensor = (signal_tensor - IQ_mean) / (IQ_std + 1e-8)
        
        # Predict
        with torch.no_grad():
            predictions = model(signal_tensor)
            probabilities = torch.softmax(predictions, dim=1)
            predicted_class = int(torch.argmax(probabilities, dim=1)[0])
            confidence = float(torch.max(probabilities, dim=1)[0])
        
        # Check if predictions vary
        if predicted_class != 2:
            predictions_vary = True
        
        print(f"Transmitter {transmitter_idx}: Predicted={predicted_class}, Confidence={confidence:.1%}")
    
    if not predictions_vary:
        print(f"\n⚠️  ALL predictions are transmitter 2 - there's still an issue!")
        return False
    else:
        print(f"\n✅ Predictions vary - model working correctly!")
        return True

def test_demo_endpoint():
    """Test the demo endpoint with a few samples"""
    
    print(f"\n" + "="*60)
    print("TESTING DEMO ENDPOINT")
    print("="*60)
    
    # Load signals
    try:
        with open("rf_signals.json", "r") as f:
            signals_data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load rf_signals.json: {e}")
        return False
    
    predictions_vary = False
    
    print("🌐 Testing FastAPI endpoint...")
    print("-" * 60)
    
    for transmitter_idx in range(min(3, len(signals_data))):
        if len(signals_data[transmitter_idx]) == 0:
            continue
            
        sample = signals_data[transmitter_idx][0]
        
        # Test the endpoint
        demo_data = {
            "signal": sample,
            "metadata": {
                "true_transmitter": transmitter_idx,
                "test_type": "json_verification"
            }
        }
        
        try:
            response = requests.post(
                "http://localhost:8000/classify",
                json=demo_data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                predicted = result['predicted_transmitter']
                confidence = result['confidence']
                
                if predicted != 2:
                    predictions_vary = True
                
                print(f"Transmitter {transmitter_idx}: API Predicted={predicted}, Confidence={confidence:.1%}")
            else:
                print(f"❌ API Error {response.status_code}: {response.text[:100]}")
                
        except Exception as e:
            print(f"❌ API Request failed: {e}")
    
    if not predictions_vary:
        print(f"\n⚠️  API still predicting transmitter 2 for everything!")
        return False
    else:
        print(f"\n✅ API predictions vary - working correctly!")
        return True

def inspect_json_format():
    """Inspect the JSON format to understand the data structure"""
    
    print(f"\n" + "="*60)
    print("INSPECTING JSON FORMAT")
    print("="*60)
    
    try:
        with open("rf_signals.json", "r") as f:
            signals_data = json.load(f)
        
        print(f"📊 JSON Structure:")
        print(f"   Type: {type(signals_data)}")
        print(f"   Length: {len(signals_data)} (should be 16 transmitters)")
        
        if len(signals_data) > 0:
            first_transmitter = signals_data[0]
            print(f"   Samples per transmitter: {len(first_transmitter)}")
            
            if len(first_transmitter) > 0:
                first_sample = first_transmitter[0]
                print(f"   Sample type: {type(first_sample)}")
                print(f"   Sample shape: {len(first_sample)} x {len(first_sample[0]) if len(first_sample) > 0 else 0}")
                print(f"   Expected: 2 x {configs.window_size}")
                
                if len(first_sample) >= 2:
                    print(f"   I channel length: {len(first_sample[0])}")
                    print(f"   Q channel length: {len(first_sample[1])}")
                    print(f"   I sample values: {first_sample[0][:5]}")
                    print(f"   Q sample values: {first_sample[1][:5]}")
                    
                    # Check data ranges
                    i_values = first_sample[0]
                    q_values = first_sample[1]
                    print(f"   I range: {min(i_values):.3f} to {max(i_values):.3f}")
                    print(f"   Q range: {min(q_values):.3f} to {max(q_values):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to inspect JSON: {e}")
        return False

def debug_single_prediction():
    """Debug a single prediction step by step"""
    
    print(f"\n" + "="*60)
    print("DEBUGGING SINGLE PREDICTION")
    print("="*60)
    
    try:
        with open("rf_signals.json", "r") as f:
            signals_data = json.load(f)
        
        # Get one sample
        sample = signals_data[0][0]  # First sample from first transmitter
        
        print(f"🔍 Raw sample shape: {len(sample)} x {len(sample[0])}")
        
        # Step 1: Convert to numpy
        signal_array = np.array(sample, dtype=np.float64)
        print(f"🔍 Numpy array shape: {signal_array.shape}")
        print(f"🔍 Numpy array dtype: {signal_array.dtype}")
        
        # Step 2: Convert to tensor
        signal_tensor = torch.tensor(signal_array, dtype=configs.datatype)
        signal_tensor = signal_tensor.unsqueeze(0)  # Add batch dimension
        print(f"🔍 Tensor shape: {signal_tensor.shape}")
        print(f"🔍 Tensor dtype: {signal_tensor.dtype}")
        
        # Step 3: Normalization
        IQ_mean = signal_tensor.mean()
        IQ_std = signal_tensor.std()
        signal_tensor_norm = (signal_tensor - IQ_mean) / (IQ_std + 1e-8)
        
        print(f"🔍 Before normalization - mean: {IQ_mean:.6f}, std: {IQ_std:.6f}")
        print(f"🔍 After normalization - mean: {signal_tensor_norm.mean():.6f}, std: {signal_tensor_norm.std():.6f}")
        
        # Step 4: Load model and predict
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CNNFingerprinter(
            num_transmitters=16,
            input_channels=2,
            input_window=configs.window_size
        ).to(device)
        
        state_dict = torch.load("RF_Model_Weights_98%.pth", map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        
        signal_tensor_norm = signal_tensor_norm.to(device)
        
        with torch.no_grad():
            predictions = model(signal_tensor_norm)
            probabilities = torch.softmax(predictions, dim=1)
            
        print(f"🔍 Raw logits: {predictions[0].cpu().tolist()}")
        print(f"🔍 Probabilities: {probabilities[0].cpu().tolist()}")
        print(f"🔍 Predicted class: {torch.argmax(probabilities, dim=1)[0].item()}")
        print(f"🔍 Confidence: {torch.max(probabilities, dim=1)[0].item():.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 LIGHTWEIGHT RF MODEL TEST")
    print("Testing with existing rf_signals.json file")
    print("")
    
    # Run lightweight tests
    print("Step 1: Inspect JSON format")
    json_ok = inspect_json_format()
    
    print(f"\nStep 2: Debug single prediction")
    debug_ok = debug_single_prediction()
    
    print(f"\nStep 3: Test model directly")
    model_ok = test_model_directly()
    
    print(f"\nStep 4: Test demo endpoint")
    api_ok = test_demo_endpoint()
    
    # Summary
    print(f"\n" + "="*60)
    print("🎯 RESULTS SUMMARY")
    print("="*60)
    print(f"JSON format OK: {'✅' if json_ok else '❌'}")
    print(f"Debug successful: {'✅' if debug_ok else '❌'}")
    print(f"Model predictions vary: {'✅' if model_ok else '❌'}")
    print(f"API predictions vary: {'✅' if api_ok else '❌'}")
    
    if not model_ok:
        print(f"\n🚨 PROBLEM: Model always predicts transmitter 2")
        print(f"   - This suggests an issue with data preprocessing")
        print(f"   - Check the debug output above for clues")
    elif not api_ok:
        print(f"\n🚨 PROBLEM: API endpoint has issues")
        print(f"   - Model works directly but API doesn't")
        print(f"   - Check server logs and data format")
    else:
        print(f"\n🎉 SUCCESS: Everything working correctly!")