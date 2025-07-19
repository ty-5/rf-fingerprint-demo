#!/usr/bin/env python3
"""
Fixed test that handles dictionary-format JSON (with string keys)
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

def test_model_with_dict_json():
    """Test the model with dictionary-format JSON"""
    
    print("="*60)
    print("TESTING MODEL WITH DICTIONARY JSON FORMAT")
    print("="*60)
    
    # Load the JSON signals (dictionary format)
    try:
        with open("rf_signals.json", "r") as f:
            signals_dict = json.load(f)
        print(f"✓ Loaded rf_signals.json")
        print(f"✓ Type: {type(signals_dict)}")
        print(f"✓ Keys: {list(signals_dict.keys())}")
        
        # Check first transmitter
        first_key = list(signals_dict.keys())[0]
        first_transmitter_data = signals_dict[first_key]
        print(f"✓ Transmitter '{first_key}' has {len(first_transmitter_data)} samples")
        
        if len(first_transmitter_data) > 0:
            sample_shape = np.array(first_transmitter_data[0]).shape
            print(f"✓ Sample shape: {sample_shape}")
        
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
    
    # Test samples from different transmitters
    print(f"\n🧪 Testing model with samples from dictionary JSON...")
    print("-" * 60)
    
    predictions_vary = False
    test_results = []
    
    # Test first 5 transmitters
    for key in sorted(signals_dict.keys())[:5]:
        transmitter_data = signals_dict[key]
        if len(transmitter_data) == 0:
            continue
            
        # Get first sample from this transmitter
        sample = transmitter_data[0]
        
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
            
            # Get top 3 predictions
            top_probs, top_indices = torch.topk(probabilities[0], 3)
            top_3 = [(int(idx), float(prob)) for idx, prob in zip(top_indices, top_probs)]
        
        # Check if predictions vary
        if predicted_class != 2:
            predictions_vary = True
        
        test_results.append({
            'transmitter_key': key,
            'predicted': predicted_class,
            'confidence': confidence,
            'top_3': top_3
        })
        
        print(f"Transmitter {key}: Predicted={predicted_class}, Confidence={confidence:.1%}")
        print(f"                Top 3: {[(f'T{idx}', f'{prob:.1%}') for idx, prob in top_3]}")
    
    print("-" * 60)
    
    if not predictions_vary:
        print(f"⚠️  ALL predictions are transmitter 2 - investigating further...")
        
        # Debug: Check if all samples look the same after preprocessing
        print(f"\n🔍 DEBUGGING: Checking if all samples look identical after preprocessing")
        
        sample_stats = []
        for key in sorted(signals_dict.keys())[:3]:
            sample = signals_dict[key][0]
            signal_array = np.array(sample, dtype=np.float64)
            signal_tensor = torch.tensor(signal_array, dtype=configs.datatype)
            signal_tensor = signal_tensor.unsqueeze(0)
            
            IQ_mean = signal_tensor.mean()
            IQ_std = signal_tensor.std()
            signal_tensor_norm = (signal_tensor - IQ_mean) / (IQ_std + 1e-8)
            
            stats = {
                'key': key,
                'raw_mean': float(signal_tensor.mean()),
                'raw_std': float(signal_tensor.std()),
                'raw_min': float(signal_tensor.min()),
                'raw_max': float(signal_tensor.max()),
                'norm_mean': float(signal_tensor_norm.mean()),
                'norm_std': float(signal_tensor_norm.std()),
                'first_5_I': signal_tensor_norm[0, 0, :5].tolist(),
                'first_5_Q': signal_tensor_norm[0, 1, :5].tolist()
            }
            sample_stats.append(stats)
            
            print(f"Transmitter {key}:")
            print(f"  Raw: mean={stats['raw_mean']:.4f}, std={stats['raw_std']:.4f}, range=[{stats['raw_min']:.4f}, {stats['raw_max']:.4f}]")
            print(f"  Normalized: mean={stats['norm_mean']:.6f}, std={stats['norm_std']:.6f}")
            print(f"  First 5 I: {[f'{x:.3f}' for x in stats['first_5_I']]}")
        
        return False
    else:
        print(f"✅ Predictions vary - model working correctly!")
        return True

def test_demo_endpoint_with_dict():
    """Test the demo endpoint with dictionary format"""
    
    print(f"\n" + "="*60)
    print("TESTING DEMO ENDPOINT WITH DICTIONARY FORMAT")
    print("="*60)
    
    # Load signals
    try:
        with open("rf_signals.json", "r") as f:
            signals_dict = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load rf_signals.json: {e}")
        return False
    
    predictions_vary = False
    
    print("🌐 Testing FastAPI endpoint...")
    print("-" * 60)
    
    # Test first 3 transmitters
    for key in sorted(signals_dict.keys())[:3]:
        transmitter_data = signals_dict[key]
        if len(transmitter_data) == 0:
            continue
            
        sample = transmitter_data[0]
        
        # Test the endpoint
        demo_data = {
            "signal": sample,
            "metadata": {
                "true_transmitter": key,
                "test_type": "dict_json_verification"
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
                
                print(f"Transmitter {key}: API Predicted={predicted}, Confidence={confidence:.1%}")
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

def inspect_sample_data():
    """Inspect the actual sample data in detail"""
    
    print(f"\n" + "="*60)
    print("DETAILED SAMPLE INSPECTION")
    print("="*60)
    
    try:
        with open("rf_signals.json", "r") as f:
            signals_dict = json.load(f)
        
        # Look at first few transmitters
        for key in sorted(signals_dict.keys())[:3]:
            print(f"\n📊 Transmitter {key}:")
            transmitter_data = signals_dict[key]
            print(f"   Number of samples: {len(transmitter_data)}")
            
            if len(transmitter_data) > 0:
                sample = transmitter_data[0]
                sample_array = np.array(sample)
                
                print(f"   Sample shape: {sample_array.shape}")
                print(f"   Sample dtype: {sample_array.dtype}")
                
                if len(sample) >= 2:
                    i_channel = np.array(sample[0])
                    q_channel = np.array(sample[1])
                    
                    print(f"   I channel: shape={i_channel.shape}, range=[{i_channel.min():.4f}, {i_channel.max():.4f}], mean={i_channel.mean():.4f}")
                    print(f"   Q channel: shape={q_channel.shape}, range=[{q_channel.min():.4f}, {q_channel.max():.4f}], mean={q_channel.mean():.4f}")
                    
                    # Check if channels are different
                    if np.array_equal(i_channel, q_channel):
                        print(f"   ⚠️  I and Q channels are identical!")
                    else:
                        print(f"   ✓ I and Q channels are different")
                        
                    # Check for all zeros or constant values
                    if np.all(i_channel == 0) or np.all(q_channel == 0):
                        print(f"   ⚠️  One channel is all zeros!")
                    elif np.all(i_channel == i_channel[0]) or np.all(q_channel == q_channel[0]):
                        print(f"   ⚠️  One channel is constant!")
                    else:
                        print(f"   ✓ Channels have varying values")
        
        return True
        
    except Exception as e:
        print(f"❌ Inspection failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 FIXED RF MODEL TEST - DICTIONARY JSON FORMAT")
    print("Testing with your existing rf_signals.json file")
    print("")
    
    # Run tests with proper dictionary handling
    print("Step 1: Inspect sample data")
    inspect_ok = inspect_sample_data()
    
    print(f"\nStep 2: Test model directly with dictionary format")
    model_ok = test_model_with_dict_json()
    
    print(f"\nStep 3: Test demo endpoint")
    api_ok = test_demo_endpoint_with_dict()
    
    # Summary
    print(f"\n" + "="*60)
    print("🎯 RESULTS SUMMARY")
    print("="*60)
    print(f"Sample inspection OK: {'✅' if inspect_ok else '❌'}")
    print(f"Model predictions vary: {'✅' if model_ok else '❌'}")
    print(f"API predictions vary: {'✅' if api_ok else '❌'}")
    
    if not model_ok:
        print(f"\n🚨 PROBLEM: Model always predicts transmitter 2")
        print(f"   Possible causes:")
        print(f"   1. All samples look identical after preprocessing")
        print(f"   2. Model weights are corrupted")
        print(f"   3. Data preprocessing doesn't match training")
        print(f"   4. JSON samples are malformed")
    elif not api_ok:
        print(f"\n🚨 PROBLEM: API endpoint has issues")
        print(f"   - Model works directly but API doesn't")
        print(f"   - Check server logs and preprocess_signal function")
    else:
        print(f"\n🎉 SUCCESS: Everything working correctly!")