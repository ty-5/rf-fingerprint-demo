#!/usr/bin/env python3
"""
Debug FastAPI Server - Shows exactly what's going wrong
"""

from fastapi import FastAPI
import sys
import os

# Create basic app first
app = FastAPI(title="Debug RF Classifier")

@app.get("/")
async def debug_root():
    return {"message": "Debug server is running!"}

@app.get("/debug")
async def debug_info():
    """Debug endpoint to check what's available"""
    
    debug_info = {
        "python_version": sys.version,
        "current_directory": os.getcwd(),
        "python_path": sys.path,
        "files_in_current_dir": os.listdir("."),
        "files_in_parent_dir": os.listdir("..") if os.path.exists("..") else "No parent dir"
    }
    
    # Test imports
    try:
        import torch
        debug_info["torch_available"] = True
        debug_info["torch_version"] = torch.__version__
    except Exception as e:
        debug_info["torch_available"] = False
        debug_info["torch_error"] = str(e)
    
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from CNN_Extended import CNNFingerprinter
        debug_info["cnn_extended_available"] = True
    except Exception as e:
        debug_info["cnn_extended_available"] = False
        debug_info["cnn_extended_error"] = str(e)
    
    try:
        import configs
        debug_info["configs_available"] = True
        debug_info["window_size"] = getattr(configs, 'window_size', 'Not found')
        debug_info["datatype"] = str(getattr(configs, 'datatype', 'Not found'))
    except Exception as e:
        debug_info["configs_available"] = False
        debug_info["configs_error"] = str(e)
    
    # Check for model weights
    weight_paths = [
        "RF_Model_Weights_98%.pth",
        "../RF_Model_Weights_98%.pth",
        "RF_Model_Weights.pth",
        "../RF_Model_Weights.pth"
    ]
    
    debug_info["weight_file_search"] = {}
    for path in weight_paths:
        debug_info["weight_file_search"][path] = os.path.exists(path)
    
    return debug_info

if __name__ == "__main__":
    import uvicorn
    print("🔍 Starting DEBUG server...")
    print("📍 Debug info at: http://localhost:8000/debug")
    try:
        uvicorn.run("debug_server:app", host="0.0.0.0", port=8000, reload=False)
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        input("Press Enter to continue...")