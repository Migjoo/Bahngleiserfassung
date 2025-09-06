#!/usr/bin/env python3
"""
Test local models functionality
"""
import sys
import os
from PIL import Image
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from local_models import LocalModelManager
    print("✓ Successfully imported LocalModelManager")
except ImportError as e:
    print(f"✗ Failed to import LocalModelManager: {e}")
    print("Make sure torch and transformers are installed:")
    print("pip install torch torchvision transformers accelerate sentencepiece")
    sys.exit(1)

def test_local_models():
    """Test both CNN and Transformer models"""
    print("Testing Local AI Models")
    print("=" * 40)
    
    # Initialize model manager
    print("Initializing model manager...")
    try:
        manager = LocalModelManager()
        print("✓ Model manager initialized")
    except Exception as e:
        print(f"✗ Failed to initialize model manager: {e}")
        return
    
    # Get available models
    available_models = manager.get_available_models()
    print(f"Available models: {available_models}")
    
    # Create test images
    test_images = [
        ("Blue Square", Image.new('RGB', (224, 224), color='blue')),
        ("Red Circle", Image.new('RGB', (224, 224), color='red')),
        ("Green Background", Image.new('RGB', (224, 224), color='green'))
    ]
    
    test_prompt = "Describe what you see in this image"
    
    # Test each model with each image
    for model_name in available_models:
        print(f"\n🤖 Testing {model_name}")
        print("-" * 30)
        
        for image_name, image in test_images:
            print(f"Processing {image_name}...")
            try:
                result = manager.generate_caption(model_name, image, test_prompt)
                print(f"  Result: {result}")
            except Exception as e:
                print(f"  ✗ Error: {e}")
            print()

def test_model_info():
    """Test model information retrieval"""
    print("\n📋 Model Information")
    print("=" * 40)
    
    try:
        manager = LocalModelManager()
        model_info = manager.get_model_info()
        
        for model_name, info in model_info.items():
            print(f"\n{model_name}:")
            print(f"  Description: {info['description']}")
            print(f"  Strengths: {info['strengths']}")
            print(f"  Size: {info['size']}")
            
    except Exception as e:
        print(f"✗ Error getting model info: {e}")

if __name__ == "__main__":
    print("🧪 Local Models Test Suite")
    print("This will download models on first run (~3GB total)")
    print()
    
    # Test model info first (doesn't require model downloads)
    test_model_info()
    
    # Ask user if they want to proceed with model testing
    response = input("\nProceed with model testing? This will download models if not cached. (y/n): ")
    if response.lower().startswith('y'):
        test_local_models()
    else:
        print("Skipping model testing.")
    
    print("\n✅ Test complete!")