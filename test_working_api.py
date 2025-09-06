#!/usr/bin/env python3
"""
Test with known working Hugging Face models
"""
import requests
import json
from PIL import Image
from io import BytesIO

def load_settings():
    try:
        with open('settings.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def test_working_models():
    """Test with models that are known to work"""
    settings = load_settings()
    api_token = settings.get('hugging_face_api_token')
    
    if not api_token:
        print("No API token found")
        return
    
    print(f"Testing with token: {api_token[:10]}...")
    
    # Create a simple test image
    test_image = Image.new('RGB', (224, 224), color='red')
    buffer = BytesIO()
    test_image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    
    # Test different API approaches
    models_to_test = [
        "Salesforce/blip-image-captioning-base-large",
        "microsoft/DialoGPT-medium",
        "google/vit-base-patch16-224"
    ]
    
    for model_name in models_to_test:
        print(f"\nTesting {model_name}...")
        
        API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
        headers = {"Authorization": f"Bearer {api_token}"}
        
        # Try different payload formats
        response = requests.post(
            API_URL,
            headers=headers,
            data=image_bytes
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            print(f"SUCCESS! Response: {response.json()}")
            break
        elif response.status_code == 503:
            print("Model is loading, please wait...")
        else:
            print(f"Error: {response.text}")

    # Also test token validity
    print("\nTesting token validity...")
    headers = {"Authorization": f"Bearer {api_token}"}
    response = requests.get("https://huggingface.co/api/whoami", headers=headers)
    print(f"Token check status: {response.status_code}")
    if response.status_code == 200:
        print(f"Token is valid. User info: {response.json()}")
    else:
        print(f"Token validation failed: {response.text}")

if __name__ == "__main__":
    test_working_models()