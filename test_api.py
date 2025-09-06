#!/usr/bin/env python3
"""
Simple API test to check Hugging Face connectivity
"""
import requests
import json
from PIL import Image
import base64
from io import BytesIO

# Load settings
def load_settings():
    try:
        with open('settings.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def test_simple_api():
    """Test basic API connectivity"""
    settings = load_settings()
    api_token = settings.get('hugging_face_api_token')
    
    if not api_token:
        print("No API token found")
        return
    
    print(f"Testing API connectivity with token: {api_token[:10]}...")
    
    # Test with a simple image captioning model
    API_URL = "https://api-inference.huggingface.co/models/nlpconnect/vit-gpt2-image-captioning"
    headers = {"Authorization": f"Bearer {api_token}"}
    
    # Create a simple test image (solid color)
    test_image = Image.new('RGB', (224, 224), color='blue')
    
    # Convert to bytes
    buffer = BytesIO()
    test_image.save(buffer, format="JPEG")
    
    print("Making API request...")
    
    response = requests.post(
        API_URL,
        headers=headers,
        files={"data": buffer.getvalue()}
    )
    
    print(f"Response status: {response.status_code}")
    print(f"Response headers: {dict(response.headers)}")
    
    if response.status_code == 200:
        print("SUCCESS!")
        print(f"Response: {response.json()}")
    else:
        print(f"ERROR: {response.text}")

if __name__ == "__main__":
    test_simple_api()