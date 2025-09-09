#!/usr/bin/env python3
"""
Model processing utilities for local and remote AI models
"""
import requests
import base64
from io import BytesIO
from PIL import Image
from typing import Dict, Any, Optional


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 string"""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str


def process_image_locally(image: Image.Image, prompt: str, model_name: str, local_manager) -> Dict[str, Any]:
    """
    Process image using local models
    """
    try:
        if model_name == "Person on Track Detector":
            # Special handling for person-on-track detection
            result = local_manager.person_on_track_detector.detect_person_on_track(image)
            return {"person_on_track_detection": result}
        else:
            caption = local_manager.generate_caption(model_name, image, prompt)
            return {"generated_text": caption}
    except Exception as e:
        return {"error": f"Local processing failed: {str(e)}"}


def query_huggingface_api(image: Image.Image, prompt: str, model_name: str, api_token: str) -> Dict[str, Any]:
    """
    Query Hugging Face API with image and prompt
    """
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {api_token}"}
    
    # Convert image to base64
    img_base64 = image_to_base64(image)
    
    # Prepare payload based on model type
    if "blip" in model_name.lower():
        # For BLIP models, send image directly
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        response = requests.post(
            API_URL,
            headers=headers,
            files={"file": buffer.getvalue()}
        )
    else:
        # For other vision-language models
        payload = {
            "inputs": {
                "image": img_base64,
                "text": prompt
            }
        }
        response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"API request failed: {response.status_code} - {response.text}"}


def process_frame(frame_data: Dict, config: Dict[str, Any], local_manager=None) -> Dict[str, Any]:
    """
    Process a single frame using the configured model
    """
    model_type = config["model_type"]
    selected_model = config["selected_model"]
    prompt = config.get("prompt", "")
    api_token = config.get("api_token")
    
    # Process frame based on model type
    if model_type == "Local Models" and local_manager:
        result = process_image_locally(
            frame_data['frame'],
            prompt,
            selected_model,
            local_manager
        )
    else:
        result = query_huggingface_api(
            frame_data['frame'],
            prompt,
            selected_model,
            api_token
        )
    
    return result