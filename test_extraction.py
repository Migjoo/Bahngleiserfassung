#!/usr/bin/env python3
"""
Test script for video extraction and processing functionality
"""
import os
import sys
import json
from io import BytesIO
import tempfile

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import extract_frames_from_video, query_huggingface_api, load_settings

def test_video_extraction():
    """Test video extraction with the problematic video file"""
    # Find the actual video file in the directory
    video_files = [f for f in os.listdir('.') if f.endswith('.mp4')]
    
    if not video_files:
        print("No MP4 files found in current directory")
        return False
    
    video_path = video_files[0]  # Use the first MP4 file found
    print(f"Using video file: {video_path}")
    print(f"Video size: {os.path.getsize(video_path) / (1024*1024):.1f} MB")
    
    # Create a file-like object for testing
    with open(video_path, 'rb') as f:
        video_data = f.read()
    
    # Create BytesIO object to simulate uploaded file
    video_file = BytesIO(video_data)
    
    print("\nTesting video frame extraction...")
    try:
        frames = extract_frames_from_video(video_file, fps=0.5)  # Extract 1 frame every 2 seconds
        
        if frames:
            print(f"Successfully extracted {len(frames)} frames")
            for i, frame_data in enumerate(frames[:3]):  # Show first 3 frames
                print(f"  Frame {i}: {frame_data['timestamp']:.1f}s, size: {frame_data['frame'].size}")
            return frames
        else:
            print("No frames extracted")
            return None
            
    except Exception as e:
        print(f"Error during extraction: {e}")
        return None

def test_api_integration(frames):
    """Test Hugging Face API integration"""
    if not frames:
        print("No frames to test API with")
        return
    
    # Load settings
    settings = load_settings()
    api_token = settings.get('hugging_face_api_token')
    
    if not api_token:
        print("No API token found in settings.json")
        return
    
    print(f"\nTesting API integration...")
    print(f"Using token: {api_token[:10]}...")
    
    # Test with first frame and simple prompt
    test_frame = frames[0]['frame']
    test_prompt = "Describe what you see in this image"
    
    # Try multiple models
    models_to_test = [
        "nlpconnect/vit-gpt2-image-captioning",
        "Salesforce/blip-image-captioning-base",
        "microsoft/git-large-coco"
    ]
    
    for model in models_to_test:
        print(f"\nTesting with model: {model}")
        print(f"Prompt: {test_prompt}")
        
        try:
            result = query_huggingface_api(test_frame, test_prompt, model, api_token)
            
            if 'error' in result:
                print(f"API Error: {result['error']}")
            else:
                print("API call successful!")
                print(f"Result: {result}")
                break  # Stop on first successful model
                
        except Exception as e:
            print(f"Exception during API call: {e}")
            continue

def main():
    print("Testing Video Frame Analyzer Functionality")
    print("=" * 50)
    
    # Test 1: Video extraction
    frames = test_video_extraction()
    
    # Test 2: API integration (if frames extracted successfully)
    if frames:
        test_api_integration(frames)
    
    print("\n" + "=" * 50)
    print("Testing complete!")

if __name__ == "__main__":
    main()