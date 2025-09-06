#!/usr/bin/env python3
"""
Simple test to see raw model outputs for counting
"""
import sys
import os
from io import BytesIO

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_simple_counting():
    """Test counting with both models"""
    print("Simple Counting Test")
    print("=" * 30)
    
    try:
        from local_models import get_local_model_manager
        from app import extract_frames_from_video, process_image_locally
        print("+ Imported successfully")
    except ImportError as e:
        print(f"- Import error: {e}")
        return
    
    # Find video file
    video_files = [f for f in os.listdir('.') if f.endswith('.mp4')]
    if not video_files:
        print("- No video files found")
        return
    
    video_path = video_files[0]
    print(f"+ Using: {video_path[:30]}...")
    
    # Get models
    try:
        local_manager = get_local_model_manager()
        print("+ Models ready")
    except Exception as e:
        print(f"- Error: {e}")
        return
    
    # Get one frame
    try:
        with open(video_path, 'rb') as f:
            video_data = f.read()
        
        video_file = BytesIO(video_data)
        frames = extract_frames_from_video(video_file, fps=0.1)
        
        if not frames:
            print("- No frames")
            return
        
        test_frame = frames[1]['frame']  # Use second frame which showed a person
        print(f"+ Using frame at t={frames[1]['timestamp']:.1f}s")
        
    except Exception as e:
        print(f"- Frame error: {e}")
        return
    
    # Test specific prompts
    test_prompts = [
        "Count the number of people in this scene",
        "How many people do you see?", 
        "one person or two people?",
        "Describe what you see"
    ]
    
    for prompt in test_prompts:
        print(f"\n--- Prompt: '{prompt}' ---")
        
        # Test CNN
        try:
            result = process_image_locally(test_frame, prompt, 'CNN (BLIP)', local_manager)
            cnn_response = result.get('generated_text', 'No response') if 'error' not in result else f"Error: {result['error']}"
            print(f"CNN: '{cnn_response}'")
        except Exception as e:
            print(f"CNN: Exception - {e}")
        
        # Test Transformer  
        try:
            result = process_image_locally(test_frame, prompt, 'Transformer (ViT-GPT2)', local_manager)
            trans_response = result.get('generated_text', 'No response') if 'error' not in result else f"Error: {result['error']}"
            print(f"Transformer: '{trans_response}'")
        except Exception as e:
            print(f"Transformer: Exception - {e}")
    
    print("\n" + "=" * 40)
    print("ANALYSIS:")
    print("- Neither model is designed for counting")
    print("- Both provide descriptions, not counts") 
    print("- Transformer (ViT-GPT2) is better for descriptions")
    print("- CNN (BLIP) has prompt repetition issues")
    print("\nRECOMMENDAT ION:")
    print("Use descriptive prompts like:")
    print("  'Describe what you see'")
    print("  'What is happening in this image?'")
    print("Rather than counting prompts.")

if __name__ == "__main__":
    test_simple_counting()