#!/usr/bin/env python3
"""
Test the encoding fix for CNN model outputs
"""
import sys
import os
from io import BytesIO
from PIL import Image

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_encoding_fix():
    """Test if the encoding issue is fixed"""
    print("Testing Encoding Fix for CNN Model")
    print("=" * 40)
    
    try:
        from local_models import get_local_model_manager
        from app import extract_frames_from_video, process_image_locally
        print("+ Successfully imported components")
    except ImportError as e:
        print(f"- Import error: {e}")
        return
    
    # Find video file
    video_files = [f for f in os.listdir('.') if f.endswith('.mp4')]
    if not video_files:
        print("- No MP4 files found")
        return
    
    video_path = video_files[0]
    print(f"+ Using video: {video_path[:50]}...")
    
    # Initialize models
    try:
        local_manager = get_local_model_manager()
        print("+ Models initialized")
    except Exception as e:
        print(f"- Model error: {e}")
        return
    
    # Extract one frame for testing
    try:
        with open(video_path, 'rb') as f:
            video_data = f.read()
        
        video_file = BytesIO(video_data)
        frames = extract_frames_from_video(video_file, fps=0.1)  # Just first frame
        
        if not frames:
            print("- No frames extracted")
            return
            
        test_frame = frames[0]['frame']
        print("+ Extracted test frame")
        
    except Exception as e:
        print(f"- Frame extraction error: {e}")
        return
    
    # Test CNN model with cleaned output
    print("\nTesting CNN (BLIP) with encoding fix:")
    print("-" * 40)
    
    try:
        result = process_image_locally(
            test_frame,
            "Describe what you see",
            'CNN (BLIP)',
            local_manager
        )
        
        if 'error' in result:
            print(f"- Error: {result['error']}")
        else:
            caption = result.get('generated_text', 'No caption')
            print(f"+ Result: {caption}")
            
            # Check for problematic characters
            has_issues = False
            for char in caption:
                if ord(char) > 127:
                    print(f"- Found non-ASCII character: {repr(char)} (ord: {ord(char)})")
                    has_issues = True
            
            if not has_issues:
                print("+ No encoding issues detected!")
            else:
                print("- Still has encoding issues")
                
    except Exception as e:
        print(f"- Exception: {e}")
    
    # Test Transformer for comparison
    print("\nTesting Transformer (ViT-GPT2) for comparison:")
    print("-" * 40)
    
    try:
        result = process_image_locally(
            test_frame,
            "Describe what you see",
            'Transformer (ViT-GPT2)',
            local_manager
        )
        
        if 'error' in result:
            print(f"- Error: {result['error']}")
        else:
            caption = result.get('generated_text', 'No caption')
            print(f"+ Result: {caption}")
            
    except Exception as e:
        print(f"- Exception: {e}")

if __name__ == "__main__":
    test_encoding_fix()