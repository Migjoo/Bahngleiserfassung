#!/usr/bin/env python3
"""
Test both models with specific instructions like counting
"""
import sys
import os
from io import BytesIO

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_instruction_following():
    """Test how well both models follow specific instructions"""
    print("Testing Instruction Following")
    print("=" * 40)
    
    try:
        from local_models import get_local_model_manager
        from app import extract_frames_from_video, process_image_locally
        print("+ Components imported")
    except ImportError as e:
        print(f"- Import error: {e}")
        return
    
    # Find video file
    video_files = [f for f in os.listdir('.') if f.endswith('.mp4')]
    if not video_files:
        print("- No MP4 files found")
        return
    
    video_path = video_files[0]
    print(f"+ Using video: {video_path[:40]}...")
    
    # Initialize models
    try:
        local_manager = get_local_model_manager()
        print("+ Models initialized")
    except Exception as e:
        print(f"- Error: {e}")
        return
    
    # Extract a few frames for testing
    try:
        with open(video_path, 'rb') as f:
            video_data = f.read()
        
        video_file = BytesIO(video_data)
        frames = extract_frames_from_video(video_file, fps=0.2)  # Every 5 seconds
        
        if not frames:
            print("- No frames extracted")
            return
        
        # Use first 3 frames for testing
        test_frames = frames[:3]
        print(f"+ Extracted {len(test_frames)} test frames")
        
    except Exception as e:
        print(f"- Frame error: {e}")
        return
    
    # Test different types of instructions
    test_prompts = [
        "Count the number of people in this scene",
        "How many people are visible?",
        "What is the main action happening?",
        "Is there a train in this image?",
        "Describe the setting"
    ]
    
    models = ['CNN (BLIP)', 'Transformer (ViT-GPT2)']
    
    for frame_idx, frame_data in enumerate(test_frames):
        print(f"\n{'='*50}")
        print(f"FRAME {frame_idx + 1} (t={frame_data['timestamp']:.1f}s)")
        print(f"{'='*50}")
        
        for prompt in test_prompts:
            print(f"\nPrompt: '{prompt}'")
            print("-" * 30)
            
            for model in models:
                try:
                    result = process_image_locally(
                        frame_data['frame'],
                        prompt,
                        model,
                        local_manager
                    )
                    
                    if 'error' in result:
                        response = f"Error: {result['error']}"
                    else:
                        response = result.get('generated_text', 'No response')
                    
                    print(f"{model}: {response}")
                    
                except Exception as e:
                    print(f"{model}: Exception - {e}")
            
            print()  # Space between prompts
    
    print("\n" + "=" * 60)
    print("INSTRUCTION FOLLOWING ANALYSIS")
    print("=" * 60)
    print("Key observations to look for:")
    print("1. Does CNN avoid repeating the prompt?")
    print("2. Do models actually count vs describe?")
    print("3. Which model answers questions more directly?")
    print("4. How do they handle yes/no questions?")

if __name__ == "__main__":
    test_instruction_following()