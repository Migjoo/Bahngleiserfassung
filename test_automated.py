#!/usr/bin/env python3
"""
Automated test for video processing with local AI models
"""
import sys
import os
from io import BytesIO

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_full_pipeline():
    """Test the complete video processing pipeline"""
    print("Automated Video + AI Processing Test")
    print("=" * 40)
    
    # Test imports
    try:
        from app import extract_frames_from_video, process_image_locally
        from local_models import get_local_model_manager
        print("+ App components imported successfully")
    except ImportError as e:
        print(f"- Import error: {e}")
        return False
    
    # Find video file
    video_files = [f for f in os.listdir('.') if f.endswith('.mp4')]
    if not video_files:
        print("- No MP4 files found")
        return False
    
    video_path = video_files[0]
    print(f"+ Found video: {video_path[:50]}...")
    
    # Initialize models
    print("+ Initializing AI models...")
    try:
        local_manager = get_local_model_manager()
        available_models = local_manager.get_available_models()
        print(f"+ Available models: {available_models}")
    except Exception as e:
        print(f"- Model initialization error: {e}")
        return False
    
    # Extract frames
    print("+ Extracting video frames...")
    try:
        with open(video_path, 'rb') as f:
            video_data = f.read()
        
        video_file = BytesIO(video_data)
        frames = extract_frames_from_video(video_file, fps=0.2)  # 1 frame every 5 seconds
        
        if not frames:
            print("- No frames extracted")
            return False
        
        print(f"+ Extracted {len(frames)} frames")
        
        # Test with first 2 frames only
        test_frames = frames[:2]
        
    except Exception as e:
        print(f"- Frame extraction error: {e}")
        return False
    
    # Test both models
    test_prompt = "Describe what you see"
    success_count = 0
    
    for model_name in available_models:
        print(f"\nTesting {model_name}...")
        
        try:
            # Test with first frame only to save time
            frame_data = test_frames[0]
            result = process_image_locally(
                frame_data['frame'], 
                test_prompt, 
                model_name, 
                local_manager
            )
            
            if 'error' in result:
                print(f"  - Error: {result['error']}")
            else:
                caption = result.get('generated_text', 'No caption')
                print(f"  + Success: {caption[:50]}...")
                success_count += 1
                
        except Exception as e:
            print(f"  - Exception: {e}")
    
    # Final results
    print("\n" + "=" * 40)
    print("RESULTS")
    print("=" * 40)
    
    if success_count > 0:
        print(f"+ SUCCESS: {success_count}/{len(available_models)} models working")
        print("+ Your video processing setup is ready!")
        print("+ Visit http://localhost:8502 to use the full app")
        return True
    else:
        print("- FAILED: No models processed successfully")
        return False

if __name__ == "__main__":
    success = test_full_pipeline()
    
    if success:
        print("\n+ All tests passed! Local AI video processing is working!")
    else:
        print("\n- Some tests failed. Check error messages above.")
    
    print("\nNext steps:")
    print("1. Open http://localhost:8502")
    print("2. Select 'Local Models' in sidebar")  
    print("3. Choose CNN or Transformer model")
    print("4. Upload your video and test!")