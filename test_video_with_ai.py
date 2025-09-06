#!/usr/bin/env python3
"""
Test video processing with local AI models
"""
import sys
import os
from io import BytesIO
from PIL import Image
import tempfile

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app import extract_frames_from_video, process_image_locally
    from local_models import get_local_model_manager
    print("+ Successfully imported app components")
except ImportError as e:
    print(f"- Import error: {e}")
    sys.exit(1)

def test_video_processing_with_ai():
    """Test video processing with local AI models"""
    print("Testing Video Processing with Local AI Models")
    print("=" * 50)
    
    # Find video file
    video_files = [f for f in os.listdir('.') if f.endswith('.mp4')]
    if not video_files:
        print("- No MP4 files found")
        return False
    
    video_path = video_files[0]
    print(f"+ Using video: {video_path}")
    
    # Initialize local model manager
    print("\nInitializing AI models...")
    try:
        local_manager = get_local_model_manager()
        available_models = local_manager.get_available_models()
        print(f"+ Available models: {available_models}")
    except Exception as e:
        print(f"- Error initializing models: {e}")
        return False
    
    # Load video and extract frames
    print(f"\nExtracting frames from video...")
    try:
        with open(video_path, 'rb') as f:
            video_data = f.read()
        
        video_file = BytesIO(video_data)
        frames = extract_frames_from_video(video_file, fps=0.2)  # 1 frame every 5 seconds
        
        if not frames:
            print("- No frames extracted")
            return False
        
        print(f"+ Extracted {len(frames)} frames")
        
        # Test with first 3 frames max to avoid long processing
        test_frames = frames[:3]
        
    except Exception as e:
        print(f"- Error extracting frames: {e}")
        return False
    
    # Test both AI models
    test_prompt = "Describe what you see in this image"
    results = {}
    
    for model_name in available_models:
        print(f"\n🤖 Testing {model_name}")
        print("-" * 30)
        
        model_results = []
        
        for i, frame_data in enumerate(test_frames):
            print(f"Processing frame {i+1}/{len(test_frames)} (t={frame_data['timestamp']:.1f}s)...")
            
            try:
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
                    print(f"  + Result: {caption}")
                    model_results.append({
                        'frame': i,
                        'timestamp': frame_data['timestamp'],
                        'caption': caption
                    })
                    
            except Exception as e:
                print(f"  - Exception: {e}")
        
        results[model_name] = model_results
    
    # Summary
    print("\n" + "=" * 50)
    print("PROCESSING SUMMARY")
    print("=" * 50)
    
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        if model_results:
            print(f"  + Successfully processed {len(model_results)} frames")
            for result in model_results:
                print(f"  Frame {result['frame']} ({result['timestamp']:.1f}s): {result['caption'][:60]}...")
        else:
            print("  - No successful results")
    
    return len(results) > 0 and any(len(r) > 0 for r in results.values())

def test_model_info():
    """Test model information display"""
    print("\n📋 Model Information")
    print("=" * 30)
    
    try:
        local_manager = get_local_model_manager()
        model_info = local_manager.get_model_info()
        
        for model_name, info in model_info.items():
            print(f"\n{model_name}:")
            print(f"  Description: {info['description']}")
            print(f"  Strengths: {info['strengths']}")
            print(f"  Size: {info['size']}")
            
        return True
        
    except Exception as e:
        print(f"- Error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Video + AI Models Test Suite")
    print("This will test both CNN and Transformer models with your video")
    print("Note: First run will download AI models (~3GB total)")
    print()
    
    # Test model info first
    info_ok = test_model_info()
    
    if info_ok:
        print("\nProceed with video processing test?")
        print("This will download AI models if not cached (~3GB)")
        response = input("Continue? (y/n): ")
        
        if response.lower().startswith('y'):
            success = test_video_processing_with_ai()
            
            if success:
                print("\n+ Video processing with local AI models SUCCESSFUL!")
                print("+ Your setup is ready to use!")
            else:
                print("\n- Some issues encountered during processing")
        else:
            print("Skipping video processing test.")
    
    print(f"\n+ Test complete! Check the Streamlit app at: http://localhost:8502")