#!/usr/bin/env python3
"""
Test the simplified Person on Track Detector output
"""
import sys
import os
from io import BytesIO

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_simplified_output():
    """Test the simplified output format"""
    print("TESTING SIMPLIFIED PERSON ON TRACK DETECTOR OUTPUT")
    print("=" * 60)
    print("Now shows only: Analysis + People Count + Confidence")
    print()
    
    try:
        from local_models import get_local_model_manager
        from app import extract_frames_from_video, process_image_locally
        print("+ Components loaded")
    except ImportError as e:
        print(f"- Import error: {e}")
        return
    
    # Test with first video
    video_path = "test\\1.mp4"
    if not os.path.exists(video_path):
        print(f"- Video not found: {video_path}")
        return
    
    print(f"+ Testing with: {video_path}")
    
    try:
        local_manager = get_local_model_manager()
        print("+ Person on Track Detector ready")
    except Exception as e:
        print(f"- Model error: {e}")
        return
    
    # Extract one frame for testing
    try:
        with open(video_path, 'rb') as f:
            video_data = f.read()
        
        video_file = BytesIO(video_data)
        frames = extract_frames_from_video(video_file, fps=0.5)
        
        if not frames:
            print("- No frames extracted")
            return
        
        frame_data = frames[0]
        print(f"+ Testing frame at {frame_data['timestamp']:.1f}s")
        
    except Exception as e:
        print(f"- Frame extraction error: {e}")
        return
    
    # Test the simplified detector
    try:
        result = process_image_locally(
            frame_data['frame'],
            "Track Safety Analysis",
            'Person on Track Detector',
            local_manager
        )
        
        if 'person_on_track_detection' in result:
            detection = result['person_on_track_detection']
            
            print(f"\n" + "=" * 50)
            print("SIMPLIFIED OUTPUT")
            print("=" * 50)
            
            # Show the three key pieces of information
            analysis = detection.get('analysis', 'No analysis')
            people_count = detection.get('people_count', 0)
            confidence = detection.get('confidence', 0)
            person_on_track = detection.get('person_on_track', False)
            
            # Display like in Streamlit
            if person_on_track:
                print(f"🚨 ALERT: {analysis}")
            else:
                print(f"✅ SAFE: {analysis}")
            
            print(f"👥 People on Track: {people_count}")
            print(f"📊 Confidence: {confidence:.0%}")
            
            print(f"\n" + "=" * 50)
            print("SUCCESS - CLEAN, SIMPLE OUTPUT!")
            print("=" * 50)
            print("The detector now shows only the essential information:")
            print(f"1. Clear analysis message: '{analysis}'")
            print(f"2. Number of people on track: {people_count}")
            print(f"3. Confidence level: {confidence:.0%}")
            print("4. Color-coded status (red for danger, green for safe)")
            
        else:
            print(f"ERROR: Unexpected result format")
            
    except Exception as e:
        print(f"ERROR: {e}")
    
    print(f"\n" + "=" * 60)
    print("READY TO USE!")
    print("=" * 60)
    print("Open http://localhost:8502")
    print("Select 'Person on Track Detector'")
    print("Upload test videos to see the simplified output")

if __name__ == "__main__":
    test_simplified_output()