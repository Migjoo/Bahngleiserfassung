#!/usr/bin/env python3
"""
Clean report of person on tracks detection results
"""
import sys
import os
from io import BytesIO
import re

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_detection_report():
    """Create clean detection report"""
    print("PERSON ON TRACKS DETECTION REPORT")
    print("=" * 50)
    
    try:
        from local_models import get_local_model_manager
        from app import extract_frames_from_video, process_image_locally
    except ImportError as e:
        print(f"Import error: {e}")
        return
    
    # Find video
    video_files = [f for f in os.listdir('.') if f.endswith('.mp4')]
    if not video_files:
        print("No video files found")
        return
    
    video_path = video_files[0]
    print(f"Video: {video_path}")
    print("Model: Transformer (ViT-GPT2)")
    print("Prompt: 'Describe the scene focusing on people and train tracks'")
    print()
    
    # Get model
    try:
        local_manager = get_local_model_manager()
    except Exception as e:
        print(f"Model error: {e}")
        return
    
    # Extract frames
    try:
        with open(video_path, 'rb') as f:
            video_data = f.read()
        
        video_file = BytesIO(video_data)
        frames = extract_frames_from_video(video_file, fps=0.5)
        
        if not frames:
            print("No frames extracted")
            return
        
        print(f"Analyzing {len(frames)} frames...")
        print()
        
    except Exception as e:
        print(f"Frame extraction error: {e}")
        return
    
    # Analyze each frame
    results = []
    person_frames = []
    
    for i, frame_data in enumerate(frames):
        frame_num = i + 1
        timestamp = frame_data['timestamp']
        
        try:
            result = process_image_locally(
                frame_data['frame'],
                "Describe the scene focusing on people and train tracks",
                'Transformer (ViT-GPT2)',
                local_manager
            )
            
            if 'error' in result:
                description = f"Error: {result['error']}"
                person_detected = False
            else:
                description = result.get('generated_text', 'No response')
                person_detected = detect_person_on_track(description)
            
            results.append({
                'frame': frame_num,
                'time': timestamp,
                'description': description,
                'person_on_track': person_detected
            })
            
            if person_detected:
                person_frames.append(frame_num)
            
            status = "[PERSON ON TRACK]" if person_detected else "[CLEAR]"
            print(f"Frame {frame_num:2d} ({timestamp:4.1f}s): {status}")
            print(f"    {description}")
            print()
            
        except Exception as e:
            print(f"Frame {frame_num:2d} ({timestamp:4.1f}s): ERROR - {e}")
            print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total = len(frames)
    detected = len(person_frames)
    
    print(f"Total frames: {total}")
    print(f"Person detected on tracks: {detected}")
    print(f"Detection rate: {100 * detected / total:.1f}%")
    
    if person_frames:
        print(f"Frames with person: {', '.join(map(str, person_frames))}")
        timestamps = [results[f-1]['time'] for f in person_frames]
        print(f"Time range: {min(timestamps):.1f}s - {max(timestamps):.1f}s")
        
        print(f"\nDETAILED DETECTIONS:")
        for frame_num in person_frames:
            frame_data = results[frame_num-1]
            print(f"  Frame {frame_num} ({frame_data['time']:.1f}s): {frame_data['description']}")
    else:
        print("No clear person detections on tracks")
    
    print(f"\nRELIABILITY ASSESSMENT:")
    print("- Model designed for image description, not object detection")
    print("- Results based on text analysis of descriptions")
    print("- Best used as preliminary screening, not definitive detection")
    
    return results

def detect_person_on_track(description):
    """Simple detection logic based on description text"""
    if not description:
        return False
    
    desc = description.lower()
    
    # Person indicators
    person_words = ['person', 'man', 'boy', 'woman', 'girl', 'people']
    has_person = any(word in desc for word in person_words)
    
    # Track indicators  
    track_words = ['track', 'tracks', 'rail', 'rails']
    has_track = any(word in desc for word in track_words)
    
    # Position indicators
    position_words = ['on', 'standing', 'walking']
    has_position = any(word in desc for word in position_words)
    
    # Strong indicators
    strong_patterns = ['standing on', 'walking on', 'on the track', 'on track']
    has_strong = any(pattern in desc for pattern in strong_patterns)
    
    return has_strong or (has_person and has_track and has_position)

if __name__ == "__main__":
    create_detection_report()