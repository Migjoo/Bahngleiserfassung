#!/usr/bin/env python3
"""
Detect if a person is on train tracks using the best model and prompt
"""
import sys
import os
from io import BytesIO
import re

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def analyze_person_on_tracks():
    """Analyze all frames to detect if person is on train tracks"""
    print("PERSON ON TRACKS DETECTION")
    print("=" * 40)
    print("Using: Transformer (ViT-GPT2) - Best performing model")
    print()
    
    try:
        from local_models import get_local_model_manager
        from app import extract_frames_from_video, process_image_locally
        print("+ Components loaded")
    except ImportError as e:
        print(f"- Import error: {e}")
        return
    
    # Find video
    video_files = [f for f in os.listdir('.') if f.endswith('.mp4')]
    if not video_files:
        print("- No video files found")
        return
    
    video_path = video_files[0]
    print(f"+ Video: {video_path}")
    
    # Initialize model
    try:
        local_manager = get_local_model_manager()
        print("+ Transformer model ready")
    except Exception as e:
        print(f"- Model error: {e}")
        return
    
    # Extract frames
    try:
        with open(video_path, 'rb') as f:
            video_data = f.read()
        
        video_file = BytesIO(video_data)
        frames = extract_frames_from_video(video_file, fps=0.5)  # Every 2 seconds
        
        if not frames:
            print("- No frames extracted")
            return
        
        print(f"+ Extracted {len(frames)} frames for analysis")
        print()
        
    except Exception as e:
        print(f"- Frame extraction error: {e}")
        return
    
    # Optimized prompt for person detection on tracks
    optimal_prompt = "Describe the scene focusing on people and train tracks"
    
    print("ANALYSIS RESULTS:")
    print("=" * 50)
    
    person_detected_frames = []
    results = []
    
    for i, frame_data in enumerate(frames):
        frame_num = i + 1
        timestamp = frame_data['timestamp']
        
        try:
            # Use the best model (Transformer) with optimal prompt
            result = process_image_locally(
                frame_data['frame'],
                optimal_prompt,
                'Transformer (ViT-GPT2)',
                local_manager
            )
            
            if 'error' in result:
                response = f"Error: {result['error']}"
                person_on_track = False
            else:
                response = result.get('generated_text', 'No response')
                
                # Analyze response for person-on-track indicators
                person_on_track = detect_person_on_track_from_text(response)
            
            # Store result
            results.append({
                'frame': frame_num,
                'timestamp': timestamp,
                'description': response,
                'person_on_track': person_on_track
            })
            
            if person_on_track:
                person_detected_frames.append(frame_num)
            
            # Display result
            status = "🚨 PERSON ON TRACK" if person_on_track else "✓ Clear"
            print(f"Frame {frame_num:2d} ({timestamp:4.1f}s): {status}")
            print(f"    Description: {response}")
            print()
            
        except Exception as e:
            print(f"Frame {frame_num:2d} ({timestamp:4.1f}s): ERROR - {e}")
            results.append({
                'frame': frame_num,
                'timestamp': timestamp,
                'description': f"Error: {e}",
                'person_on_track': False
            })
            print()
    
    # Summary analysis
    print("=" * 60)
    print("DETECTION SUMMARY")
    print("=" * 60)
    
    total_frames = len(frames)
    person_frames = len(person_detected_frames)
    
    print(f"Total frames analyzed: {total_frames}")
    print(f"Frames with person on tracks: {person_frames}")
    print(f"Percentage: {100 * person_frames / total_frames:.1f}%")
    
    if person_detected_frames:
        print(f"\nPerson detected in frames: {', '.join(map(str, person_detected_frames))}")
        
        # Find time ranges
        timestamps = [results[f-1]['timestamp'] for f in person_detected_frames]
        print(f"Time periods: {min(timestamps):.1f}s - {max(timestamps):.1f}s")
    else:
        print("\nNo person clearly detected on train tracks")
    
    print(f"\n📊 CONFIDENCE ASSESSMENT:")
    confidence_scores = []
    for r in results:
        if r['person_on_track']:
            # Assess confidence based on description keywords
            desc = r['description'].lower()
            confidence = 0.5  # Base confidence
            
            if any(word in desc for word in ['person', 'man', 'boy', 'woman', 'people']):
                confidence += 0.3
            if any(word in desc for word in ['standing', 'walking', 'on', 'track', 'rail']):
                confidence += 0.2
            
            confidence_scores.append(min(confidence, 1.0))
    
    if confidence_scores:
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        print(f"Average detection confidence: {avg_confidence:.1f}/1.0")
    else:
        print("No confident detections")
    
    # Save results
    print(f"\n+ Analysis complete!")
    return results

def detect_person_on_track_from_text(description):
    """Analyze text description to determine if person is on train tracks"""
    if not description:
        return False
    
    desc_lower = description.lower()
    
    # Keywords indicating person presence
    person_keywords = ['person', 'man', 'boy', 'woman', 'girl', 'people', 'someone']
    
    # Keywords indicating track/rail location
    track_keywords = ['track', 'tracks', 'rail', 'rails', 'railway']
    
    # Positioning keywords
    position_keywords = ['on', 'standing', 'walking', 'sitting', 'near', 'beside', 'next to']
    
    # Check for person presence
    has_person = any(keyword in desc_lower for keyword in person_keywords)
    
    # Check for track presence
    has_track = any(keyword in desc_lower for keyword in track_keywords)
    
    # Check for positioning that suggests person is ON the tracks
    has_position = any(keyword in desc_lower for keyword in position_keywords)
    
    # Look for specific phrases that strongly suggest person on tracks
    strong_indicators = [
        'standing on', 'walking on', 'on the track', 'on track', 'on rail',
        'person.*track', 'man.*track', 'boy.*track'
    ]
    
    has_strong_indicator = any(re.search(pattern, desc_lower) for pattern in strong_indicators)
    
    # Decision logic
    if has_strong_indicator:
        return True
    elif has_person and has_track and has_position:
        return True
    else:
        return False

if __name__ == "__main__":
    analyze_person_on_tracks()