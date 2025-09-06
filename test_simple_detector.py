#!/usr/bin/env python3
"""
Test the NEW simple but reliable Person on Track Detector
"""
import sys
import os
from io import BytesIO
import glob

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_simple_detector():
    """Test the new simple detector on multiple videos"""
    print("TESTING NEW SIMPLE PERSON ON TRACK DETECTOR")
    print("=" * 60)
    print("Much simpler approach - only uses Transformer model")
    print("Should give more accurate results!")
    print()
    
    try:
        from local_models import get_local_model_manager
        from app import extract_frames_from_video, process_image_locally
        print("+ Components loaded")
    except ImportError as e:
        print(f"- Import error: {e}")
        return
    
    # Test multiple videos
    test_videos = glob.glob("test\\*.mp4")[:4]  # Test first 4 videos
    if not test_videos:
        print("- No test videos found")
        return
    
    print(f"+ Testing {len(test_videos)} videos")
    
    try:
        local_manager = get_local_model_manager()
        print("+ Simple detector ready")
    except Exception as e:
        print(f"- Model error: {e}")
        return
    
    all_results = []
    
    # Test each video
    for video_idx, video_path in enumerate(test_videos):
        video_name = os.path.basename(video_path)
        print(f"\n" + "=" * 50)
        print(f"VIDEO {video_idx + 1}: {video_name}")
        print("=" * 50)
        
        try:
            # Extract frames
            with open(video_path, 'rb') as f:
                video_data = f.read()
            
            video_file = BytesIO(video_data)
            frames = extract_frames_from_video(video_file, fps=0.5)
            
            if not frames:
                print(f"- No frames from {video_name}")
                continue
            
            # Test first frame from each video
            frame_data = frames[0]
            timestamp = frame_data['timestamp']
            
            print(f"\nFrame 1 ({timestamp:.1f}s):")
            print("-" * 30)
            
            try:
                result = process_image_locally(
                    frame_data['frame'],
                    "Track Safety Analysis",
                    'Person on Track Detector',
                    local_manager
                )
                
                if 'person_on_track_detection' in result:
                    detection = result['person_on_track_detection']
                    
                    people_count = detection.get('people_count', 0)
                    confidence = detection.get('confidence', 0)
                    analysis = detection.get('analysis', 'No analysis')
                    person_on_track = detection.get('person_on_track', False)
                    
                    # Show detailed analysis
                    detailed = detection.get('detailed_analysis', {})
                    scene_desc = detailed.get('scene_description', 'N/A')
                    person_mentions = detailed.get('person_mentions', 0)
                    track_mentions = detailed.get('track_mentions', 0)
                    
                    # Display results
                    if person_on_track:
                        print(f"ALERT: {analysis}")
                    else:
                        print(f"SAFE: {analysis}")
                    
                    print(f"People Count: {people_count}")
                    print(f"Confidence: {confidence:.0%}")
                    print(f"Scene: '{scene_desc}'")
                    print(f"Keywords: Person={person_mentions}, Track={track_mentions}")
                    
                    all_results.append({
                        'video': video_name,
                        'on_track': person_on_track,
                        'people_count': people_count,
                        'confidence': confidence,
                        'analysis': analysis,
                        'scene': scene_desc
                    })
                    
                else:
                    print(f"ERROR: Unexpected result format")
                    
            except Exception as e:
                print(f"ERROR: {e}")
        
        except Exception as e:
            print(f"- Failed to process {video_name}: {e}")
    
    # Summary
    print(f"\n" + "=" * 70)
    print("SUMMARY OF NEW SIMPLE DETECTOR")
    print("=" * 70)
    
    if all_results:
        total = len(all_results)
        on_track_count = sum(1 for r in all_results if r['on_track'])
        safe_count = total - on_track_count
        avg_confidence = sum(r['confidence'] for r in all_results) / total
        
        print(f"Total videos tested: {total}")
        print(f"Person on track detections: {on_track_count}")
        print(f"Safe detections: {safe_count}")
        print(f"Average confidence: {avg_confidence:.0%}")
        
        print(f"\nDETAILED RESULTS:")
        for r in all_results:
            status = "ON TRACK" if r['on_track'] else "SAFE"
            print(f"  {r['video']}: {status} - {r['people_count']} people ({r['confidence']:.0%})")
            print(f"    Scene: {r['scene'][:60]}...")
        
        # Assessment
        print(f"\n" + "=" * 70)
        print("ASSESSMENT")
        print("=" * 70)
        
        if safe_count > 0:
            print("+ SUCCESS: Detector now gives SAFE results!")
            print("+ No longer stuck on always detecting danger")
        else:
            print("- STILL PROBLEMATIC: Only danger detections")
        
        if avg_confidence > 60:
            print("+ Good confidence levels")
        else:
            print("- Low confidence, may need adjustment")
        
        print(f"\nThe new simple approach:")
        print("1. Uses only reliable Transformer model")
        print("2. Simple keyword counting (person + track words)")
        print("3. Conservative decision logic")
        print("4. Clear scene descriptions for verification")
    
    print(f"\nREADY TO TEST IN STREAMLIT:")
    print("Open http://localhost:8502")
    print("Select 'Person on Track Detector'")
    print("Upload test videos to see improved results")
    
    return all_results

if __name__ == "__main__":
    test_simple_detector()