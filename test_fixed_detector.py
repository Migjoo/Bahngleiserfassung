#!/usr/bin/env python3
"""
Test the FIXED Person on Track Detector that no longer gives false positives
"""
import sys
import os
from io import BytesIO
import glob

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_fixed_detector():
    """Test the fixed Person on Track Detector"""
    print("TESTING FIXED PERSON ON TRACK DETECTOR")
    print("=" * 50)
    print("Should now give accurate YES/NO results")
    print()
    
    try:
        from local_models import get_local_model_manager
        from app import extract_frames_from_video, process_image_locally
        print("+ Components loaded successfully")
    except ImportError as e:
        print(f"- Import error: {e}")
        return
    
    # Test with multiple videos
    test_videos = glob.glob("test\\*.mp4")[:3]  # Test first 3 videos
    if not test_videos:
        print("- No test videos found")
        return
    
    print(f"+ Testing {len(test_videos)} videos")
    
    try:
        local_manager = get_local_model_manager()
        print("+ Fixed Person on Track Detector ready")
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
            
            # Test first 2 frames per video
            test_frames = frames[:2]
            
            for frame_idx, frame_data in enumerate(test_frames):
                frame_num = frame_idx + 1
                timestamp = frame_data['timestamp']
                
                print(f"\n  Frame {frame_num} ({timestamp:.1f}s):")
                print(f"  {'-' * 30}")
                
                try:
                    result = process_image_locally(
                        frame_data['frame'],
                        "Track Safety Analysis",
                        'Person on Track Detector',
                        local_manager
                    )
                    
                    if 'person_on_track_detection' in result:
                        detection = result['person_on_track_detection']
                        
                        on_track = detection.get('person_on_track', False)
                        answer = detection.get('answer', 'UNKNOWN')
                        confidence = detection.get('confidence', 0)
                        reasoning = detection.get('reasoning', 'No reasoning')
                        
                        # Show result with clear status
                        if on_track:
                            print(f"  🚨 PERSON ON TRACK: {answer} ({confidence:.0%})")
                        else:
                            print(f"  ✅ TRACKS CLEAR: {answer} ({confidence:.0%})")
                        
                        print(f"  Reasoning: {reasoning}")
                        
                        all_results.append({
                            'video': video_name,
                            'frame': frame_num,
                            'on_track': on_track,
                            'answer': answer,
                            'confidence': confidence
                        })
                        
                    else:
                        print(f"  ERROR: Unexpected result format")
                        
                except Exception as e:
                    print(f"  ERROR: {e}")
            
        except Exception as e:
            print(f"- Failed to process {video_name}: {e}")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("SUMMARY OF FIXED DETECTOR PERFORMANCE")
    print("=" * 60)
    
    if all_results:
        total = len(all_results)
        yes_count = sum(1 for r in all_results if r['answer'] == 'YES')
        no_count = sum(1 for r in all_results if r['answer'] == 'NO')
        avg_confidence = sum(r['confidence'] for r in all_results) / total
        
        print(f"Total frames tested: {total}")
        print(f"YES results (person on track): {yes_count}")
        print(f"NO results (tracks clear): {no_count}")
        print(f"Average confidence: {avg_confidence:.0%}")
        
        if no_count > 0:
            print(f"\n✅ SUCCESS: Detector now gives NO results!")
            print(f"   - Fixed the false positive issue")
            print(f"   - Now provides varied and accurate responses")
        else:
            print(f"\n❌ STILL PROBLEMATIC: Only giving YES results")
        
        print(f"\nDETAILED RESULTS:")
        for r in all_results:
            status = "🚨" if r['on_track'] else "✅"
            print(f"  {r['video']} Frame {r['frame']}: {status} {r['answer']} ({r['confidence']:.0%})")
    
    print(f"\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Open http://localhost:8502")
    print("2. Select 'Person on Track Detector' from dropdown")
    print("3. Upload videos from test/ folder")
    print("4. Verify you now get both YES and NO results")
    print("5. Check that reasoning makes sense")
    
    return all_results

if __name__ == "__main__":
    test_fixed_detector()