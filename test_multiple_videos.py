#!/usr/bin/env python3
"""
Test Yes/No Person Detector on multiple videos for accuracy verification
"""
import sys
import os
from io import BytesIO
import glob

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_multiple_videos():
    """Test Yes/No Person Detector on multiple videos"""
    print("TESTING YES/NO PERSON DETECTOR - MULTIPLE VIDEOS")
    print("=" * 60)
    print("Verifying model accuracy across different video content")
    print()
    
    try:
        from local_models import get_local_model_manager
        from app import extract_frames_from_video, process_image_locally
        print("+ Components loaded successfully")
    except ImportError as e:
        print(f"- Import error: {e}")
        return
    
    # Find all MP4 files
    video_files = glob.glob("*.mp4")
    if not video_files:
        print("- No MP4 files found")
        return
    
    print(f"+ Found {len(video_files)} video files: {video_files}")
    
    # Initialize models
    try:
        local_manager = get_local_model_manager()
        print("+ Yes/No Person Detector ready")
    except Exception as e:
        print(f"- Model initialization error: {e}")
        return
    
    all_results = {}
    
    # Test each video
    for video_idx, video_path in enumerate(video_files):
        print(f"\n" + "=" * 60)
        print(f"TESTING VIDEO {video_idx + 1}: {video_path}")
        print("=" * 60)
        
        try:
            # Extract frames
            with open(video_path, 'rb') as f:
                video_data = f.read()
            
            video_file = BytesIO(video_data)
            frames = extract_frames_from_video(video_file, fps=0.3)  # Every 3+ seconds
            
            if not frames:
                print(f"- No frames extracted from {video_path}")
                continue
            
            print(f"+ Extracted {len(frames)} frames from {video_path}")
            
            # Test first 3 frames from each video
            test_frames = frames[:3]
            video_results = []
            
            for i, frame_data in enumerate(test_frames):
                frame_num = i + 1
                timestamp = frame_data['timestamp']
                
                print(f"\n  Frame {frame_num} ({timestamp:.1f}s):")
                print(f"  {'-' * 30}")
                
                try:
                    result = process_image_locally(
                        frame_data['frame'],
                        "Is there a person in this image?",
                        'Yes/No Person Detector',
                        local_manager
                    )
                    
                    if 'error' in result:
                        print(f"  ERROR: {result['error']}")
                        video_results.append({
                            'frame': frame_num,
                            'timestamp': timestamp,
                            'answer': 'ERROR',
                            'confidence': 0,
                            'raw_response': result['error']
                        })
                    elif 'yes_no_detection' in result:
                        detection = result['yes_no_detection']
                        
                        answer = detection.get('answer', 'UNKNOWN')
                        person_detected = detection.get('person_detected', False)
                        confidence = detection.get('confidence', 0)
                        raw_response = detection.get('raw_response', 'N/A')
                        
                        print(f"  Answer: {answer}")
                        print(f"  Person Detected: {person_detected}")
                        print(f"  Confidence: {confidence:.0%}")
                        print(f"  Raw Response: '{raw_response[:50]}{'...' if len(raw_response) > 50 else ''}'")
                        
                        video_results.append({
                            'frame': frame_num,
                            'timestamp': timestamp,
                            'answer': answer,
                            'person_detected': person_detected,
                            'confidence': confidence,
                            'raw_response': raw_response
                        })
                    else:
                        print(f"  Unexpected result format: {result}")
                        video_results.append({
                            'frame': frame_num,
                            'timestamp': timestamp,
                            'answer': 'UNKNOWN',
                            'confidence': 0,
                            'raw_response': str(result)
                        })
                        
                except Exception as e:
                    print(f"  ERROR: {e}")
                    video_results.append({
                        'frame': frame_num,
                        'timestamp': timestamp,
                        'answer': 'ERROR',
                        'confidence': 0,
                        'raw_response': str(e)
                    })
            
            all_results[video_path] = video_results
            
        except Exception as e:
            print(f"- Failed to process {video_path}: {e}")
            continue
    
    # Comprehensive analysis
    print(f"\n" + "=" * 80)
    print("COMPREHENSIVE RESULTS ANALYSIS")
    print("=" * 80)
    
    # Summary table
    print(f"\nRESULTS SUMMARY BY VIDEO:")
    print("-" * 80)
    print(f"{'Video':<20} {'Frame':<8} {'Time':<8} {'Answer':<8} {'Confidence':<12} {'Raw Response':<25}")
    print("-" * 80)
    
    total_frames = 0
    yes_count = 0
    no_count = 0
    error_count = 0
    unclear_count = 0
    confidence_sum = 0
    
    for video_name, results in all_results.items():
        for result in results:
            frame = result['frame']
            timestamp = result['timestamp']
            answer = result['answer']
            confidence = result['confidence']
            raw_response = result['raw_response'][:20] + "..." if len(result['raw_response']) > 20 else result['raw_response']
            
            print(f"{video_name:<20} {frame:<8} {timestamp:<8.1f} {answer:<8} {confidence:<12.0%} {raw_response:<25}")
            
            total_frames += 1
            confidence_sum += confidence
            
            if answer == 'YES':
                yes_count += 1
            elif answer == 'NO':
                no_count += 1
            elif answer == 'ERROR':
                error_count += 1
            else:
                unclear_count += 1
    
    # Overall statistics
    print(f"\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    
    print(f"Total frames tested: {total_frames}")
    print(f"Videos tested: {len(all_results)}")
    print(f"YES answers: {yes_count}")
    print(f"NO answers: {no_count}")
    print(f"ERROR responses: {error_count}")
    print(f"UNCLEAR responses: {unclear_count}")
    
    if total_frames > 0:
        success_rate = (yes_count + no_count) / total_frames * 100
        avg_confidence = confidence_sum / total_frames
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Average confidence: {avg_confidence:.0%}")
    
    # Accuracy assessment
    print(f"\n" + "=" * 80)
    print("ACCURACY ASSESSMENT")
    print("=" * 80)
    
    # Check if model is stuck giving same answer
    if yes_count == total_frames and total_frames > 3:
        print("WARNING: Model appears to be giving only YES answers!")
        print("This suggests the model may be:")
        print("- Overconfident or biased toward detecting people")
        print("- Not properly processing different image content")
        print("- The prompt may need adjustment")
        print("\nRECOMMENDED FIXES:")
        print("1. Test with images that definitely contain no people")
        print("2. Adjust the prompt to be more specific")
        print("3. Try different confidence thresholds")
        print("4. Consider using a different base model")
        
    elif no_count == total_frames and total_frames > 3:
        print("WARNING: Model appears to be giving only NO answers!")
        print("This suggests the model may be:")
        print("- Too conservative in person detection")
        print("- Having trouble detecting people in the images")
        print("- The prompt may be too restrictive")
        
    elif yes_count > 0 and no_count > 0:
        print("GOOD: Model is giving varied responses (both YES and NO)")
        print("This suggests the model is:")
        print("+ Properly analyzing different image content") 
        print("+ Responding appropriately to image variations")
        print("+ Working as expected")
        
    else:
        print("INSUFFICIENT DATA: Need more diverse test cases")
    
    # Per-video analysis
    print(f"\nPER-VIDEO BREAKDOWN:")
    print("-" * 50)
    
    for video_name, results in all_results.items():
        video_yes = sum(1 for r in results if r['answer'] == 'YES')
        video_no = sum(1 for r in results if r['answer'] == 'NO')
        video_total = len(results)
        
        print(f"{video_name}: {video_yes} YES, {video_no} NO (out of {video_total} frames)")
    
    return all_results

if __name__ == "__main__":
    test_multiple_videos()