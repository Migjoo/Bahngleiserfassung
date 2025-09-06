#!/usr/bin/env python3
"""
Test multiple models for simple yes/no person detection
"""
import sys
import os
from io import BytesIO
import requests
import base64
from PIL import Image

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_yes_no_models():
    """Test multiple models for yes/no person detection"""
    print("TESTING MULTIPLE MODELS FOR YES/NO PERSON DETECTION")
    print("=" * 60)
    
    try:
        from local_models import get_local_model_manager
        from app import extract_frames_from_video, process_image_locally, query_huggingface_api
        print("+ Components loaded successfully")
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
    
    # Extract 3 test frames
    try:
        with open(video_path, 'rb') as f:
            video_data = f.read()
        
        video_file = BytesIO(video_data)
        frames = extract_frames_from_video(video_file, fps=0.3)  # Every 3+ seconds
        
        if len(frames) < 3:
            print(f"- Only {len(frames)} frames extracted, need at least 3")
            return
        
        test_frames = frames[:3]  # Use first 3 frames
        print(f"+ Using {len(test_frames)} frames for testing")
        
    except Exception as e:
        print(f"- Frame extraction error: {e}")
        return
    
    # Initialize local models
    try:
        local_manager = get_local_model_manager()
        print("+ Local models ready")
    except Exception as e:
        print(f"- Local model error: {e}")
        return
    
    # Define models to test
    models_to_test = {
        "Local CNN (BLIP)": {
            "type": "local",
            "model_name": "CNN (BLIP)",
            "prompt": "Is there a person in this image? Answer only yes or no."
        },
        "Local Transformer": {
            "type": "local", 
            "model_name": "Transformer (ViT-GPT2)",
            "prompt": "Is there a person in this image? Answer only yes or no."
        },
        "Remote BLIP": {
            "type": "remote",
            "model_name": "Salesforce/blip-image-captioning-large",
            "prompt": "Is there a person in this image? Answer only yes or no."
        },
        "Remote GIT": {
            "type": "remote",
            "model_name": "microsoft/git-large-coco", 
            "prompt": "Is there a person in this image? Answer only yes or no."
        },
        "Remote ViT-GPT2": {
            "type": "remote",
            "model_name": "nlpconnect/vit-gpt2-image-captioning",
            "prompt": "Is there a person in this image? Answer only yes or no."
        }
    }
    
    # API token (you may need to update this)
    api_token = "os.getenv("HF_TOKEN")"
    
    # Results storage
    results = {}
    
    print(f"\nTesting {len(models_to_test)} models on {len(test_frames)} frames:")
    print("=" * 80)
    
    # Test each model
    for model_display_name, config in models_to_test.items():
        print(f"\nTesting: {model_display_name}")
        print("-" * 50)
        
        model_results = []
        
        for i, frame_data in enumerate(test_frames):
            frame_num = i + 1
            timestamp = frame_data['timestamp']
            
            try:
                if config["type"] == "local":
                    # Test local model
                    result = process_image_locally(
                        frame_data['frame'],
                        config["prompt"],
                        config["model_name"],
                        local_manager
                    )
                    
                    if 'error' in result:
                        response = f"ERROR: {result['error']}"
                        yes_no = "ERROR"
                    else:
                        response = result.get('generated_text', 'No response')
                        yes_no = extract_yes_no(response)
                
                else:
                    # Test remote model
                    result = query_huggingface_api(
                        frame_data['frame'],
                        config["prompt"],
                        config["model_name"],
                        api_token
                    )
                    
                    if 'error' in result:
                        response = f"ERROR: {result['error']}"
                        yes_no = "ERROR"
                    else:
                        # Handle different response formats
                        if isinstance(result, list) and len(result) > 0:
                            response = result[0].get('generated_text', str(result[0]))
                        elif 'generated_text' in result:
                            response = result['generated_text']
                        else:
                            response = str(result)
                        
                        yes_no = extract_yes_no(response)
                
                model_results.append({
                    'frame': frame_num,
                    'timestamp': timestamp,
                    'response': response[:100] + "..." if len(response) > 100 else response,
                    'yes_no': yes_no
                })
                
                print(f"  Frame {frame_num} ({timestamp:.1f}s): {yes_no} - {response[:50]}...")
                
            except Exception as e:
                model_results.append({
                    'frame': frame_num,
                    'timestamp': timestamp,
                    'response': f"Exception: {str(e)}",
                    'yes_no': "ERROR"
                })
                print(f"  Frame {frame_num} ({timestamp:.1f}s): ERROR - {str(e)}")
        
        results[model_display_name] = model_results
    
    # Create comparison table
    print(f"\n" + "=" * 80)
    print("RESULTS COMPARISON TABLE")
    print("=" * 80)
    
    # Header
    header = f"{'Frame':<8} {'Time':<8}"
    for model_name in models_to_test.keys():
        header += f" {model_name:<15}"
    print(header)
    print("-" * len(header))
    
    # Data rows
    for i in range(len(test_frames)):
        frame_num = i + 1
        timestamp = test_frames[i]['timestamp']
        
        row = f"{frame_num:<8} {timestamp:<8.1f}"
        for model_name in models_to_test.keys():
            yes_no = results[model_name][i]['yes_no']
            row += f" {yes_no:<15}"
        print(row)
    
    # Analysis and recommendation
    print(f"\n" + "=" * 80)
    print("ANALYSIS & RECOMMENDATION")
    print("=" * 80)
    
    # Count successful yes/no responses per model
    model_scores = {}
    for model_name, model_results in results.items():
        success_count = sum(1 for r in model_results if r['yes_no'] in ['YES', 'NO'])
        error_count = sum(1 for r in model_results if r['yes_no'] == 'ERROR')
        unclear_count = sum(1 for r in model_results if r['yes_no'] == 'UNCLEAR')
        
        model_scores[model_name] = {
            'success': success_count,
            'error': error_count,
            'unclear': unclear_count,
            'success_rate': success_count / len(model_results) * 100
        }
    
    print("\nModel Performance:")
    print(f"{'Model':<20} {'Success':<8} {'Errors':<8} {'Unclear':<8} {'Success Rate':<12}")
    print("-" * 70)
    
    for model_name, scores in model_scores.items():
        print(f"{model_name:<20} {scores['success']:<8} {scores['error']:<8} {scores['unclear']:<8} {scores['success_rate']:<12.1f}%")
    
    # Find best model
    best_model = max(model_scores.items(), key=lambda x: x[1]['success_rate'])
    print(f"\nðŸ† BEST MODEL: {best_model[0]}")
    print(f"   Success Rate: {best_model[1]['success_rate']:.1f}%")
    print(f"   Recommendation: Use this model for yes/no person detection")
    
    return results, best_model[0]

def extract_yes_no(response):
    """Extract yes/no from model response"""
    if not response:
        return "UNCLEAR"
    
    response_lower = response.lower().strip()
    
    # Direct yes/no detection
    if response_lower == "yes" or response_lower.startswith("yes"):
        return "YES"
    elif response_lower == "no" or response_lower.startswith("no"):
        return "NO"
    
    # Look for yes/no anywhere in response
    if "yes" in response_lower and "no" not in response_lower:
        return "YES"
    elif "no" in response_lower and "yes" not in response_lower:
        return "NO"
    
    # Check for person-related keywords as backup
    person_words = ['person', 'people', 'man', 'woman', 'boy', 'girl', 'human']
    if any(word in response_lower for word in person_words):
        return "YES"
    
    # If response contains negative words
    negative_words = ['not', 'none', 'empty', 'no one', 'nobody']
    if any(word in response_lower for word in negative_words):
        return "NO"
    
    return "UNCLEAR"

if __name__ == "__main__":
    test_yes_no_models()
