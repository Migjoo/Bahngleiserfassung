#!/usr/bin/env python3
"""
Compare CNN and Transformer models on video frames with table results
"""
import sys
import os
import time
from io import BytesIO
import pandas as pd
from tabulate import tabulate as tabulate_func

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def compare_ai_models_on_video():
    """Compare both AI models on all video frames"""
    print("AI Models Comparison Test")
    print("=" * 50)
    
    # Test imports
    try:
        from app import extract_frames_from_video, process_image_locally
        from local_models import get_local_model_manager
        print("+ Successfully imported components")
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
    
    # Initialize models
    print("+ Initializing AI models...")
    try:
        local_manager = get_local_model_manager()
        available_models = local_manager.get_available_models()
        print(f"+ Available models: {available_models}")
    except Exception as e:
        print(f"- Model initialization error: {e}")
        return
    
    # Extract frames
    print("+ Extracting video frames...")
    try:
        with open(video_path, 'rb') as f:
            video_data = f.read()
        
        video_file = BytesIO(video_data)
        frames = extract_frames_from_video(video_file, fps=0.5)  # 1 frame every 2 seconds
        
        if not frames:
            print("- No frames extracted")
            return
        
        print(f"+ Extracted {len(frames)} frames")
        
    except Exception as e:
        print(f"- Frame extraction error: {e}")
        return
    
    # Test prompt
    test_prompt = "Describe what you see in this image"
    
    # Prepare results storage
    results_data = []
    
    print(f"\n+ Processing {len(frames)} frames with both models...")
    print("+ This may take a few minutes for model downloads and processing...")
    
    # Process each frame with both models
    for i, frame_data in enumerate(frames):
        frame_num = i + 1
        timestamp = frame_data['timestamp']
        
        print(f"\nProcessing Frame {frame_num}/{len(frames)} (t={timestamp:.1f}s)")
        print("-" * 40)
        
        frame_result = {
            'Frame': frame_num,
            'Timestamp': f"{timestamp:.1f}s",
            'CNN_Result': 'Error',
            'CNN_Time': 0,
            'Transformer_Result': 'Error', 
            'Transformer_Time': 0
        }
        
        # Test CNN (BLIP) Model
        print("  Testing CNN (BLIP)...")
        try:
            start_time = time.time()
            result = process_image_locally(
                frame_data['frame'],
                test_prompt,
                'CNN (BLIP)',
                local_manager
            )
            processing_time = time.time() - start_time
            
            if 'error' in result:
                frame_result['CNN_Result'] = f"Error: {result['error']}"
            else:
                caption = result.get('generated_text', 'No caption')
                frame_result['CNN_Result'] = caption
                frame_result['CNN_Time'] = processing_time
                print(f"    + Success ({processing_time:.1f}s): {caption[:50]}...")
                
        except Exception as e:
            print(f"    - Exception: {e}")
            frame_result['CNN_Result'] = f"Exception: {str(e)}"
        
        # Test Transformer (ViT-GPT2) Model
        print("  Testing Transformer (ViT-GPT2)...")
        try:
            start_time = time.time()
            result = process_image_locally(
                frame_data['frame'],
                test_prompt,
                'Transformer (ViT-GPT2)',
                local_manager
            )
            processing_time = time.time() - start_time
            
            if 'error' in result:
                frame_result['Transformer_Result'] = f"Error: {result['error']}"
            else:
                caption = result.get('generated_text', 'No caption')
                frame_result['Transformer_Result'] = caption
                frame_result['Transformer_Time'] = processing_time
                print(f"    + Success ({processing_time:.1f}s): {caption[:50]}...")
                
        except Exception as e:
            print(f"    - Exception: {e}")
            frame_result['Transformer_Result'] = f"Exception: {str(e)}"
        
        results_data.append(frame_result)
    
    # Create results table
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS TABLE")
    print("=" * 80)
    
    # Create DataFrame for better table formatting
    df = pd.DataFrame(results_data)
    
    # Display full table
    print("\nDetailed Results:")
    print(tabulate_func(df, headers='keys', tablefmt='grid', showindex=False))
    
    # Create summary statistics
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    
    # Count successes
    cnn_successes = sum(1 for r in results_data if not r['CNN_Result'].startswith(('Error', 'Exception')))
    transformer_successes = sum(1 for r in results_data if not r['Transformer_Result'].startswith(('Error', 'Exception')))
    
    # Calculate average times (only for successful runs)
    cnn_times = [r['CNN_Time'] for r in results_data if r['CNN_Time'] > 0]
    transformer_times = [r['Transformer_Time'] for r in results_data if r['Transformer_Time'] > 0]
    
    cnn_avg_time = sum(cnn_times) / len(cnn_times) if cnn_times else 0
    transformer_avg_time = sum(transformer_times) / len(transformer_times) if transformer_times else 0
    
    # Summary table
    summary_data = [
        ['Model', 'Success Rate', 'Avg Time (s)', 'Total Frames'],
        ['CNN (BLIP)', f"{cnn_successes}/{len(frames)} ({100*cnn_successes/len(frames):.1f}%)", f"{cnn_avg_time:.1f}", len(frames)],
        ['Transformer (ViT-GPT2)', f"{transformer_successes}/{len(frames)} ({100*transformer_successes/len(frames):.1f}%)", f"{transformer_avg_time:.1f}", len(frames)]
    ]
    
    print(tabulate_func(summary_data[1:], headers=summary_data[0], tablefmt='grid'))
    
    # Model comparison insights
    print("\n" + "=" * 50)
    print("MODEL COMPARISON INSIGHTS")
    print("=" * 50)
    
    if cnn_successes > 0 and transformer_successes > 0:
        if cnn_avg_time < transformer_avg_time:
            print(f"+ CNN (BLIP) is faster: {cnn_avg_time:.1f}s vs {transformer_avg_time:.1f}s avg")
        else:
            print(f"+ Transformer (ViT-GPT2) is faster: {transformer_avg_time:.1f}s vs {cnn_avg_time:.1f}s avg")
        
        print(f"+ CNN success rate: {100*cnn_successes/len(frames):.1f}%")
        print(f"+ Transformer success rate: {100*transformer_successes/len(frames):.1f}%")
        
        # Sample comparison for first successful frame
        for r in results_data:
            if not r['CNN_Result'].startswith(('Error', 'Exception')) and not r['Transformer_Result'].startswith(('Error', 'Exception')):
                print(f"\nSample Comparison (Frame {r['Frame']}):")
                print(f"  CNN: {r['CNN_Result']}")
                print(f"  Transformer: {r['Transformer_Result']}")
                break
    
    # Save results to CSV
    csv_filename = 'ai_models_comparison_results.csv'
    df.to_csv(csv_filename, index=False)
    print(f"\n+ Results saved to: {csv_filename}")
    
    print(f"\n+ Comparison complete! Processed {len(frames)} frames with both models")

if __name__ == "__main__":
    try:
        import pandas as pd
        from tabulate import tabulate as tabulate_func
    except ImportError:
        print("Installing required packages for table formatting...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas', 'tabulate'])
        import pandas as pd
        from tabulate import tabulate as tabulate_func
    
    compare_ai_models_on_video()