#!/usr/bin/env python3
"""
Main Streamlit application for video frame analysis with ontology-based risk assessment
Refactored for better code organization and maintainability
"""
import streamlit as st
import json
from dotenv import load_dotenv

# Import our modular components
from video_processing import extract_frames_from_video
from ontology_integration import analyze_scene_with_ontology, extract_scene_description
from model_processing import process_frame
from ui_components import (
    render_sidebar_config,
    render_input_section,
    render_prompt_section,
    render_process_button,
    render_results_header,
    render_frame_result,
    render_validation_errors,
    render_instructions
)

# Try to import local models, fall back gracefully if not available
try:
    from local_models import get_local_model_manager
    LOCAL_MODELS_AVAILABLE = True
except ImportError as e:
    LOCAL_MODELS_AVAILABLE = False
    print(f"Local models not available: {e}")
    def get_local_model_manager():
        return None

# Load environment variables
load_dotenv()


def load_settings():
    """Load settings from JSON file"""
    try:
        with open('settings.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


@st.cache_resource
def initialize_local_models():
    """Initialize local model manager"""
    return get_local_model_manager()


def initialize_app():
    """Initialize the Streamlit application"""
    st.set_page_config(
        page_title="Video Frame Analyzer with Ontology",
        page_icon="🎥",
        layout="wide"
    )
    
    st.title("🎥 Video Frame Analyzer with Ontology-Based Risk Assessment")
    st.markdown("Upload a video and analyze frames using AI models with ontology-based safety classification")


def setup_local_models():
    """Setup local models and return availability status"""
    local_manager = None
    local_models_available = False
    
    if LOCAL_MODELS_AVAILABLE:
        try:
            local_manager = initialize_local_models()
            local_models_available = True
            st.success("🤖 Local AI models initialized successfully!")
        except Exception as e:
            st.warning(f"Local AI models not available: {str(e)}")
            st.info("💡 Install AI packages: `pip install torch torchvision transformers accelerate sentencepiece`")
            local_models_available = False
    else:
        st.info("💡 Local AI models not installed. Install with: `pip install torch torchvision transformers accelerate sentencepiece`")
    
    return local_manager, local_models_available


def process_video_frames(video_file, config, local_manager=None):
    """
    Process all frames in the video and return results
    """
    # Extract frames
    frames = extract_frames_from_video(video_file, config["fps"])
    
    if not frames:
        st.error("No frames could be extracted from the video")
        return []
    
    st.success(f"Extracted {len(frames)} frames from video")
    
    # Process each frame
    results = []
    progress_bar = st.progress(0)
    
    # Add prompt to config for processing
    processing_config = config.copy()
    processing_config["prompt"] = config.get("prompt", "")
    
    for i, frame_data in enumerate(frames):
        with st.spinner(f"Analyzing frame {i+1}/{len(frames)}..."):
            # Process frame with selected model
            result = process_frame(frame_data, processing_config, local_manager)
            
            # Extract scene description for ontology analysis
            scene_description = extract_scene_description(result)
            
            # Apply ontology analysis
            ontology_analysis = analyze_scene_with_ontology(scene_description, config["use_ontology"])
            
            results.append({
                'frame_number': frame_data['frame_number'],
                'timestamp': frame_data['timestamp'],
                'image': frame_data['frame'],
                'result': result,
                'ontology_analysis': ontology_analysis
            })
            
            progress_bar.progress((i + 1) / len(frames))
    
    return results


def validate_inputs(video_file, prompt, config, local_models_available):
    """
    Validate all required inputs
    """
    model_type = config["model_type"]
    selected_model = config["selected_model"]
    api_token = config["api_token"]
    
    # Check basic requirements
    if not video_file:
        return False
    
    # Check prompt requirements
    if not prompt and not (model_type == "Local Models" and selected_model == "Person on Track Detector"):
        return False
    
    # Check API token for remote models
    if not api_token and model_type == "Remote API":
        return False
    
    # Check local models availability
    if model_type == "Local Models" and not local_models_available:
        return False
    
    return True


def main():
    """Main application entry point"""
    # Initialize application
    initialize_app()
    
    # Load settings and setup models
    settings = load_settings()
    local_manager, local_models_available = setup_local_models()
    
    # Create main layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Render sidebar configuration
        config = render_sidebar_config(settings, local_models_available, local_manager)
        
        # Render input section
        input_data = render_input_section()
        video_file = input_data["video_file"]
        
        # Render prompt section
        prompt = render_prompt_section(config)
        
        # Render process button
        process_button = render_process_button()
    
    with col2:
        # Render results section
        results_container = render_results_header()
    
    # Main processing logic
    if process_button:
        if validate_inputs(video_file, prompt, config, local_models_available):
            # Add prompt to config for processing
            config["prompt"] = prompt
            
            with st.spinner("Processing video..."):
                # Process video frames
                results = process_video_frames(video_file, config, local_manager)
                
                # Display results
                if results:
                    with results_container:
                        st.subheader("Analysis Results")
                        
                        # Display summary statistics
                        severity_counts = {}
                        for result in results:
                            severity = result['ontology_analysis'].get('severity', 'NONE')
                            severity_counts[severity] = severity_counts.get(severity, 0) + 1
                        
                        if config["use_ontology"] and severity_counts:
                            st.write("**Summary:**")
                            summary_cols = st.columns(len(severity_counts))
                            for i, (severity, count) in enumerate(severity_counts.items()):
                                icon_map = {
                                    'NONE': '✅', 'LOW': '🟢', 'MEDIUM': '🟠', 
                                    'HIGH': '⚠️', 'CRITICAL': '🚨'
                                }
                                with summary_cols[i]:
                                    st.metric(f"{icon_map.get(severity, '❓')} {severity}", count)
                            st.divider()
                        
                        # Display individual frame results
                        for result_data in results:
                            render_frame_result(result_data)
        else:
            # Show validation errors
            render_validation_errors(
                video_file, prompt, config["api_token"], 
                config["model_type"], local_models_available, config["selected_model"]
            )
    
    # Render instructions
    render_instructions()


if __name__ == "__main__":
    main()