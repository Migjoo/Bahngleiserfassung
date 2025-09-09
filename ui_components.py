#!/usr/bin/env python3
"""
UI components for the Streamlit application
"""
import streamlit as st
from typing import Dict, List, Any, Optional
from local_models import get_local_model_manager


# Available Hugging Face models for remote API
AVAILABLE_MODELS = {
    "microsoft/kosmos-2-patch14-224": "Kosmos-2",
    "Salesforce/blip-image-captioning-large": "BLIP Image Captioning",
    "microsoft/DialoGPT-medium": "DialoGPT",
    "microsoft/git-large-coco": "GIT Large COCO",
    "nlpconnect/vit-gpt2-image-captioning": "ViT-GPT2"
}


def render_sidebar_config(settings: Dict, local_models_available: bool, local_manager: Optional[Any]) -> Dict[str, Any]:
    """
    Render the sidebar configuration panel
    Returns configuration settings
    """
    with st.sidebar:
        st.header("Configuration")
        
        # Model type selection
        available_options = []
        if local_models_available:
            available_options.append("Local Models")
        available_options.append("Remote API")
        
        model_type = st.radio(
            "Model Type",
            available_options,
            help="Choose between local AI models or remote Hugging Face API"
        )
        
        # Model selection based on type
        if model_type == "Local Models" and local_models_available:
            selected_model, api_token = _render_local_model_config(local_manager)
        else:
            selected_model, api_token = _render_remote_model_config(settings)
        
        # Frame extraction rate
        fps = st.slider(
            "Frames per second to extract",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1
        )
        
        # Ontology settings
        st.subheader("Ontology Analysis")
        use_ontology = st.checkbox(
            "Enable Ontology Analysis",
            value=True,
            help="Use ontology-based classification (NONE/LOW/MEDIUM/HIGH/CRITICAL)"
        )
        
        if not use_ontology:
            st.info("🔄 Ontology analysis disabled - showing raw model output only")
    
    return {
        "model_type": model_type,
        "selected_model": selected_model,
        "api_token": api_token,
        "fps": fps,
        "use_ontology": use_ontology
    }


def _render_local_model_config(local_manager) -> tuple:
    """Render local model configuration"""
    available_local_models = local_manager.get_available_models()
    selected_model = st.selectbox(
        "Select Local Model",
        options=available_local_models,
        help="Choose between CNN (fast) or Transformer (detailed) models"
    )
    
    # Show model info
    model_info = local_manager.get_model_info()
    if selected_model in model_info:
        with st.expander("Model Information"):
            st.write(f"**Description:** {model_info[selected_model]['description']}")
            st.write(f"**Strengths:** {model_info[selected_model]['strengths']}")
            st.write(f"**Size:** {model_info[selected_model]['size']}")
    
    return selected_model, None  # No API token needed for local models


def _render_remote_model_config(settings: Dict) -> tuple:
    """Render remote API model configuration"""
    default_token = settings.get('hugging_face_api_token', '')
    api_token = st.text_input(
        "Hugging Face API Token",
        value=default_token,
        type="password",
        help="Get your token from https://huggingface.co/settings/tokens or save in settings.json"
    )
    
    selected_model = st.selectbox(
        "Select Model",
        options=list(AVAILABLE_MODELS.keys()),
        format_func=lambda x: AVAILABLE_MODELS[x]
    )
    
    return selected_model, api_token


def render_input_section() -> Dict[str, Any]:
    """
    Render the input section for video upload and prompts
    Returns input data
    """
    st.header("Input")
    
    # Video upload
    video_file = st.file_uploader(
        "Upload Video",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file to analyze"
    )
    
    return {
        "video_file": video_file
    }


def render_prompt_section(config: Dict[str, Any]) -> str:
    """
    Render prompt input section based on model configuration
    """
    model_type = config["model_type"]
    selected_model = config["selected_model"]
    
    # Prompt input (conditional based on model)
    if (model_type == "Local Models" and 
        selected_model == "Person on Track Detector"):
        # Person on Track Detector works automatically
        st.info("🤖 Person on Track Detector works automatically - no prompt needed!")
        return "automatic"
    else:
        # Regular models need user prompt
        return st.text_area(
            "Analysis Prompt",
            placeholder="Describe what you see in the image...",
            help="Enter the prompt to analyze each frame"
        )


def render_process_button() -> bool:
    """Render the process button"""
    return st.button("Process Video", type="primary")


def render_results_header():
    """Render the results section header"""
    st.header("Results")
    return st.container()


def render_frame_result(result_data: Dict[str, Any]):
    """
    Render a single frame result with ontology analysis
    """
    ontology = result_data['ontology_analysis']
    severity_icon = ontology.get('severity_icon', '✅')
    severity = ontology.get('severity', 'NONE')
    
    # Create expander title with severity indicator
    expander_title = f"{severity_icon} {severity} - Frame {result_data['frame_number']} (t={result_data['timestamp']:.1f}s)"
    
    with st.expander(expander_title):
        col_img, col_text = st.columns([1, 2])
        
        with col_img:
            st.image(
                result_data['image'],
                caption=f"Frame {result_data['frame_number']}",
                use_container_width=True
            )
        
        with col_text:
            # Display ontology analysis first if enabled
            if ontology.get('ontology_used', False):
                _render_ontology_analysis(ontology)
                st.divider()
            
            # Display original model results
            _render_model_output(result_data['result'])


def _render_ontology_analysis(ontology: Dict[str, Any]):
    """Render ontology analysis section"""
    severity = ontology.get('severity', 'NONE')
    severity_icon = ontology.get('severity_icon', '✅')
    severity_color = ontology.get('severity_color', 'green')
    
    # Severity display with color
    st.markdown(f"**Safety Assessment:** :{severity_color}[{severity_icon} {severity}]")
    
    # Score display
    if ontology.get('score', 0) > 0:
        st.metric("Risk Score", f"{ontology['score']}/100")
    
    # Show explanations if available
    if ontology.get('explanations'):
        st.write("**Ontology Analysis:**")
        for explanation in ontology['explanations']:
            st.write(f"• {explanation}")
    
    # Show fired rules if available
    if ontology.get('fired_rules'):
        with st.expander("Technical Details"):
            st.write("**Triggered Rules:**")
            for rule in ontology['fired_rules']:
                st.code(rule)
            
            if ontology.get('labels'):
                st.write("**Detected Hazard Labels:**")
                for label in ontology['labels']:
                    st.code(label)


def _render_model_output(result: Dict[str, Any]):
    """Render original model output section"""
    st.write("**Model Output:**")
    
    if 'error' in result:
        st.error(f"Error: {result['error']}")
    elif 'person_on_track_detection' in result:
        _render_person_detection_result(result['person_on_track_detection'])
    else:
        _render_general_model_result(result)


def _render_person_detection_result(detection: Dict[str, Any]):
    """Render person on track detection specific results"""
    people_count = detection.get('people_count', 0)
    confidence = detection.get('confidence', 0)
    analysis = detection.get('analysis', 'No analysis')
    
    st.write(f"**Detection Analysis:** {analysis}")
    
    # Show metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("👥 People Detected", people_count)
    with col2:
        st.metric("📊 Model Confidence", f"{confidence:.0%}")


def _render_general_model_result(result: Dict[str, Any]):
    """Render general model results (captioning, etc.)"""
    if 'generated_text' in result:
        st.write(f"*{result['generated_text']}*")
    elif isinstance(result, list) and len(result) > 0:
        if 'generated_text' in result[0]:
            st.write(f"*{result[0]['generated_text']}*")
        else:
            st.json(result[0])
    else:
        st.json(result)


def render_validation_errors(video_file, prompt, api_token, model_type, local_models_available, selected_model):
    """
    Render validation error messages
    """
    if not video_file:
        st.error("Please upload a video file")
    if not prompt and not (model_type == "Local Models" and selected_model == "Person on Track Detector"):
        st.error("Please enter an analysis prompt")
    if not api_token and model_type == "Remote API":
        st.error("Please provide your Hugging Face API token for remote models")
    if model_type == "Local Models" and not local_models_available:
        st.error("Local models failed to initialize. Check your installation.")


def render_instructions():
    """Render the instructions section"""
    with st.expander("How to use"):
        st.markdown("""
        ## Local AI Models (Recommended)
        1. **Upload a video**: Choose a video file (MP4, AVI, MOV, or MKV)
        2. **Select model type**: Choose "Local Models" for offline processing
        3. **Choose AI model**: 
           - **CNN (BLIP)**: Fast, good for object detection (~1.2GB)
           - **Transformer (ViT-GPT2)**: Detailed descriptions (~1.8GB)
        4. **Enter a prompt**: Describe what you want the AI to analyze
        5. **Enable/Disable Ontology**: Toggle ontology-based risk assessment
        6. **Adjust frame rate**: Set frames per second to extract (default: 1 fps)
        7. **Click Process**: Frames are processed locally on your machine
        
        ## Ontology Analysis
        - **✅ NONE**: No safety concerns detected
        - **🟢 LOW**: Minor safety considerations
        - **🟠 MEDIUM**: Moderate safety risk
        - **⚠️ HIGH**: Significant safety risk
        - **🚨 CRITICAL**: Immediate safety hazard
        
        ## Remote API Models (Optional)
        1. **Get API token**: Visit [Hugging Face Settings](https://huggingface.co/settings/tokens)
        2. **Select "Remote API"** in model type
        3. **Enter token** and select remote model
        
        ## Video Support Features
        - **Automatic corruption repair**: Handles videos with corrupted moov atoms
        - **FFmpeg integration**: Auto-repairs problematic video files
        - **Multiple formats**: MP4, AVI, MOV, MKV support
        
        ## Requirements
        - **Python packages**: torch, transformers, accelerate (see requirements.txt)
        - **Optional**: FFmpeg for video repair (download from https://ffmpeg.org)
        - **Storage**: ~3GB for both local models
        """)