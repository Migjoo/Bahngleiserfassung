import streamlit as st
import cv2
import os
import tempfile
import requests
import base64
import subprocess
import json
from io import BytesIO
from PIL import Image
import numpy as np
from dotenv import load_dotenv
from ontology_eval import Observation, evaluate, OntologyContext, decision_to_triples, triples_to_turtle, Severity
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

# Local models configuration
LOCAL_MODELS_ENABLED = LOCAL_MODELS_AVAILABLE
REMOTE_MODELS_ENABLED = True  # Always allow remote API as fallback

# Initialize local model manager
@st.cache_resource
def initialize_local_models():
    """Initialize local model manager"""
    return get_local_model_manager()

# Hugging Face models for vision-language tasks (kept for compatibility)
AVAILABLE_MODELS = {
    "microsoft/kosmos-2-patch14-224": "Kosmos-2",
    "Salesforce/blip-image-captioning-large": "BLIP Image Captioning",
    "microsoft/DialoGPT-medium": "DialoGPT",
    "microsoft/git-large-coco": "GIT Large COCO",
    "nlpconnect/vit-gpt2-image-captioning": "ViT-GPT2"
}

def repair_video_with_ffmpeg(input_path, output_path):
    """
    Repair corrupted video by moving moov atom to the beginning
    """
    try:
        # Try to fix the video using FFmpeg
        cmd = [
            'ffmpeg', 
            '-i', input_path,
            '-c', 'copy',
            '-movflags', 'faststart',
            '-avoid_negative_ts', 'make_zero',
            '-y',  # Overwrite output file
            output_path
        ]
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=300  # 5 minute timeout
        )
        
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def extract_frames_from_video(video_file, fps=1):
    """
    Extract frames from video at specified FPS (default 1 frame per second)
    Automatically handles corrupted videos by attempting repair with FFmpeg
    """
    frames = []
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        tmp_file_path = tmp_file.name
    
    repaired_path = None
    
    try:
        # First attempt: try to open video directly
        cap = cv2.VideoCapture(tmp_file_path)
        
        # Check if video opened successfully and has frames
        if not cap.isOpened() or cap.get(cv2.CAP_PROP_FRAME_COUNT) == 0:
            cap.release()
            
            # Second attempt: try to repair the video with FFmpeg
            st.warning("Video appears corrupted (moov atom issue). Attempting repair...")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='_repaired.mp4') as repaired_file:
                repaired_path = repaired_file.name
            
            if repair_video_with_ffmpeg(tmp_file_path, repaired_path):
                st.success("Video repair successful! Processing frames...")
                cap = cv2.VideoCapture(repaired_path)
            else:
                st.error("Failed to repair video. FFmpeg may not be installed or video is severely corrupted.")
                return frames
        
        # Extract video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30  # Default fallback FPS
            
        frame_interval = int(video_fps / fps) if video_fps > fps else 1
        
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append({
                    'frame': pil_image,
                    'timestamp': frame_count / video_fps,
                    'frame_number': extracted_count
                })
                extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        
    finally:
        # Clean up temporary files
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        if repaired_path and os.path.exists(repaired_path):
            os.unlink(repaired_path)
    
    return frames

def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def process_image_locally(image, prompt, model_name, local_manager):
    """
    Process image using local models
    """
    try:
        if model_name == "Person on Track Detector":
            # Special handling for person-on-track detection
            result = local_manager.person_on_track_detector.detect_person_on_track(image)
            return {"person_on_track_detection": result}
        else:
            caption = local_manager.generate_caption(model_name, image, prompt)
            return {"generated_text": caption}
    except Exception as e:
        return {"error": f"Local processing failed: {str(e)}"}

def query_huggingface_api(image, prompt, model_name, api_token):
    """
    Query Hugging Face API with image and prompt
    """
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {api_token}"}
    
    # Convert image to base64
    img_base64 = image_to_base64(image)
    
    # Prepare payload based on model type
    if "blip" in model_name.lower():
        # For BLIP models, send image directly
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        response = requests.post(
            API_URL,
            headers=headers,
            files={"file": buffer.getvalue()}
        )
    else:
        # For other vision-language models
        payload = {
            "inputs": {
                "image": img_base64,
                "text": prompt
            }
        }
        response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"API request failed: {response.status_code} - {response.text}"}

def analyze_scene_with_ontology(scene_description, use_ontology=True):
    """
    Analyze scene description using ontology-based evaluation
    Returns classification and explanation
    """
    if not use_ontology:
        return {
            "severity": "NONE",
            "severity_icon": "✅",
            "score": 0,
            "explanation": "Ontology-based analysis skipped",
            "ontology_used": False,
            "raw_description": scene_description
        }
    
    # Extract relevant information from scene description for ontology
    scene_lower = scene_description.lower().strip() if scene_description else ""
    
    # Initialize observation based on scene analysis
    obs = Observation()
    
    # Analyze scene for ontology features
    person_words = ['person', 'people', 'man', 'woman', 'boy', 'girl', 'human', 'individual', 'someone']
    track_words = ['track', 'tracks', 'rail', 'rails', 'railway', 'railroad']
    platform_words = ['platform', 'station', 'bahnsteig']
    danger_words = ['fallen', 'lying', 'down', 'accident', 'emergency']
    fire_words = ['fire', 'smoke', 'flames', 'burning']
    crowd_words = ['crowd', 'many people', 'group', 'mehrere personen']
    safe_words = ['no people', 'empty', 'clear', 'safe', 'nobody', 'without people']
    
    # Set observation values based on keyword analysis
    person_mentions = sum(1 for word in person_words if word in scene_lower)
    track_mentions = sum(1 for word in track_words if word in scene_lower)
    platform_mentions = sum(1 for word in platform_words if word in scene_lower)
    danger_mentions = sum(1 for word in danger_words if word in scene_lower)
    fire_mentions = sum(1 for word in fire_words if word in scene_lower)
    crowd_mentions = sum(1 for word in crowd_words if word in scene_lower)
    safe_mentions = sum(1 for word in safe_words if word in scene_lower)
    
    # Person on track detection (but not if explicitly safe)
    if person_mentions > 0 and track_mentions > 0 and safe_mentions == 0:
        # Check if person is actually on the tracks vs just mentioned
        on_track_indicators = ['on track', 'on the track', 'on rails', 'on the rails', 'standing on', 'walking on']
        on_track_specific = sum(1 for phrase in on_track_indicators if phrase in scene_lower)
        if on_track_specific > 0:
            obs.on_track_person = min(0.8, 0.6 + on_track_specific * 0.1)
        elif person_mentions > 0 and track_mentions > 0:
            # General co-occurrence but less confident - need stronger evidence
            near_indicators = ['near', 'close to', 'next to', 'beside', 'by the']
            near_mentions = sum(1 for phrase in near_indicators if phrase in scene_lower)
            if near_mentions > 0:
                # Person near tracks but not necessarily on them - lower confidence
                obs.on_track_person = min(0.4, 0.25 + near_mentions * 0.05)
            else:
                # Just mention of person and tracks together - very low confidence
                obs.on_track_person = min(0.3, 0.2 + (person_mentions + track_mentions) * 0.02)
    
    # Fallen person detection
    if person_mentions > 0 and danger_mentions > 0:
        obs.fallen_person = min(0.7, 0.4 + danger_mentions * 0.1)
    
    # Fire/smoke detection
    if fire_mentions > 0:
        obs.smoke_or_fire = min(0.8, 0.5 + fire_mentions * 0.15)
    
    # Crowd detection
    if crowd_mentions > 0 and (track_mentions > 0 or platform_mentions > 0):
        obs.crowd_on_track = min(0.7, 0.4 + crowd_mentions * 0.1)
    
    # Generic object detection (if no person but something mentioned on tracks)
    if track_mentions > 0 and person_mentions == 0 and any(word in scene_lower for word in ['object', 'item', 'thing', 'debris']):
        obs.object_on_track = 0.6
    
    # Evaluate using ontology
    decision = evaluate(obs)
    
    # Map severity to icons and colors
    severity_mapping = {
        Severity.NONE: {"icon": "✅", "color": "green"},
        Severity.LOW: {"icon": "🟢", "color": "lightgreen"}, 
        Severity.MEDIUM: {"icon": "🟠", "color": "orange"},
        Severity.HIGH: {"icon": "⚠️", "color": "red"},
        Severity.CRITICAL: {"icon": "🚨", "color": "darkred"}
    }
    
    severity_info = severity_mapping[decision.severity]
    
    return {
        "severity": decision.severity.name,
        "severity_icon": severity_info["icon"],
        "severity_color": severity_info["color"],
        "score": decision.score_0_100,
        "labels": [label.value for label in decision.labels],
        "explanations": decision.explanations,
        "fired_rules": decision.fired_rules,
        "ontology_used": True,
        "raw_description": scene_description,
        "observation": obs,
        "decision": decision
    }

def main():
    st.set_page_config(
        page_title="Video Frame Analyzer",
        page_icon="🎥",
        layout="wide"
    )
    
    st.title("🎥 Video Frame Analyzer with Local AI Models")
    st.markdown("Upload a video, provide a prompt, and analyze each frame using local AI models (CNN or Transformer)")
    
    # Load settings and initialize local models
    settings = load_settings()
    
    # Initialize local models if enabled
    local_manager = None
    local_models_available = False
    
    if LOCAL_MODELS_ENABLED:
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
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model type selection
        available_options = []
        if local_models_available:
            available_options.append("Local Models")
        if REMOTE_MODELS_ENABLED:
            available_options.append("Remote API")
        
        if not available_options:
            available_options = ["Remote API"]  # Fallback
            
        model_type = st.radio(
            "Model Type",
            available_options,
            help="Choose between local AI models or remote Hugging Face API"
        )
        
        if model_type == "Local Models" and local_models_available:
            # Local model selection
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
            
            api_token = None  # Not needed for local models
            
        else:
            # Remote API configuration
            default_token = settings.get('hugging_face_api_token', '')
            api_token = st.text_input(
                "Hugging Face API Token",
                value=default_token,
                type="password",
                help="Get your token from https://huggingface.co/settings/tokens or save in settings.json"
            )
            
            # Remote model selection
            selected_model = st.selectbox(
                "Select Model",
                options=list(AVAILABLE_MODELS.keys()),
                format_func=lambda x: AVAILABLE_MODELS[x]
            )
        
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
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input")
        
        # Video upload
        video_file = st.file_uploader(
            "Upload Video",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to analyze"
        )
        
        # Prompt input (conditional based on model)
        if model_type == "Local Models" and local_models_available and selected_model == "Person on Track Detector":
            # Person on Track Detector works automatically
            st.info("🤖 Person on Track Detector works automatically - no prompt needed!")
            prompt = "automatic"  # Set automatic prompt
        else:
            # Regular models need user prompt
            prompt = st.text_area(
                "Analysis Prompt",
                placeholder="Describe what you see in the image...",
                help="Enter the prompt to analyze each frame"
            )
        
        # Process button
        process_button = st.button("Process Video", type="primary")
    
    with col2:
        st.header("Results")
        results_container = st.container()
    
    # Processing logic
    if process_button and video_file and (prompt or (model_type == "Local Models" and selected_model == "Person on Track Detector")) and (api_token or model_type == "Local Models"):
        with st.spinner("Processing video..."):
            # Extract frames
            frames = extract_frames_from_video(video_file, fps)
            
            if not frames:
                st.error("No frames could be extracted from the video")
                return
            
            st.success(f"Extracted {len(frames)} frames from video")
            
            # Process each frame
            results = []
            progress_bar = st.progress(0)
            
            for i, frame_data in enumerate(frames):
                with st.spinner(f"Analyzing frame {i+1}/{len(frames)}..."):
                    # Process frame based on model type
                    if model_type == "Local Models" and local_models_available:
                        result = process_image_locally(
                            frame_data['frame'],
                            prompt,
                            selected_model,
                            local_manager
                        )
                    else:
                        result = query_huggingface_api(
                            frame_data['frame'],
                            prompt,
                            selected_model,
                            api_token
                        )
                    
                    # Extract scene description for ontology analysis
                    scene_description = ""
                    if 'person_on_track_detection' in result:
                        # For person detection results, use the analysis text
                        scene_description = result['person_on_track_detection'].get('detailed_analysis', {}).get('scene_description', '')
                    elif 'generated_text' in result:
                        scene_description = result['generated_text']
                    elif isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
                        scene_description = result[0]['generated_text']
                    
                    # Apply ontology analysis
                    ontology_analysis = analyze_scene_with_ontology(scene_description, use_ontology)
                    
                    results.append({
                        'frame_number': frame_data['frame_number'],
                        'timestamp': frame_data['timestamp'],
                        'image': frame_data['frame'],
                        'result': result,
                        'ontology_analysis': ontology_analysis
                    })
                    
                    progress_bar.progress((i + 1) / len(frames))
            
            # Display results
            with results_container:
                st.subheader("Analysis Results")
                
                for result_data in results:
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
                                # Severity display with color
                                severity_color = ontology.get('severity_color', 'green')
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
                                
                                st.divider()
                            
                            # Display original model results
                            st.write("**Model Output:**")
                            if 'error' in result_data['result']:
                                st.error(f"Error: {result_data['result']['error']}")
                            elif 'person_on_track_detection' in result_data['result']:
                                # Handle person-on-track detection results
                                detection = result_data['result']['person_on_track_detection']
                                
                                people_count = detection.get('people_count', 0)
                                confidence = detection.get('confidence', 0)
                                analysis = detection.get('analysis', 'No analysis')
                                person_on_track = detection.get('person_on_track', False)
                                
                                st.write(f"**Detection Analysis:** {analysis}")
                                
                                # Show metrics
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("👥 People Detected", people_count)
                                with col2:
                                    st.metric("📊 Model Confidence", f"{confidence:.0%}")
                            else:
                                if 'generated_text' in result_data['result']:
                                    # Handle direct generated_text response (local models)
                                    st.write(f"*{result_data['result']['generated_text']}*")
                                elif isinstance(result_data['result'], list) and len(result_data['result']) > 0:
                                    # Handle list responses (common for captioning models)
                                    if 'generated_text' in result_data['result'][0]:
                                        st.write(f"*{result_data['result'][0]['generated_text']}*")
                                    else:
                                        st.json(result_data['result'][0])
                                else:
                                    st.json(result_data['result'])
    
    elif process_button:
        if not video_file:
            st.error("Please upload a video file")
        if not prompt and not (model_type == "Local Models" and selected_model == "Person on Track Detector"):
            st.error("Please enter an analysis prompt")
        if not api_token and model_type == "Remote API":
            st.error("Please provide your Hugging Face API token for remote models")
        if model_type == "Local Models" and not local_models_available:
            st.error("Local models failed to initialize. Check your installation.")
    
    # Instructions
    with st.expander("How to use"):
        st.markdown("""
        ## Local AI Models (Recommended)
        1. **Upload a video**: Choose a video file (MP4, AVI, MOV, or MKV)
        2. **Select model type**: Choose "Local Models" for offline processing
        3. **Choose AI model**: 
           - **CNN (BLIP)**: Fast, good for object detection (~1.2GB)
           - **Transformer (ViT-GPT2)**: Detailed descriptions (~1.8GB)
        4. **Enter a prompt**: Describe what you want the AI to analyze
        5. **Adjust frame rate**: Set frames per second to extract (default: 1 fps)
        6. **Click Process**: Frames are processed locally on your machine
        
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
        
        ## Example Prompts
        - "Describe what you see in this image"
        - "Count the number of people in this scene"
        - "What objects are visible in this frame?"
        - "Describe the emotions and actions in this scene"
        - "What is the main activity happening here?"
        """)

if __name__ == "__main__":
    main()