# Video Frame Analyzer with Hugging Face

A Streamlit application that extracts frames from videos and analyzes them using Hugging Face vision-language models.

## Features

- Upload video files (MP4, AVI, MOV, MKV)
- Extract frames at configurable intervals (fps)
- Analyze each frame using various Hugging Face models
- Custom prompt input for frame analysis
- Real-time results display

## Setup

1. Create a Python virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. Upgrade pip and install setuptools:
```bash
python -m pip install --upgrade pip setuptools wheel
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Get a Hugging Face API token:
   - Visit https://huggingface.co/settings/tokens
   - Create a new token

6. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Enter your Hugging Face API token in the sidebar
2. Select a vision-language model
3. Upload a video file
4. Enter your analysis prompt
5. Adjust frame extraction rate if needed
6. Click "Process Video"

## Available Models

- Kosmos-2: General vision-language understanding
- BLIP Image Captioning: Image captioning and description
- GIT Large COCO: Visual question answering
- ViT-GPT2: Image to text generation

## Example Prompts

- "Describe what you see in this image"
- "Count the number of people in this scene"
- "What objects are visible in this frame?"
- "Describe the emotions of people in this image"