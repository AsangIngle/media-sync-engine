# Media Sync Engine – Automated Video Commentary

This project is a Media Sync Engine that automatically generates sports-style commentary for videos by combining:

1. Computer Vision (YOLOv8) for object detection  
2. BLIP for image captioning  
3. LLMs (LLaMA via Ollama) for generating narrative-style commentary  
4. gTTS for text-to-speech  
5. MoviePy & OpenCV for stitching video with audio commentary  

---

## Features
1. Frame Analysis – Detects objects with YOLOv8 and captions them using BLIP.  
2. Narration Script Generation – Uses LLaMA 3 (via Ollama) to generate ESPN-style live commentary from captions.  
3. Text-to-Speech – Converts narration into realistic audio with gTTS.  
4. Audio-Video Sync – Stitches commentary audio with video using MoviePy/OpenCV.  
5. Frontend Support – Streamlit app for uploads, playback, and download.  

---

## Repository Structure
├── final_video_audio.py # Main FastAPI backend: API for video -> commentary pipeline
├── app.py # Test backend for video + audio stitching
├── client.py # Streamlit frontend: upload video, get narrated output
├── app2.py # Script to check audio generation separately
├── video_play.py # Script to test video with audio playback locally
├── outputs/ # Sample outputs (narrated videos & audio files)
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## Installation

git clone https://github.com/your-username/media-sync-engine.git
cd media-sync-engine

# Create and activate environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

Dependencies:

1.ultralytics (YOLOv8)
2.transformers (BLIP)
3.langchain-ollama
4.gTTS
5.moviepy
6.fastapi, uvicorn
7.streamlit
8.opencv-python, Pillow, playsound

1. Run Backend (FastAPI)
   uvicorn final_video_audio:app --reload
2. Run Client (Streamlit)
   streamlit run client.py
Steps:

1.Upload a video file.
2.Wait for processing (captioning → script → TTS → sync).
3.Preview and download the narrated video.

Local Testing Scripts
1.app.py → test backend for video stitching
2.app2.py → check only audio generation
3.video_play.py → play video with generated audio commentary locally


flowchart TD
    A[Video Input] --> B[YOLOv8 Object Detection]
    B --> C[BLIP Captioning]
    C --> D[LLaMA 3 (Ollama) - Commentary Script]
    D --> E[gTTS Text-to-Speech]
    E --> F[MoviePy/OpenCV - Audio/Video Sync]
    F --> G[Final Narrated Video]
    G --> H[Streamlit / FastAPI Interface]
