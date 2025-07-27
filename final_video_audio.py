import os
import cv2
from PIL import Image
import re
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

from gtts import gTTS
from langchain_ollama import OllamaLLM
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
from moviepy.editor import VideoFileClip, AudioFileClip
from langchain.prompts import ChatPromptTemplate
import uvicorn

# Initialize models
model = YOLO('yolov8n.pt')
processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
caption_model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
llm = OllamaLLM(model='llama3')

app = FastAPI(
    title='API for creating commentary',
    version='1.0',
    description='An API to add commentary to a video'
)

def remove_brackets(text):
    cleaned = re.sub(r'\([^)]*\)', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    dialogs = []
    count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("\n[INFO] Starting frame processing...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        frame = cv2.resize(frame, (width // 6, height // 6))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame_rgb)

        inputs = processor(images=frame_image, return_tensors='pt')
        outputs = caption_model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)

        if caption not in dialogs:
            dialogs.append(caption)

        count += 1
        print(f"Processed frame {count}/{total_frames} - Caption: {caption}")

    cap.release()
    return dialogs

@app.post('/stich_video_with_audio')
async def create_audio_and_stitch(file: UploadFile = File(...)):
    input_path = 'input_video.mp4'
    audio_output = 'video_commentary.mp3'
    output_path = 'output_with_audio.mp4'

    # Save uploaded video
    with open(input_path, 'wb') as f:
        f.write(await file.read())

    dialogs = process_video(input_path)
    full_dialog = '. '.join(dialogs)

    print("\n[INFO] Generating voiceover script using LLM...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a passionate soccer commentator narrating a highlight video. The following text contains scene captions from a soccer match. Write a short, vivid, emotionally charged voiceover — like you're calling a live ESPN highlight. Use energetic storytelling, avoid lists, and keep it under 30 seconds."),
        ("human", "{text}")
    ])
    formatted_prompt = prompt.format_messages(text=full_dialog)
    response = llm.invoke(formatted_prompt)
    print(f"\n[LLM Response]: {response}")

    clean_text = remove_brackets(response)
    tts = gTTS(text=clean_text, lang='en', slow=False)
    tts.save(audio_output)

    # Wait until the audio file is ready
    while not os.path.exists(audio_output):
        pass

    print("\n[INFO] Stitching audio with video using moviepy...")
    video_clip = VideoFileClip(input_path)
    audio_clip = AudioFileClip(audio_output)
    looped_video = video_clip.loop(duration=audio_clip.duration)
    final_clip = looped_video.set_audio(audio_clip)
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

    return FileResponse(output_path, media_type="video/mp4", filename="output_with_audio.mp4")

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
