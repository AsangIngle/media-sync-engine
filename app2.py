import os
import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
from pydub import AudioSegment
import simpleaudio as sa
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

from pydub import AudioSegment
AudioSegment.converter = r"C:/Users/HP/ffmpeg-7.1.1-essentials_build/ffmpeg-7.1.1-essentials_build/bin/ffmpeg.exe"
AudioSegment.ffprobe = r"C:/Users/HP/ffmpeg-7.1.1-essentials_build/ffmpeg-7.1.1-essentials_build/bin/ffprobe.exe"


# Load BLIP model and processor         
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load the video
video_path = "pexels-c-technical-6753390 (2160p).mp4"
cap = cv2.VideoCapture(video_path)

# Extract frames every 15th frame
frame_count = 0
captions = []

while True:
    success, frame = cap.read()
    if not success:
        break
    if frame_count % 15 == 0:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=image, return_tensors="pt")
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)
        captions.append(caption)
    frame_count += 1

cap.release()

# Use LangChain to turn captions into a story/narration
llm = Ollama(model='llama3')
prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful assistant. Generate a short story or voiceover-style narration using these scene captions.'),
    ('human', '{text}')
])
chain = prompt | llm
final_story = chain.invoke({'text': '. '.join(captions)})

# Generate and save TTS from story
tts = gTTS(final_story, lang='en')
tts.save("video_commentary.mp3")

# Convert mp3 to wav for reliable playback
sound = AudioSegment.from_mp3("video_commentary.mp3")
sound.export("video_commentary.wav", format="wav")

# Play the audio using simpleaudio (safer than playsound)
wave_obj = sa.WaveObject.from_wave_file("video_commentary.wav")
play_obj = wave_obj.play()
play_obj.wait_done()

# Optional: Clean up temporary files
os.remove("video_commentary.mp3")
os.remove("video_commentary.wav")
