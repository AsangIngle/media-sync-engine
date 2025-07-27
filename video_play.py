import os 
import cv2
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
import playsound
from PIL import Image
import threading
import math
import time
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate


# Load YOLO and BLIP models
video_path = "C:/Users/HP/Downloads/3195366-uhd_3840_2160_25fps.mp4"
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(video_path)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
vtt_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

dialogs=[]


def process_video_frames(cap, model, processor, vtt_model, dialogs):
    count = 0
    total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        frame = cv2.resize(frame, (width // 6, height // 6))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_arr = Image.fromarray(frame_rgb)

        results = model.predict(frame)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                cls_name = model.names[cls]
                conf = math.ceil((box.conf[0] * 100)) / 100

                if conf > 0.4:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    cv2.putText(frame, f'{cls_name}', (max(0, x1), max(35, y1)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # Captioning
        inputs = processor(images=frame_arr, return_tensors='pt')
        outputs = vtt_model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        if caption not in dialogs:
            dialogs.append(caption)
        
        
        count += 1
        print(f'Processing frame {count}/{total_frames} ({(count/total_frames)*100:.2f}%)')

    cap.release()
    cv2.destroyAllWindows()
    return dialogs

dialogs_list=process_video_frames(cap, model, processor, vtt_model, dialogs)

full_dialog='.'.join(dialogs_list)
llm = OllamaLLM(model='llama3')
prompt=ChatPromptTemplate([
    ('system','You are a helpfull assistant.Please Generate a short story or voiceover-style narrative using these scene captions'),
    ('human','{text}')
])
        
formatted_prompt=prompt.format_messages(text=full_dialog)
response = llm.invoke(formatted_prompt[1].content)
tts=gTTS(text=response,lang='en',slow=False)# slow=False for normal explicitly speed

tts.save('video_comentry.mp3')


while not os.path.exists('video_comentry.mp3'):
    time.sleep(0.1)

cap = cv2.VideoCapture(video_path)
def play_video_with_audio(cap):
    
   
    audio_thread=threading.Thread(target=playsound.playsound,args=('video_comentry.mp3',))
    audio_thread.start()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        frame = cv2.resize(frame, (width // 6, height // 6))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        cv2.imshow('frame_rgb',frame_rgb)

        if cv2.waitKey(0) &0xFF==ord('q'):
            break
        

    cap.release()
    cv2.destroyAllWindows()
    os.remove('video_comentry.mp3')
    

play_video_with_audio(cap)