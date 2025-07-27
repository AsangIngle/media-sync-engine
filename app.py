import os
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2
from PIL import Image
from gtts import gTTS
import playsound
import torch
#from googletrans import Translator

#translator=Translator()

img = cv2.imread("C:/Users/HP/Downloads/pexels-camcasey-1687093.jpg")
img = cv2.resize(img, (img.shape[1] // 6, img.shape[0] // 6))

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_pil = Image.fromarray(img_rgb)

# Load correct BLIP model and processor
processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')

inputs = processor(images=img_pil, return_tensors='pt')
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)


#hindi_caption=translator.translate(caption,dest='hi')
tts=gTTS(text=caption,lang='en')


tts.save('caption_audio.mp3')
cv2.putText(img, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
playsound.playsound('caption_audio.mp3')


while True:
    cv2.imshow('BLIP Caption', img)
    
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
os.remove("caption_audio.mp3")
