from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

image_path = "img_4.jpg"
image = Image.open(image_path)

inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    output = model.generate(**inputs)

description = processor.decode(output[0], skip_special_tokens=True)

print("Image description:", description)
