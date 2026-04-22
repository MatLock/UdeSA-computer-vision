import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

'''
Generates product title using Pre-trained Model (BLIP-2)
'''
def predict_product_title(img_array: np.ndarray) -> str:
  image = Image.fromarray(img_array).convert('RGB')
  prompt = "the product type is"
  inputs = _processor(image, text=prompt, return_tensors="pt")
  outputs = _model.generate(**inputs, max_new_tokens=10)
  caption = _processor.decode(outputs[0], skip_special_tokens=True)
  return caption.replace(prompt, "").strip()