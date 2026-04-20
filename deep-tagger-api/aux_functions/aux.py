import numpy as np
import requests
from PIL import Image
from io import BytesIO

KMEANS_VALUES = 2

def download_image(image_url: str) -> np.ndarray:
  response = requests.get(image_url)
  response.raise_for_status()
  img = Image.open(BytesIO(response.content)).convert("RGBA")
  return np.array(img)

def remove_specific_color_background(img_array: np.ndarray, background_color=(255, 255, 255),
                                     threshold=10) -> np.ndarray:
  img = Image.fromarray(img_array).convert("RGBA")
  datas = img.getdata()
  newData = []

  for item in datas:
    if all(abs(c - tc) <= threshold for c, tc in zip(item[:3], background_color)):
      newData.append((255, 255, 255, 0))
    else:
      newData.append(item)

  img.putdata(newData)
  return np.array(img)
