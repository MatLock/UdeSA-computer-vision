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
                                     threshold=30) -> np.ndarray:
  rgb = img_array[:, :, :3]
  diff = np.abs(rgb.astype(int) - np.array(background_color, dtype=int))
  mask = np.all(diff <= threshold, axis=2)

  result = img_array.copy()
  result[mask] = [255, 255, 255, 0]
  return result
