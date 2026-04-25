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

def _detect_background_color(img_array: np.ndarray, sample_depth=5) -> tuple:
  h, w = img_array.shape[:2]
  edge_pixels = []
  for i in range(sample_depth):
    edge_pixels.extend(img_array[i, :, :3].tolist())       # top rows
    edge_pixels.extend(img_array[h - 1 - i, :, :3].tolist()) # bottom rows
    edge_pixels.extend(img_array[:, i, :3].tolist())        # left cols
    edge_pixels.extend(img_array[:, w - 1 - i, :3].tolist()) # right cols

  edge_pixels = np.array(edge_pixels)
  median_color = np.median(edge_pixels, axis=0).astype(int)
  return tuple(median_color)


def remove_specific_color_background(img_array: np.ndarray, background_color=None,
                                     threshold=30) -> np.ndarray:
  if background_color is None:
    background_color = _detect_background_color(img_array)

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
