import numpy as np
from sklearn.cluster import KMeans
from aux_functions.aux import remove_specific_color_background

BASE_COLORS = {
    "red": [255, 0, 0],
    "dark red": [139, 0, 0],
    "green": [0, 128, 0],
    "dark green": [0, 100, 0],
    "light green": [144, 238, 144],
    "blue": [0, 0, 255],
    "navy blue": [0, 0, 128],
    "royal blue": [65, 105, 225],
    "light blue": [173, 216, 230],
    "white": [255, 255, 255],
    "black": [0, 0, 0],
    "yellow": [255, 255, 0],
    "cyan": [0, 255, 255],
    "magenta": [255, 0, 255],
    "gray": [128, 128, 128],
    "light gray": [192, 192, 192],
    "dark gray": [64, 64, 64],
    "orange": [255, 165, 0],
    "purple": [128, 0, 128],
    "pink": [255, 192, 203],
    "brown": [139, 69, 19],
    "beige": [245, 222, 179],
}

K_NEIGHBORS = 1

def _classify_color(rgb) -> str:
    rgb = np.array(rgb)
    min_dist = float("inf")
    closest_name = None
    for name, value in BASE_COLORS.items():
        dist = np.linalg.norm(rgb - np.array(value))
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name


def _dominant_color_names(color_list: list) -> set:
    return {_classify_color(color) for color in color_list}


def _extract_dominant_colors(img_array: np.ndarray) -> list:
  alpha = img_array[:, :, 3]
  rgb = img_array[:, :, :3]
  # Keep only non-transparent pixels
  opaque_pixels = rgb[alpha > 0].reshape((-1, 3))
  kmeans = KMeans(n_clusters=K_NEIGHBORS, n_init=10, random_state=42)
  kmeans.fit(opaque_pixels)
  dominant_colors = np.uint8(kmeans.cluster_centers_)
  return dominant_colors.tolist()


def predict_k_colors(img_array: np.ndarray) -> tuple[set, list]:
    no_bg = remove_specific_color_background(img_array)
    dominant_colors_rgb = _extract_dominant_colors(no_bg)
    return _dominant_color_names(dominant_colors_rgb), dominant_colors_rgb