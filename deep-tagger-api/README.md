# Deep Tagger API

A FastAPI-based service that automatically generates e-commerce product metadata from a single image URL. It combines multiple AI/ML techniques — a CNN classifier, K-Means color extraction, a vision-language transformer, and an LLM — to produce a structured product listing in one request.

## How It Works

```
                         ┌──────────────┐
                         │  Image URL   │
                         └──────┬───────┘
                                │
                        download & preprocess
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                  │
     ┌────────▼────────┐ ┌─────▼──────┐  ┌────────▼────────┐
     │  K-Means        │ │ TinyVGG CNN│  │  BLIP-2 VLM     │
     │  Color Extract. │ │ Product    │  │  Title           │
     │                 │ │ Classifier │  │  Generation      │
     └────────┬────────┘ └─────┬──────┘  └────────┬────────┘
              │                │                   │
              └────────┬───────┴───────────────────┘
                       │
               ┌───────▼───────┐
               │  Claude LLM   │
               │  Description  │
               │  Generation   │
               └───────┬───────┘
                       │
              ┌────────▼────────┐
              │ Structured JSON │
              │    Response     │
              └─────────────────┘
```

## Models & Techniques

### 1. Product Type Classification — CNN (TinyVGG)

A custom convolutional neural network trained on **Fashion MNIST** (70k images, 10 classes).

| Detail | Value |
|---|---|
| Architecture | TinyVGG — 2 conv blocks (Conv2d -> ReLU -> Conv2d -> ReLU -> MaxPool2d) + linear classifier |
| Input | 28x28 grayscale image |
| Output | One of 10 classes: T-shirt/top, Pants, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot |
| Weights | `deep_learning/torch_state/fashion_model_classifier_tiny_vgg_v2.pth` |

At inference time, the input image is resized to 28x28, converted to grayscale, color-inverted (to match Fashion MNIST conventions), and normalized before being fed to the model.

**Source:** `deep_learning/product_type_classifier.py`

### 2. Dominant Color Extraction — K-Means Clustering + KNN

A traditional ML pipeline that identifies the dominant colors of the product.

1. The image background is removed (white/transparent pixels are discarded).
2. **K-Means clustering** (k=2) groups the remaining pixels into two dominant color clusters.
3. Each cluster centroid is matched to the nearest named color from a 22-color dictionary using **Euclidean distance** in RGB space.

This returns human-readable color names (e.g. "navy blue", "red") without any deep learning overhead.

**Source:** `machine_learning/knn_model.py`

### 3. Product Title Generation — BLIP Vision-Language Transformer

Uses the pre-trained **Salesforce/blip-image-captioning-base** model from Hugging Face to generate a short product title from the image.

The model receives the image along with the prompt *"the product name is"* and generates up to 10 tokens to complete it. The result is a concise product name like "short sleeve cotton tee".

**Source:** `transformer/blip_transformer.py`

### 4. Product Description Generation — Claude LLM

Once the product type, title, and tags have been determined by the previous models, they are sent to **Anthropic's Claude** (`claude-sonnet-4-20250514`) to generate a 2–3 sentence product description suitable for an e-commerce listing.

If no API key is configured, the endpoint still works and returns a placeholder description.

**Source:** `llm/claude_client.py`

## API

### `GET /`

Health check.

```json
{ "health": "UP" }
```

### `POST /predict-image`

Generate product metadata from an image URL.

**Request:**

```json
{
  "image_url": "https://example.com/product.jpg"
}
```

**Response:**

```json
{
  "product_type": "T-shirt/top",
  "title": "short sleeve cotton tee",
  "description": "A classic short-sleeve cotton tee perfect for everyday wear...",
  "tags": {
    "color": "blue/navy blue",
    "material": "Denim",
    "occasion": "Casual",
    "season": "Summer"
  }
}
```

> **Note:** `material`, `occasion`, and `season` are currently hardcoded placeholders. Only `color` is dynamically predicted.

Interactive docs are available at `/docs` (Swagger UI) and `/redoc` once the server is running.

## Getting Started

### Prerequisites

- Python 3.12+
- An Anthropic API key (optional — the API still works without it)

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set the API key for description generation (optional):

```bash
export ANTHROPIC_API_KEY=your_key_here
```

### Run

```bash
python main.py
```

The server starts on `http://0.0.0.0:8080`.

### Example

```bash
curl -X POST http://localhost:8080/predict-image \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://i.postimg.cc/DwnSW-Dnh/test-shirt.avif"}'
```

## Project Structure

```
deep-tagger-api/
├── main.py                        # FastAPI app & prediction endpoint
├── requirements.txt
├── model/
│   ├── request/DeepTaggerRequest.py
│   └── response/DeepTaggerResponse.py
├── aux_functions/aux.py           # Image download & preprocessing
├── machine_learning/knn_model.py  # K-Means + KNN color extraction
├── transformer/blip_transformer.py# BLIP title generation
├── deep_learning/
│   ├── product_type_classifier.py # TinyVGG CNN classifier
│   └── torch_state/               # Saved model weights
├── llm/claude_client.py           # Claude description generation
└── notebook/                      # Training & exploration notebooks
    ├── fashion_model_classifier_tiny_vgg.ipynb
    ├── multi_headed_cnn.ipynb
    └── knn.ipynb
```

## Tech Stack

- **FastAPI** + **Uvicorn** — API server
- **PyTorch** + **torchvision** — CNN training & inference
- **Hugging Face Transformers** — BLIP vision-language model
- **scikit-learn** — K-Means clustering
- **OpenCV** + **Pillow** — image processing
- **Anthropic SDK** — Claude LLM integration

## Author

Jorge flores - jfflores90@gmail.com
Hernán Marano - herchugm@gmail.com
Nicolás Velázquez - 