# Deep Tagger API

Servicio basado en FastAPI que genera automáticamente metadatos de productos de e-commerce a partir de la URL de una sola imagen. Combina múltiples técnicas de IA/ML —un clasificador CNN, extracción de colores con K-Means, un transformer visión-lenguaje y un LLM— para producir un listado estructurado del producto en una única request.

## Cómo Funciona

```
                         ┌──────────────┐
                         │   URL Imagen │
                         └──────┬───────┘
                                │
                       descarga & preprocesado
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                  │
     ┌────────▼────────┐ ┌─────▼──────┐  ┌────────▼────────┐
     │  K-Means + KNN  │ │ TinyVGG CNN│  │  BLIP-2 VLM     │
     │  Extracción de  │ │ Clasificad.│  │  Generación de   │
     │  Colores        │ │ Producto   │  │  Título          │
     └────────┬────────┘ └─────┬──────┘  └────────┬────────┘
              │                │                   │
              └────────┬───────┴───────────────────┘
                       │
               ┌───────▼───────┐
               │  Claude LLM   │
               │  Generación   │
               │  Descripción  │
               └───────┬───────┘
                       │
              ┌────────▼────────┐
              │ Respuesta JSON  │
              │  Estructurada   │
              └─────────────────┘
```

## Modelos y Técnicas

### 1. Clasificación del Tipo de Producto — CNN (TinyVGG)

Una red neuronal convolucional personalizada entrenada sobre **Fashion MNIST** (70k imágenes, 10 clases).

| Detalle | Valor |
|---|---|
| Arquitectura | TinyVGG — 2 bloques conv (Conv2d -> ReLU -> Conv2d -> ReLU -> MaxPool2d) + clasificador lineal |
| Entrada | Imagen en escala de grises de 28x28 |
| Salida | Una de 10 clases: T-shirt/top, Pants, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot |
| Pesos | `deep_learning/torch_state/fashion_model_classifier_tiny_vgg_v2.pth` |

En tiempo de inferencia, la imagen de entrada se redimensiona a 28x28, se convierte a escala de grises, se invierten los colores (para coincidir con las convenciones de Fashion MNIST) y se normaliza antes de alimentarla al modelo.

**Fuente:** `deep_learning/product_type_classifier.py`

### 2. Extracción de Colores Dominantes — K-Means + KNN

Un pipeline de ML tradicional que identifica los colores dominantes del producto.

1. Se elimina el fondo de la imagen (los píxeles blancos/transparentes se descartan).
2. **K-Means clustering** (k=2) agrupa los píxeles restantes en dos clusters de colores dominantes.
3. Cada centroide del cluster se asocia al color con nombre más cercano de un diccionario de 22 colores usando **distancia euclídea** en el espacio RGB.

Esto devuelve nombres de colores legibles para humanos (p. ej. "navy blue", "red") sin la sobrecarga del deep learning.

**Fuente:** `machine_learning/knn_model.py`

### 3. Generación del Título del Producto — Transformer Visión-Lenguaje BLIP

Utiliza el modelo pre-entrenado **Salesforce/blip-image-captioning-base** de Hugging Face para generar un título corto del producto a partir de la imagen.

El modelo recibe la imagen junto con el prompt *"the product name is"* y genera hasta 10 tokens para completarlo. El resultado es un nombre conciso de producto como "short sleeve cotton tee".

**Fuente:** `transformer/blip_transformer.py`

### 4. Generación de la Descripción del Producto — LLM Claude

Una vez determinados el tipo de producto, el título y las etiquetas por los modelos previos, se envían a **Claude de Anthropic** (`claude-sonnet-4-20250514`) para generar una descripción de 2 a 3 oraciones apta para un listado de e-commerce.

Si no se configura una API key, el endpoint sigue funcionando y devuelve una descripción placeholder.

**Fuente:** `llm/claude_client.py`

## API

### `GET /`

Health check.

```json
{ "health": "UP" }
```

### `POST /predict-image`

Genera metadatos de producto a partir de la URL de una imagen.

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

> **Nota:** `material`, `occasion` y `season` son actualmente placeholders hardcodeados. Solo `color` se predice dinámicamente.

La documentación interactiva está disponible en `/docs` (Swagger UI) y `/redoc` una vez que el servidor esté corriendo.

## Primeros Pasos

### Requisitos previos

- Python 3.12+
- Una API key de Anthropic (opcional — la API funciona aunque no se provea)

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configurar la API key para la generación de descripciones (opcional):

```bash
export ANTHROPIC_API_KEY=your_key_here
```

### Ejecutar

```bash
python main.py
```

El servidor arranca en `http://0.0.0.0:8080`.

### Ejemplo

```bash
curl -X POST http://localhost:8080/predict-image \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://i.postimg.cc/DwnSW-Dnh/test-shirt.avif"}'
```

## Estructura del Proyecto

```
deep-tagger-api/
├── main.py                        # App FastAPI y endpoint de predicción
├── requirements.txt
├── model/
│   ├── request/DeepTaggerRequest.py
│   └── response/DeepTaggerResponse.py
├── aux_functions/auxiliary.py     # Descarga y preprocesado de imagen
├── machine_learning/knn_model.py  # Extracción de colores K-Means + KNN
├── transformer/blip_transformer.py# Generación de título con BLIP
├── deep_learning/
│   ├── product_type_classifier.py # Clasificador CNN TinyVGG
│   └── torch_state/               # Pesos del modelo guardados
├── llm/claude_client.py           # Generación de descripción con Claude
└── notebook/                      # Notebooks de entrenamiento y exploración
    ├── fashion_model_classifier_tiny_vgg.ipynb
    ├── multi_headed_cnn.ipynb
    └── knn.ipynb
```

## Stack Tecnológico

- **FastAPI** + **Uvicorn** — servidor de la API
- **PyTorch** + **torchvision** — entrenamiento e inferencia de la CNN
- **Hugging Face Transformers** — modelo visión-lenguaje BLIP
- **scikit-learn** — clustering K-Means
- **OpenCV** + **Pillow** — procesamiento de imágenes
- **Anthropic SDK** — integración con el LLM Claude

## Autor

jfflores90@gmail.com
