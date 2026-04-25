# UdeSA Computer Vision

Proyecto del curso de Computer Vision en la Universidad de San Andrés (UdeSA), que reúne herramientas para el etiquetado de imágenes con IA y la preparación de datasets.

## Deep Tagger UI

Aplicación web en React para el etiquetado de imágenes con IA. Los usuarios envían la URL de una imagen de producto y reciben predicciones generadas por IA, que incluyen el título, el tipo, la descripción y las etiquetas visuales superpuestas sobre la imagen.

Ver [`deep-tagger-ui/README.md`](deep-tagger-ui/README.md) para los detalles de instalación y uso.

## Deep Tagger API

Backend en FastAPI que genera metadatos de productos de e-commerce a partir de la URL de una sola imagen. Encadena cuatro modelos de IA/ML en un pipeline: una CNN TinyVGG clasifica el tipo de producto, K-Means extrae los colores dominantes, un transformer visión-lenguaje BLIP genera el título del producto y el LLM Claude produce la descripción.

Ver [`deep-tagger-api/README.md`](deep-tagger-api/README.md) para los detalles de instalación y uso.

## Image Puller

Utilidad en Python que descarga imágenes de productos desde archivos CSV etiquetados y registra sus rutas locales. Soporta múltiples tipos de productos (tops, vestidos, pantalones) y organiza las imágenes descargadas por categoría.

Ver [`img-puller/README.md`](img-puller/README.md) para los detalles de instalación y uso.

## Autor

Jorge Federico Flores — jfflores90@gmail.com
Hernán Marano - herchugm@gmail.com
Nicolás Velázquez - 
