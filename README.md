# UdeSA Computer Vision

A Computer Vision course project at Universidad de San Andrés (UdeSA), consisting of tools for AI-powered image tagging and dataset preparation.

## Deep Tagger UI

A React web application for AI-powered image tagging. Users submit a product image URL and receive AI-generated predictions including the product title, type, description, and visual tags overlaid on the image.

See [`deep-tagger-ui/README.md`](deep-tagger-ui/README.md) for setup and usage details.

## Deep Tagger API

A FastAPI backend that generates e-commerce product metadata from a single image URL. It chains four AI/ML models in a pipeline: a TinyVGG CNN classifies the product type, K-Means clustering extracts dominant colors, a BLIP vision-language transformer generates the product title, and Claude LLM produces a product description.

See [`deep-tagger-api/README.md`](deep-tagger-api/README.md) for setup and usage details.

## Image Puller

A Python utility that downloads product images from tagged CSV files and records their local paths. It supports multiple product types (tops, dresses, pants) and organizes downloaded images by category.

See [`img-puller/README.md`](img-puller/README.md) for setup and usage details.

## Author

Jorge Federico Flores — jfflores90@gmail.com