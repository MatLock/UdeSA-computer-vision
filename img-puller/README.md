# UdeSa Image Puller

Descarga imágenes de productos desde archivos CSV etiquetados y registra sus rutas locales.

## Requisitos

- Python 3.8+
- Librería `requests`

```bash
pip install -r requirements.txt
```

## Uso

```bash
python main.py <ruta-al-csv>
```

### Ejemplos

```bash
python main.py data/tops_tags.csv
python main.py data/shoes_tags.csv
python main.py data/pants_tags.csv
```

## Qué hace

1. Lee el archivo CSV y detecta el tipo de producto a partir del nombre del archivo (`tops`, `shoes` o `pants`).
2. Descarga cada imagen desde la columna `image_url` en `images/<product_type>/`.
3. Agrega una columna `relative_path` al CSV con la ruta local de cada imagen descargada.

## Estructura de salida

```
.
├── data/
│   ├── tops_tags.csv
│   ├── shoes_tags.csv
│   └── pants_tags.csv
├── images/
│   ├── tops/
│   ├── shoes/
│   └── pants/
└── main.py
```
