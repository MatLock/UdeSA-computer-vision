# Muestra del Hito 2

Los tres CSV etiquetados de `img-puller` y la taxonomía de los modelos de atributos del prototipo, copiados **sin modificar**. Es la muestra sobre la que corre [`notebooks/eda_viabilidad.ipynb`](../../notebooks/eda_viabilidad.ipynb).

## Origen

| Archivo | Ruta en el prototipo |
|---|---|
| `tops_tags.csv` | `img-puller/data/tops_tags.csv` |
| `shoes_tags.csv` | `img-puller/data/shoes_tags.csv` |
| `pants_tags.csv` | `img-puller/data/pants_tags.csv` |
| `taxonomia/tops.labels.json` | `deep-tagger-api/deep_learning/torch_state/baseline_v1/multilabel_classifier_tops_v1.labels.json` |
| `taxonomia/shoes.labels.json` | `deep-tagger-api/deep_learning/torch_state/baseline_v1/multilabel_classifier_shoes_v1.labels.json` |
| `taxonomia/pants.labels.json` | `deep-tagger-api/deep_learning/torch_state/baseline_v1/multilabel_classifier_pants_v1.labels.json` |

- **Repositorio:** [MatLock/UdeSA-computer-vision](https://github.com/MatLock/UdeSA-computer-vision), el prototipo original. Este repo es un fork suyo, así que el commit también está en su historia y los mismos archivos están en `img-puller/data/` y en `deep-tagger-api/deep_learning/torch_state/baseline_v1/`.
- **Commit:** `b9b3b2b7e023b668233a5b3ff39efee1f66123ec`
- **Fecha de extracción:** 2026-10-20
- El SHA-256 y el tamaño de cada archivo están en [`origen.csv`](origen.csv). Si un archivo no coincide con su hash, no es la muestra.

`subconjunto_imagenes.csv` es la lista de las 300 imágenes que se bajan para las mediciones sobre imagen (100 por categoría, estratificadas por `season`, semilla 42).

## Lo que sabemos y lo que no de la procedencia

- Las etiquetas son de Lily AI, sobre productos de sus empresas clientes. Las pusieron a mano expertos en moda, con mucha revisión humana, para entrenar la automatización de esa misma tarea. Jorge Federico Flores autorizó usarlas en la materia y publicar este extracto en el repo (2026-10-20).
- Los productos se publicaron entre junio y julio de 2025. El dataset original trae la fecha; este extracto no.
- En el dataset original, alrededor del 70 % de los productos son de Rue Gilt Groupe (Rue La La y Gilt) y el resto, de otros retailers como GAP y Foot Locker. En este extracto, las imágenes se sirven casi todas desde `asset1.ruecdn.com`, el CDN de Rue Gilt Groupe, con el código `RLLDTH` en el nombre del archivo.
- La taxonomía de `taxonomia/*.labels.json` es la del modelo entrenado, que descarta las clases con menos de 20 ejemplos. No es una taxonomía oficial del origen.

## Cómo se regenera

Desde la raíz del repo, con Python 3.14:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # en Linux/macOS: .venv/bin/python
.venv/Scripts/python scripts/armar_muestra.py               # CSV y taxonomía, fijados al commit
.venv/Scripts/python scripts/descargar_imagenes.py          # 300 imágenes en data/images/ (no se commitean)
.venv/Scripts/python scripts/armar_planilla.py              # planilla y visor del re-etiquetado
.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace notebooks/eda_viabilidad.ipynb
```

El notebook no necesita red: el chequeo de URLs está cacheado en `data/cache/url_check.csv`. Sin las imágenes descargadas, las celdas que las usan lo dicen y siguen.

## Exclusiones

- Las imágenes no están en el repo: se bajan con `scripts/descargar_imagenes.py` a `data/images/`, que está en `.gitignore`.
- La muestra cubre solo tres categorías (tops, calzado y pantalones). Lo que diga el EDA no vale para vestidos, abrigos ni accesorios.
- No hay título ni descripción: `img-puller/describer.py` nunca se corrió sobre estos CSV.
- No se limpió nada. El encabezado repetido de `shoes_tags.csv` (`relative_path` dos veces) está tal cual vino.
