"""Baja un subconjunto estratificado de imágenes de la muestra, con semilla fija.

Elige N imágenes por categoría (tops, shoes, pants), con asignación proporcional
a `season`, y guarda la lista en data/muestra/subconjunto_imagenes.csv (se
commitea). Las imágenes van a data/images/<categoria>/<id><ext> en su tamaño
original y no se commitean. Es idempotente: no vuelve a bajar lo que ya está.

Uso (desde la raíz del repo):
    python scripts/descargar_imagenes.py            # 100 por categoría
    python scripts/descargar_imagenes.py --n 50
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

SEMILLA = 42
CATEGORIAS = ("tops", "shoes", "pants")
RAIZ = Path(__file__).resolve().parent.parent
MUESTRA = RAIZ / "data" / "muestra"
IMAGENES = RAIZ / "data" / "images"
LISTA = MUESTRA / "subconjunto_imagenes.csv"
HEADERS = {"User-Agent": "UdeSA-toi-hito2/1.0 (trabajo academico)"}


def elegir(df: pd.DataFrame, n: int, estrato: str) -> pd.DataFrame:
    """Muestreo estratificado con asignación proporcional (resto mayor) y semilla fija."""
    grupos = df[estrato].fillna("<vacío>")
    proporcion = grupos.value_counts(normalize=True).sort_index()
    cuota = proporcion * n
    asignado = np.floor(cuota).astype(int)
    faltan = n - asignado.sum()
    for clave in (cuota - asignado).sort_values(ascending=False).index[:faltan]:
        asignado[clave] += 1
    partes = [df[grupos == clave].sample(n=k, random_state=SEMILLA) for clave, k in asignado.items() if k > 0]
    return pd.concat(partes).sort_values("id")


def bajar(fila) -> str:
    destino = RAIZ / fila.ruta
    if destino.exists():
        return "ya estaba"
    destino.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(fila.image_url, timeout=30, headers=HEADERS)
        resp.raise_for_status()
        destino.write_bytes(resp.content)
        return "ok"
    except requests.RequestException as e:
        return f"error: {e.__class__.__name__}"
    finally:
        time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="imágenes por categoría")
    args = parser.parse_args()

    elegidas = []
    for cat in CATEGORIAS:
        df = pd.read_csv(MUESTRA / f"{cat}_tags.csv", dtype=str)
        sub = elegir(df, args.n, "season")[["id", "image_url", "season"]].copy()
        sub.insert(1, "categoria", cat)
        ext = sub["image_url"].map(lambda u: Path(urlparse(u).path).suffix or ".jpg")
        sub["ruta"] = "data/images/" + cat + "/" + sub["id"] + ext
        elegidas.append(sub)
    lista = pd.concat(elegidas, ignore_index=True)
    lista[["id", "categoria", "season", "image_url", "ruta"]].to_csv(LISTA, index=False)

    with ThreadPoolExecutor(max_workers=4) as pool:
        estados = list(pool.map(bajar, lista.itertuples()))
    print(pd.Series(estados).value_counts().to_string())
    print(f"Lista en {LISTA.relative_to(RAIZ)}; imágenes en {IMAGENES.relative_to(RAIZ)}/")


if __name__ == "__main__":
    main()
