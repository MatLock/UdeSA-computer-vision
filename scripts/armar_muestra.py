"""Arma la muestra del Hito 2 a partir del prototipo MatLock/UdeSA-computer-vision.

Baja los tres CSV de img-puller y los labels.json de los modelos multilabel desde
GitHub, fijados a un commit, y los guarda en data/muestra/ byte por byte, sin
modificarlos. Registra el SHA-256 de cada archivo en data/muestra/origen.csv.

Uso (desde la raíz del repo):
    python scripts/armar_muestra.py
"""

import csv
import hashlib
from datetime import date
from pathlib import Path

import requests

REPO = "MatLock/UdeSA-computer-vision"
COMMIT = "b9b3b2b7e023b668233a5b3ff39efee1f66123ec"
RAW = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/"

ARCHIVOS = {
    "tops_tags.csv": "img-puller/data/tops_tags.csv",
    "shoes_tags.csv": "img-puller/data/shoes_tags.csv",
    "pants_tags.csv": "img-puller/data/pants_tags.csv",
    "taxonomia/tops.labels.json":
        "deep-tagger-api/deep_learning/torch_state/baseline_v1/multilabel_classifier_tops_v1.labels.json",
    "taxonomia/shoes.labels.json":
        "deep-tagger-api/deep_learning/torch_state/baseline_v1/multilabel_classifier_shoes_v1.labels.json",
    "taxonomia/pants.labels.json":
        "deep-tagger-api/deep_learning/torch_state/baseline_v1/multilabel_classifier_pants_v1.labels.json",
}

RAIZ = Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "data" / "muestra"


def main():
    filas = []
    for local, remoto in ARCHIVOS.items():
        resp = requests.get(RAW + remoto, timeout=60)
        resp.raise_for_status()
        dest = DESTINO / local
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        sha = hashlib.sha256(resp.content).hexdigest()
        filas.append({"archivo": local, "origen": f"{REPO}@{COMMIT}:{remoto}",
                      "bytes": len(resp.content), "sha256": sha})
        print(f"{local}: {len(resp.content):,} bytes  sha256={sha[:12]}…")

    with open(DESTINO / "origen.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["archivo", "origen", "bytes", "sha256", "fecha_extraccion"])
        writer.writeheader()
        for fila in filas:
            writer.writerow({**fila, "fecha_extraccion": date.today().isoformat()})


if __name__ == "__main__":
    main()
