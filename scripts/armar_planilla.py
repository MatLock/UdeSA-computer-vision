"""Arma la planilla y el visor para el re-etiquetado humano (confiabilidad, celda R6).

Elige 60 imágenes del subconjunto descargado (20 por categoría, semilla fija) y
genera, en data/etiquetado/:
  - planilla_base.csv: id, categoria, ruta_imagen y una columna vacía por atributo,
    más tipo_foto, color_principal y coincide_categoria.
  - visor.html: página local que muestra cada imagen con opciones de la taxonomía
    del prototipo y exporta etiquetador_A.csv o etiquetador_B.csv. No muestra la
    etiqueta original, para no sesgar.

Uso (desde la raíz del repo, después de scripts/descargar_imagenes.py):
    python scripts/armar_planilla.py
"""

import json
from pathlib import Path

import pandas as pd

SEMILLA = 42
POR_CATEGORIA = 20
RAIZ = Path(__file__).resolve().parent.parent
MUESTRA = RAIZ / "data" / "muestra"
SALIDA = RAIZ / "data" / "etiquetado"
ATRIBUTOS = {
    "tops": ["neck_style", "fit_silhouette", "season"],
    "shoes": ["wearing_occasion", "style_silhouette", "season"],
    "pants": ["pocket_details", "fit_silhouette", "season"],
}
TODOS = list(dict.fromkeys(a for v in ATRIBUTOS.values() for a in v))
EXTRAS = {
    "tipo_foto": ["packshot", "con modelo", "otra"],
    "color_principal": ["negro", "blanco", "gris", "beige", "marrón", "azul", "celeste", "verde", "rojo",
                        "rosa", "violeta", "amarillo", "naranja", "dorado/plateado", "multicolor/estampado"],
    "coincide_categoria": ["sí", "no"],
}


def main():
    sub = pd.read_csv(MUESTRA / "subconjunto_imagenes.csv", dtype=str)
    elegidas = (pd.concat(sub[sub["categoria"] == cat].sample(n=POR_CATEGORIA, random_state=SEMILLA)
                          for cat in ATRIBUTOS)
                  .sample(frac=1, random_state=SEMILLA))           # mezcladas, para no etiquetar por bloques
    base = elegidas[["id", "categoria"]].copy()
    base["ruta_imagen"] = elegidas["ruta"]
    for col in TODOS + list(EXTRAS):
        base[col] = ""
    SALIDA.mkdir(parents=True, exist_ok=True)
    base.to_csv(SALIDA / "planilla_base.csv", index=False)

    taxonomia = {}
    for cat in ATRIBUTOS:
        meta = json.loads((MUESTRA / "taxonomia" / f"{cat}.labels.json").read_text(encoding="utf-8"))
        taxonomia[cat] = {g["name"]: g["classes"] + ["otro / no se ve"] for g in meta["groups"]}

    items = [{"id": r.id, "categoria": r.categoria, "img": "../../" + r.ruta_imagen} for r in base.itertuples()]
    html = (Path(__file__).with_name("visor_plantilla.html").read_text(encoding="utf-8")
            .replace("__ITEMS__", json.dumps(items))
            .replace("__TAXONOMIA__", json.dumps(taxonomia, ensure_ascii=False))
            .replace("__ATRIBUTOS__", json.dumps(ATRIBUTOS))
            .replace("__EXTRAS__", json.dumps(EXTRAS, ensure_ascii=False))
            .replace("__COLUMNAS__", json.dumps(list(base.columns))))
    (SALIDA / "visor.html").write_text(html, encoding="utf-8")
    print(f"{len(base)} imágenes -> {SALIDA.relative_to(RAIZ)}/planilla_base.csv y visor.html")


if __name__ == "__main__":
    main()
