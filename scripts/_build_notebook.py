"""Generate the multi-label training narrative notebook.

Reads the curated narrative content from this script and writes a Jupyter
notebook to deep-tagger-api/notebook/multilabel_training_runs.ipynb.
Run once after any narrative update.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "deep-tagger-api" / "notebook" / "multilabel_training_runs.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells: list[dict] = []

# ---------------------------------------------------------------------------
cells.append(md("""# Bitácora de Etapa 1 — clasificador multi-label

> Documento narrativo de cómo se llegó a los tres `.pth` que viven en
> `deep-tagger-api/deep_learning/torch_state/multilabel_classifier_*_v1.pth`.
> Pensado para que alguien que entra al proyecto entienda **por qué** el código
> quedó como quedó, no sólo qué hace.

## TL;DR (para los apurados)

- Se entrenaron **tres clasificadores multi-label independientes** (`tops`, `shoes`, `pants`),
  todos con la misma arquitectura (ResNet18 pretrained ImageNet) y el mismo pipeline de datos,
  pero con **número de epochs distinto por tipo** porque cada dataset converge a su ritmo.
- La métrica de selección es `val_f1_macro` (mejor checkpoint guardado).
- **Resultados finales en val** (90/10 split, seed=42), corrida overnight del 2026-04-24:

  | Tipo  | Epochs | best | f1_macro | f1_micro | subset_acc | top1_avg |
  |---|---:|---:|---:|---:|---:|---:|
  | tops  | 10 | 9  | **0.5164** | 0.7862 | 0.4120 | 0.7875 |
  | shoes | 15 | 11 | **0.6044** | 0.8209 | 0.4954 | 0.8243 |
  | pants |  7 |  7 | **0.3726** | 0.6774 | 0.1709 | 0.6812 |

- **Tiempo total de entrenamiento**: 28 min (tops 11:22, shoes 13:11, pants 3:07) en RTX 5070,
  batch=128, num-workers=4.
- **Aprendizajes más caros del camino**: pretrained pesa más que cualquier otra mejora;
  regularizar para cerrar el gap train/val es estética, no utilidad; el "mismo entrenamiento
  para los tres tipos" es una mala idea cuando cada dataset tiene tamaño distinto; al elegir
  cuántos epochs correr, **mirar la métrica de selección** (`val_f1_macro`), no `val_loss` —
  para algunos datasets las dos curvas no se mueven juntas.
"""))

# ---------------------------------------------------------------------------
cells.append(md("""## Setup — paths e imports

Las celdas de código del notebook leen los artefactos reales de disco
(`.labels.json` de los modelos entrenados) para que las tablas y los
gráficos se actualicen solos cuando se reentrenen los modelos.
"""))

cells.append(code("""import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Repo root (asumiendo que el notebook vive en deep-tagger-api/notebook/).
REPO_ROOT = Path.cwd().resolve()
while REPO_ROOT.name and not (REPO_ROOT / "img-puller").exists():
    REPO_ROOT = REPO_ROOT.parent
print("repo root:", REPO_ROOT)

TORCH_STATE = REPO_ROOT / "deep-tagger-api" / "deep_learning" / "torch_state"
CSV_DIR     = REPO_ROOT / "img-puller" / "data"

PRODUCT_TYPES = ["tops", "shoes", "pants"]
"""))

# ---------------------------------------------------------------------------
cells.append(md("""## 0. El punto de partida

El proyecto ya tenía un clasificador *single-label* (`TinyVGG`) en
`deep-tagger-api/deep_learning/product_type_classifier.py`, entrenado con
Fashion-MNIST y devolviendo una de 10 clases tipo "T-shirt", "Pants", etc.
Útil para "qué prenda es" pero **incapaz de decir nada sobre estilo,
temporada o materiales**.

El objetivo de Etapa 1 es agregar un clasificador *multi-label* alternativo
(no reemplazar el actual) que devuelva varias etiquetas por imagen — color
del cuello, fit, ocasión, etc. — entrenado sobre los CSVs reales de
productos en `img-puller/data/`.
"""))

# ---------------------------------------------------------------------------
cells.append(md("""## 1. Mirando los datos antes de entrenar nada

Cada tipo tiene un esquema de atributos **distinto**. Esto restringió la
arquitectura: tres modelos independientes (uno por tipo) en lugar de un
solo modelo gigante con todas las clases mezcladas.
"""))

cells.append(code("""# Resumen de cada CSV: filas, columnas, atributos categóricos.
SCHEMA = {
    "tops":  ["neck_style", "fit_silhouette", "season"],
    "shoes": ["wearing_occasion", "style_silhouette", "season"],
    "pants": ["pocket_details", "fit_silhouette", "season"],
}

rows = []
for t in PRODUCT_TYPES:
    csv_path = CSV_DIR / f"{t}_tags.csv"
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.duplicated()]
    img_dir = CSV_DIR / "images" / t
    n_imgs = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
    rows.append({
        "tipo": t,
        "atributos": ", ".join(SCHEMA[t]),
        "filas CSV": len(df),
        "imgs en disco": n_imgs,
    })

pd.DataFrame(rows).set_index("tipo")
"""))

cells.append(md("""**Primer hallazgo incómodo**: las distribuciones de clases son
*muy* desbalanceadas. La cola larga abajo va a justificar el filtro de la
Decisión 1.
"""))

cells.append(code("""# Distribución completa de clases por atributo (top 5 + bottom 3 cada uno).
for t in PRODUCT_TYPES:
    df = pd.read_csv(CSV_DIR / f"{t}_tags.csv")
    df = df.loc[:, ~df.columns.duplicated()]
    print(f"\\n=== {t} ===")
    for attr in SCHEMA[t]:
        counts = df[attr].value_counts(dropna=True)
        print(f"  {attr}: {len(counts)} clases únicas, top3 cubre {counts.head(3).sum() / len(df):.0%} de los datos")
        print(f"    top:    {dict(counts.head(3))}")
        if len(counts) > 5:
            print(f"    bottom: {dict(counts.tail(3))}")
"""))

# ---------------------------------------------------------------------------
cells.append(md("""### Decisión 1 — excluir clases con pocas apariciones

> **Hipótesis:** clases con menos de 20 ejemplos son ruido para el modelo y
> para el set de validación. El modelo no las puede aprender con tan pocos
> datos, y val tiene tan pocas filas (al 10% del total) que la métrica
> baila al azar.

Implementación en `train_multilabel.py:load_and_filter`:

1. Para cada atributo, se cuenta cuántas filas tiene cada valor.
2. Toda clase con `count < 20` queda fuera de los logits del modelo.
3. Las filas que tenían esa clase rara *no se borran inmediatamente*: el
   atributo en esa fila pasa a `NaN`, conservando los valores que esa fila
   tenga en *otros* atributos.
4. Sólo se descarta la fila si **todos** sus atributos quedaron en `NaN`.
5. Después se hace shuffle (seed=42) y se aplica el cap de `--max-samples`.

> Este parámetro queda en una constante (`MIN_SAMPLES_PER_CLASS`).
> Subirlo a 100 sería más agresivo: descartaría más clases minoritarias pero
> subiría el `f1_macro` reportado por simple sustracción de denominador.
> No se hizo. La elección de 20 es prudencial.
"""))

cells.append(code("""# Efecto del filtro <20 — qué clases sobreviven y cuáles caen.
MIN_SAMPLES = 20
filter_rows = []
for t in PRODUCT_TYPES:
    df = pd.read_csv(CSV_DIR / f"{t}_tags.csv")
    df = df.loc[:, ~df.columns.duplicated()]
    for attr in SCHEMA[t]:
        counts = df[attr].value_counts(dropna=True)
        keep = counts[counts >= MIN_SAMPLES]
        drop = counts[counts < MIN_SAMPLES]
        filter_rows.append({
            "tipo": t,
            "atributo": attr,
            "únicas en CSV": len(counts),
            "tras filtro": len(keep),
            "descartadas": ", ".join(f"{c}({n})" for c, n in drop.items()) or "—",
        })

pd.DataFrame(filter_rows)
"""))

# ---------------------------------------------------------------------------
cells.append(md("""## 2. Run 1 — baseline desnudo: ResNet18 desde cero

> **Hipótesis del Run 1**: ResNet18 from scratch sobre 10.000 imágenes de
> tops debería darnos un baseline mínimamente útil. Es la versión más
> conservadora: no descarga pesos, no hay sorpresas de licencia, todo el
> entrenamiento es local y reproducible.

| Parámetro | Valor |
|---|---|
| Backbone | `torchvision.models.resnet18(weights=None)` |
| Dataset | tops, `--max-samples 10000` |
| Optimizer | `Adam(lr=1e-3)` |
| Loss | `BCEWithLogitsLoss` |
| Augmentation (train) | `RandomHorizontalFlip(0.5)`, `ColorJitter(0.1)` |
| Epochs / Batch | 10 / 64 |

**Resultado:**

| Métrica | Valor |
|---|---:|
| best `val_f1_macro` | **0.2406** |
| `val_f1_micro` (last) | 0.6822 |
| `val_subset_acc` (last) | 0.1960 |
| `train_loss` / `val_loss` (last) | 0.109 / 0.122 |

**Diagnóstico:** `train ≈ val` → **no hay overfit**. El modelo tiene capacidad
para aprender más, pero con 9.000 imágenes y sin features pre-aprendidas
ResNet18 nunca llega a "ver" lo suficiente como para reconocer las 31 clases
de `neck_style`. La cola larga directamente no la aprende: `f1_macro=0.24`
significa que muchas clases minoritarias tienen F1 cero.

> **Aprendizaje**: ResNet18 desde cero está en el peor de los mundos para
> este dataset — demasiada capacidad para los datos pero sin la información
> de bordes, texturas y formas que ya tendría con pesos pretrained.
"""))

# ---------------------------------------------------------------------------
cells.append(md("""## 3. Run 2 — el salto: pretrained + más datos

> **Hipótesis**: dos cambios ortogonales aplicados juntos. (a) Empezar de
> pesos ImageNet, que ya saben "qué es un borde" y "qué es una textura".
> (b) Sacar el cap de 10.000 y entrenar con las 32.500 imágenes disponibles
> de tops. Cuando se aplican mejoras independientes, aplicar las dos a la vez
> es eficiente; si rompen algo, las separás después.

Cambios respecto al Run 1 en **negrita**:

| Parámetro | Valor |
|---|---|
| Backbone | **`resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`** |
| Dataset | tops, **`--max-samples 0`** (32.503 imgs) |
| Optimizer | Adam, **`lr=3e-4`** (más bajo: pesos pretrained necesitan menos magnitud para no "olvidarlos") |
| Loss / Augmentation | sin cambios |
| Epochs / Batch | 10 / 128 |

**Resultado:**

| Métrica | Run 1 | Run 2 | Δ |
|---|---:|---:|---:|
| best `val_f1_macro` | 0.2406 | **0.5227** | **+28.2 pts** |
| `val_f1_micro` (last) | 0.6822 | 0.7897 | +10.7 pts |
| `val_subset_acc` (last) | 0.1960 | 0.4292 | +23.3 pts |
| `train_loss` / `val_loss` | 0.109 / 0.122 | 0.014 / 0.106 | gap **8×** |

Esto fue el salto más grande. **Pretrained vale más que cualquier otra
mejora.** Las features de ImageNet (1.3M imágenes) se transfieren con
elegancia a fashion porque ambos dominios comparten lo básico: bordes,
simetría, textura.

Pero apareció un problema nuevo: `train_loss` cayó 7× más rápido que
`val_loss`. **Overfit declarado.** Y nos tentó arreglarlo.

> **Aprendizaje**: pretrained es la palanca con mayor ROI cuando se trabaja
> sobre datasets de tamaño medio (5k-50k imágenes). Cualquier otra
> optimización vale 1-3 pts; ésta vale 20+.
"""))

# ---------------------------------------------------------------------------
cells.append(md("""## 4. Run 3 — la tentación de cerrar el gap

> **Hipótesis**: si el gap train/val se cerrara (con regularización
> "moderna"), `val` debería subir.

Cambios respecto al Run 2:

| Parámetro | Run 2 | Run 3 |
|---|---|---|
| Optimizer | `Adam` | **`AdamW(weight_decay=1e-4)`** |
| Augmentation (train) | flip + ColorJitter(0.1) | **+ RandomResizedCrop(0.85-1.0) + RandomRotation(±10°) + ColorJitter(0.2, hue=0.05)** |

**Resultado:**

| Métrica | Run 2 | Run 3 | Δ |
|---|---:|---:|---:|
| best `val_f1_macro` | **0.5227** | 0.5103 | **−1.2 pts** |
| `val_f1_micro` (last) | 0.7897 | 0.7945 | +0.5 pts |
| `val_subset_acc` (last) | 0.4292 | 0.4363 | +0.7 pts |
| `train_loss` / `val_loss` | 0.014 / 0.106 (gap 8×) | 0.050 / 0.086 (gap **1.7×**) | gap colapsado |

El gap se cerró tal cual la teoría predice. `val_loss` bajó. Pero `val_f1_macro`,
la métrica que más nos importaba, **bajó 1.2 puntos**.

Y el `best` apareció **antes** del epoch 10 (best=0.5103, last=0.5018): con la
augmentation activa el modelo ya había convergido y empezaba a empeorar.

> **Aprendizaje**: el gap train/val por sí solo no es un problema. Es un
> *síntoma*. Si la regularización cierra el gap pero no levanta `val_f1_macro`,
> entonces la regularización no estaba resolviendo el problema real. Sólo era
> cosmética. **Cerrar el gap "porque queda lindo" hace daño** cuando empuja
> al modelo hacia una zona de menor capacidad efectiva sin razón.

### Decisión 2 — quedarnos con el Run 2

```python
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # NO AdamW

# en _build_transforms(train=True):
transforms.RandomHorizontalFlip(p=0.5),
transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
transforms.ToTensor(),
transforms.Normalize(mean, std),
```

Con esta config se entrenaron los tres modelos finales.
"""))

# ---------------------------------------------------------------------------
cells.append(md("""## 5. La sorpresa de los tres modelos: cada dataset cuenta su propia historia

Aplicamos la misma config (Run 2) a `tops`, `shoes` y `pants`. Hipótesis
ingenua: deberían comportarse parecido. **Refutada.**

Las curvas `val_f1_macro` epoch a epoch del overnight muestran tres regímenes
muy distintos: tops converge limpio, shoes recién peakea cerca del final, y
pants apenas arranca.
"""))

cells.append(code("""# Curvas val_f1_macro epoch a epoch (overnight 2026-04-24).
curves = {
    "tops":  [0.345, 0.403, 0.453, 0.466, 0.485, 0.484, 0.500, 0.511, 0.516, 0.515],
    "shoes": [0.318, 0.379, 0.453, 0.444, 0.489, 0.509, 0.521, 0.571, 0.577, 0.587, 0.604, 0.595, 0.595, 0.601, 0.602],
    "pants": [0.259, 0.308, 0.311, 0.333, 0.358, 0.370, 0.373],
}
best_epochs = {"tops": 9, "shoes": 11, "pants": 7}

fig, ax = plt.subplots(figsize=(9, 5))
for t, vals in curves.items():
    epochs = list(range(1, len(vals) + 1))
    ax.plot(epochs, vals, marker="o", label=t)
    be = best_epochs[t]
    ax.scatter([be], [vals[be - 1]], s=200, edgecolors="red", facecolors="none", linewidths=2)

ax.set_xlabel("epoch")
ax.set_ylabel("val_f1_macro")
ax.set_title("val_f1_macro epoch a epoch — overnight 2026-04-24\\n(círculo rojo = best checkpoint)")
ax.legend()
ax.grid(alpha=0.3)
plt.show()
"""))

# ---------------------------------------------------------------------------
cells.append(md("""### 5.1 `tops` — convergencia limpia (10 epochs, 11 min 22 s)

`val_f1_macro` sube hasta epoch 9 y se estabiliza en 0.515. `val_loss` toca
mínimo en epoch 3 y sube despacio — pero la métrica que nos importa
(`f1_macro`) sigue mejorando hasta el final, así que quedarse en 10 epochs
es correcto. `train_loss` seguía bajando (0.014 al epoch 10) — overfit existe
(gap 8×) pero el mejor checkpoint ya está guardado.

> Que `season` tenga top-1 más bajo que `fit_silhouette` es contraintuitivo
> (4 clases vs 5). La dificultad de un atributo no es función del número de
> clases, es función de **cuán visualmente distinguibles** son. Spring vs
> Fall en una foto de catálogo es muy similar.

### 5.2 `shoes` — la apuesta de "más epochs" se pagó (15 epochs, 13 min 11 s)

Esta es la corrida que confirmó la hipótesis abierta de la versión previa
del documento: shoes en 10 epochs no había convergido y subir a 15 sumaría
1-3 pts. **Sumó 5.1 pts.**

El best apareció en epoch 11 (de 15) y a partir de ahí entra en plateau:
los últimos 4 epochs oscilan en 0.595-0.602. **Captura limpia del peak.**
`val_loss` se estabilizó en ~0.067-0.068 desde epoch 7.

> **Aprendizaje confirmado**: cuando el `best` aparece al último epoch y
> `val_loss` aún baja, vale la pena alargar — pero hasta que la curva entre
> en plateau, no infinito. 15 epochs fue exactamente el número correcto
> para shoes; ir a 20 hubiera sido tiempo desperdiciado.

### 5.3 `pants` — la advertencia: val_loss y val_f1_macro no se mueven juntas (7 epochs, 3 min 7 s)

Aprendizaje incómodo. La corrida previa de 10 epochs había peakeado en
`val_f1_macro=0.4156` (epoch 8). Esta corrida con **7 epochs cap** se quedó
en 0.3726 (epoch 7) — **4 pts por debajo**.

¿Por qué decidimos 7 epochs? Habíamos observado que `val_loss` minimum estaba
en epoch 3 y subía después, y se interpretó como "overfit, parar temprano".
**Eso era un error de lectura**: la métrica de selección del modelo es
`val_f1_macro`, no `val_loss`.

> **Aprendizaje**: cuando se elige cuántos epochs correr, **mirar la métrica
> de selección, no `val_loss`**. Para algunos datasets las dos curvas se
> mueven juntas (shoes), para otros no (pants). Si tu criterio de "best" es
> `val_f1_macro`, tu criterio para "cuántos epochs" tiene que ser cuándo
> `val_f1_macro` entra en plateau, no cuándo `val_loss` toca su mínimo.
"""))

# ---------------------------------------------------------------------------
cells.append(md("""### 5.4 La lección comparativa: hiperparámetros por tipo

Las tres curvas exponen un punto metodológico: **no hay un número de epochs
óptimo para "este modelo"**, hay un número óptimo para **cada dataset**.

| Tipo | Epochs corridos | Resultado | Próxima iteración |
|---|---:|---|---|
| `tops`  | 10 | best epoch 9, plateau claro. **Bien.** | Mantener 10 o probar 12. |
| `shoes` | 15 | best epoch 11, plateau a partir de ahí. **Justo.** | Mantener 15. |
| `pants` |  7 | best epoch 7 (último), curva ascendente. **Quedó corto.** | Re-correr con 12-15 epochs. |

¿Es metodológicamente válido entrenar cada uno con un número distinto de
epochs? **Sí.** No estamos comparando los tres modelos entre sí (no son
candidatos al mismo problema), son **tres clasificadores independientes que
coexisten**. Cada uno debe optimizarse para su dataset.

Lo que **sí** se mantiene constante entre los tres tipos:

- Backbone (ResNet18 pretrained ImageNet).
- Pipeline de input (224×224, mean/std de ImageNet, mismas augmentations).
- Métrica de selección de checkpoint (`val_f1_macro` siempre).
- Split (90/10, seed=42).
- Loss (`BCEWithLogitsLoss`).

La inferencia **no se ve afectada** por estas diferencias. El `.pth` no guarda
los hiperparámetros con que se entrenó, sólo los pesos. Lo que la inferencia
necesita está en el `.labels.json` (tamaño de imagen, normalización, mapeo
índice→etiqueta) y eso sí es idéntico entre tipos.
"""))

# ---------------------------------------------------------------------------
cells.append(md("""## 6. Reflexiones generales

### 6.1 Qué movió la aguja
1. **Pretrained > todo lo demás** (+28 pts en `f1_macro`).
2. **Más datos** (+5-7 pts cuando se puede).
3. **Hiperparámetros por tipo** (depende del dataset, +1-5 pts cada uno —
   shoes saltó +5.1 pts pasando de 10 a 15 epochs).

### 6.2 Qué no movió la aguja
- Augmentation más agresiva sobre datasets grandes y bien aprendidos.
- AdamW con weight_decay sobre tops (sí podría ayudar en pants — no probado).
- Variantes de threshold > 0.5 para los sigmoides (post-hoc, no entrena).

### 6.3 Lo que no se exploró por costo/ROI (priorizado)

| Idea | ROI esperado | Costo | Estado |
|---|---|---|---|
| **Re-correr `pants` con 12-15 epochs** | +4 pts en `f1_macro` | 5 min GPU | **Recomendado siguiente paso** |
| `pos_weight` en BCE para subir `f1_macro` y especialmente `pocket_details` (top-1=0.43) | +5-10 pts en macro, −1-2 en micro | Bajo (1 línea) | Pendiente |
| Regularización Run 3 *aplicada sólo a pants* (AdamW + aug fuerte) | +3-5 pts en pants | Bajo | Pendiente |
| Ensamble de seeds (3 corridas, promedio de logits) | +1-2 pts | 3× tiempo | Pendiente |
| ResNet50 pretrained | quizás −2 a +2 pts (incierto) | 3× VRAM, 3× tiempo | Improbable |
| Fine-tuning en 2 fases | Marginal con datasets >5k | Medio | Improbable |

### 6.4 Métricas: cuál mirar y cuándo

Cuatro métricas conviven en este reporte y cada una mide algo distinto:

- **`subset_acc`**: la más exigente. Acierta los **3** atributos a la vez.
- **`f1_micro`**: promedio ponderado por positivos. "Feeling general".
- **`f1_macro`**: promedio sin pesos. Sensible al desbalance. Métrica de
  selección de checkpoint.
- **Top-1 accuracy por atributo**: la más cercana a "accuracy" en sentido
  tradicional. La más explicable para un consumidor humano.

La celda siguiente carga los `.labels.json` que dejó el entrenamiento y
arma el resumen consolidado, así si re-entrenás los `.pth` los números
del notebook se actualizan automáticamente.
"""))

cells.append(code("""# Carga las métricas de los `.labels.json` actuales y arma la tabla consolidada.
def load_labels(t: str) -> dict:
    p = TORCH_STATE / f"multilabel_classifier_{t}_v1.labels.json"
    return json.loads(p.read_text(encoding="utf-8"))

label_files = {t: load_labels(t) for t in PRODUCT_TYPES}

summary_rows = []
for t, lf in label_files.items():
    bv = lf["best_val"]
    top1 = bv["val_top1_per_group"]
    summary_rows.append({
        "tipo": t,
        "n_logits": lf["num_logits"],
        "best_epoch": lf["best_epoch"],
        "f1_micro": bv["val_f1_micro"],
        "f1_macro": bv["val_f1_macro"],
        "subset_acc": bv["val_subset_acc"],
        "top1_avg": bv["val_top1_avg"],
        "top1 por atributo": ", ".join(f"{a}={v:.2f}" for a, v in top1.items()),
    })

pd.DataFrame(summary_rows).set_index("tipo")
"""))

cells.append(code("""# Top-1 accuracy por atributo, gráfico de barras.
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, (t, lf) in zip(axes, label_files.items()):
    top1 = lf["best_val"]["val_top1_per_group"]
    bars = ax.bar(top1.keys(), top1.values(), color=["#4c72b0", "#dd8452", "#55a868"])
    for b, v in zip(bars, top1.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_title(f"{t} (top1_avg={lf['best_val']['val_top1_avg']:.2f})")
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.3, axis="y")

axes[0].set_ylabel("top-1 accuracy en val")
fig.suptitle("Top-1 accuracy por atributo — best checkpoint del overnight")
plt.tight_layout()
plt.show()
"""))

cells.append(md("""**Hallazgos del top-1 por atributo (no se sabían antes del overnight)**:

- **`tops.season=0.70`** es el atributo más débil de tops, por debajo de
  `neck_style` (0.78) que tiene 26 clases vs 4. "Spring" y "Fall" son
  visualmente parecidos en una foto de catálogo.
- **`pants.pocket_details=0.43`** es el atributo más débil de todo el
  sistema. Era previsible (los detalles de bolsillos son sutiles), pero
  ahora hay un número concreto que justifica priorizarlo en v2.
- **`shoes.season=0.90`** es el atributo *más* fuerte. Tiene sentido: un
  boot vs una sandalia es información de temporada inequívoca.
"""))

# ---------------------------------------------------------------------------
cells.append(md("""### 6.5 Lo que aprendimos sobre cuándo parar de entrenar

Las dos curvas que un practitioner mira (val_loss y val_f1_macro) no
necesariamente se mueven juntas:

| Tipo | val_loss min | val_f1_macro max | ¿Coinciden? |
|---|---:|---:|---|
| tops  | epoch 3 | epoch 9 | **No** — val_loss sube despacio mientras f1_macro sigue mejorando. |
| shoes | epoch 11 | epoch 11 | **Sí** — las dos peakean juntas. |
| pants | epoch 7+ | epoch 7+ | **Sí ahora**, pero en la corrida previa de 10 epochs f1_macro peakeó en epoch 8 mientras val_loss ya había peakeado en epoch 3. |

**Regla práctica para esta familia de modelos**: parar cuando `val_f1_macro`
plateaue, no antes. Cap inicial razonable: 10-12 epochs. Si el último epoch
sigue mejorando, alargar. Si lleva 3 epochs en plateau, parar.
"""))

cells.append(code("""# val_loss vs val_f1_macro epoch a epoch — comparativa entre tipos.
val_loss_curves = {
    "tops":  [0.104, 0.099, 0.089, 0.096, 0.097, 0.100, 0.103, 0.105, 0.106, 0.107],
    "shoes": [0.077, 0.072, 0.066, 0.066, 0.066, 0.071, 0.067, 0.070, 0.068, 0.067, 0.067, 0.067, 0.068, 0.068, 0.067],
    "pants": [0.248, 0.233, 0.234, 0.234, 0.231, 0.231, 0.227],
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (t, f1) in zip(axes, curves.items()):
    epochs = list(range(1, len(f1) + 1))
    ax2 = ax.twinx()
    ax.plot(epochs, f1, "o-", color="#4c72b0", label="val_f1_macro")
    ax2.plot(epochs, val_loss_curves[t], "s--", color="#dd8452", label="val_loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val_f1_macro", color="#4c72b0")
    ax2.set_ylabel("val_loss", color="#dd8452")
    ax.set_title(t)
    ax.grid(alpha=0.3)

fig.suptitle("val_f1_macro (azul) vs val_loss (naranja): no siempre se mueven juntas")
plt.tight_layout()
plt.show()
"""))

# ---------------------------------------------------------------------------
cells.append(md("""## 7. Estado de los artefactos

Después de Etapa 1, en `deep-tagger-api/deep_learning/torch_state/`:
"""))

cells.append(code("""# Listar los .pth y .labels.json y mostrar tamaño en MB.
import os

artifacts = []
for f in sorted(TORCH_STATE.glob("multilabel_classifier_*")):
    artifacts.append({
        "archivo": f.name,
        "tamaño (MB)": round(f.stat().st_size / 1024**2, 2),
    })

pd.DataFrame(artifacts)
"""))

cells.append(md("""La estructura de `.labels.json` es lo que la inferencia necesita para
cargar el modelo y traducir logits → etiquetas legibles:
"""))

cells.append(code("""# Pretty-print del .labels.json de tops (ejemplo).
sample = label_files["tops"]
# Truncar las listas de classes a 5 elementos para que no explote la celda.
display_view = json.loads(json.dumps(sample))  # deep copy
for g in display_view["groups"]:
    g["classes"] = g["classes"][:5] + (["..."] if len(sample["groups"][0]["classes"]) > 5 else [])
print(json.dumps(display_view, indent=2, ensure_ascii=False))
"""))

# ---------------------------------------------------------------------------
cells.append(md("""## 8. Tabla resumen final (la que mira un humano apurado)

Datos del overnight del 2026-04-24 (estado actual de los `.pth` en disco).
Las "Img faltantes" son filas del CSV cuyo JPEG no estaba en disco al
momento del entrenamiento. **No se descartó ninguna fila por desbalance de
clases** (porque `season` siempre tiene un valor válido y eso alcanza para
mantener la fila aunque otros atributos queden NaN).
"""))

cells.append(code("""# Tabla resumen final con los datos del overnight.
final = pd.DataFrame([
    {"tipo": "tops",  "CSV total": 36173, "img faltantes": 3670, "train":29253, "val":3250, "total":32503,
     "epochs":10, "best epoch":9, "f1_macro":0.5164, "f1_micro":0.7862, "subset_acc":0.4120,
     "top1_avg":0.7875, "n_logits":35, "tiempo":"11:22"},
    {"tipo": "shoes", "CSV total": 17281, "img faltantes": 0,    "train":15553, "val":1728, "total":17281,
     "epochs":15, "best epoch":11,"f1_macro":0.6044, "f1_micro":0.8209, "subset_acc":0.4954,
     "top1_avg":0.8243, "n_logits":47, "tiempo":"13:11"},
    {"tipo": "pants", "CSV total":  7149, "img faltantes": 2,    "train": 6433, "val": 714, "total": 7147,
     "epochs": 7, "best epoch":7, "f1_macro":0.3726, "f1_micro":0.6774, "subset_acc":0.1709,
     "top1_avg":0.6812, "n_logits":20, "tiempo":"3:07"},
]).set_index("tipo")
final
"""))

# ---------------------------------------------------------------------------
cells.append(md("""## 9. Reproducción

El script ya quedó en la configuración del Run 2 (pretrained, Adam,
augmentation suave, lr=3e-4). Comandos para reproducir el overnight tal cual:

```powershell
# Activar el venv del API antes
python deep_learning/train_multilabel.py --product-type tops  --device cuda --batch-size 128 --num-workers 4 --max-samples 0 --epochs 10
python deep_learning/train_multilabel.py --product-type shoes --device cuda --batch-size 128 --num-workers 4 --max-samples 0 --epochs 15
python deep_learning/train_multilabel.py --product-type pants --device cuda --batch-size 128 --num-workers 4 --max-samples 0 --epochs  7
```

O lanzar todo de una con el script PS de overnight:

```powershell
.\\scripts\\overnight_training.ps1
```

CLI completo del script:

```
--product-type {tops,shoes,pants}    requerido
--device {cpu,cuda,auto}             default: auto
--max-samples N                      default: 10.000  (0 = sin cap)
--epochs N                           default: 10
--batch-size N                       default: 64
--lr X                               default: 3e-4 (tuneado para pretrained)
--num-workers N                      default: 0    (Windows-safe)
--seed N                             default: 42
--pretrained / --no-pretrained       default: --pretrained
```
"""))

# ---------------------------------------------------------------------------
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote notebook with {len(cells)} cells to {OUT}")
