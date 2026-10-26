# Definición de la PoC - Deep Tagger

**Documento del Hito 3 · Jorge Federico Flores, Nicolás Velázquez, Nilayvi Barrientos, Rubén Ortigoza**

*(H)*: hipótesis o decisión nuestra, como en el [canvas](canvas.md). Celdas: [notebook del Hito 2](notebooks/eda_viabilidad.ipynb).

## Historial de versiones

| Versión | Fecha | Qué cambió |
|---|---|---|
| 1, Hito 3 | 2026-10-26 | Primera versión: propuesta, spec, diseño y plan |

El documento vive entero en este archivo.

---

## 1. Propuesta

### 1.1 Overview

Quien carga el catálogo (canvas, sección 1) sube la foto de un top, un calzado o un pantalón y elige la categoría. La PoC propone los atributos que la muestra trae etiquetados: cuello y calce en tops, silueta y ocasión en calzado, bolsillos y calce en pantalones, y temporada en los tres. Cada valor sale con su confianza, y los dudosos se marcan para revisar. Los datos son reales: las 60.603 filas de `img-puller`. El veredicto del Hito 2 fue **ajustar el alcance**, y este es el alcance nuevo: 80 de las 135 clases llegan a 100 ejemplos y cubren entre el 96,6 % y el 100 % de las filas de cada atributo (celda V1), pero no hay color ni tipo de prenda (celda C2). Fundamentos: [veredicto](hito-2/4-veredicto.md), sección 2. Queda afuera el resto de la ficha y el camino a Google Shopping (1.3).

### 1.2 Objetivo

1. **Supuesto principal** (canvas, sección 3, Acción; riesgo de valor): la confianza por campo permite revisar solo lo dudoso. Lo valida que, sobre el conjunto congelado, los campos no marcados acierten ≥ 90 % y los marcados sean ≤ 30 % (2.3).
2. **Supuesto secundario** (canvas, sección 3, refutación; riesgo de factibilidad): un modelo propio aporta algo frente a un generalista con un prompt. Lo valida que quede a no más de 2 pp de macro-F1 del *baseline* y sea 10 veces más rápido (2.3).
3. **Decisión que habilita:**
   - **Seguir** si se cumplen los dos: conectar la salida con la etapa 1 del pipeline de Google y conseguir fotos de marcas argentinas.
   - **Cambiar el enfoque** si se cumple el principal y gana el *baseline*: el generalista pasa a ser el motor.
   - **Parar y replantear** si el propio no llega al principal.

### 1.3 Fuera de alcance

- **Color y tipo de prenda** (K-Means y TinyVGG): la muestra no los trae (celda C2). La categoría la elige la persona.
- **Título y descripción** (BLIP y Claude): no hay texto orgánico para medirlos (canvas, sección 5).
- **Material, género y talle:** los declara la marca.
- **Vestidos, abrigos y accesorios:** 0 % de la muestra (celda P5).
- **Control de coherencia foto vs. ficha:** depende de que la confianza funcione, que es lo que prueba esta PoC.
- **Feed, publicación y campañas de Google** (`positioner/`): ya probados aparte, no dependen del modelo.
- **Temporada como criterio:** su etiqueta coincide solo en el 77,1 % (celda R6). Se muestra igual.
- **`baseline_v1` y CLIP** (3.1.1): los pesos vieron el congelado; CLIP no entra en tres semanas.
- **Detector aparte de prendas fuera de categoría** (3.1.3): alcanza la confianza baja.
- **Reentrenar con las correcciones:** solo se registran.
- **Escalabilidad, usuarios y despliegue** (3.1.5): la demo corre en una notebook.

### 1.4 Riesgos

**Representatividad.** Del veredicto (sección 3): 7 pp menos de acierto con fotos de marcas argentinas, porque 60.602 de 60.603 fotos son de un retailer de EE. UU. (celda P5). La PoC puede cumplir 2.3 y fallar con una foto de celular. *Plan B:* en la semana 7 sacamos 30 fotos de ropa propia, dos integrantes las etiquetan y en la 9 reportamos aparte su exactitud de no marcados. Si cae más de 10 pp, seguir queda atado a conseguir fotos de marcas reales.

**Confiabilidad de las etiquetas.** Del veredicto (sección 3): 4 pp más de campos a corregir; lo único medido es el 77,1 % de coincidencia de `season` (celda R6). Con más de 10 % de etiquetas erróneas, el 90 % de 2.3 es inalcanzable. *Plan B:* en la semana 7, dos integrantes re-etiquetan por separado las 60 imágenes de `data/etiquetado/`. Si la coincidencia con la etiqueta original da menos de 90 %, esa cifra reemplaza al 90 % de 2.3.

**Confianza mal calibrada.** De la decisión 3.1.1: las redes suelen ser sobreconfiadas, más con una prenda fuera de categoría, y entonces "revisar" no separa lo dudoso. *Plan B:* como los umbrales se eligen en validación, se ve en la semana 8. Si hay que marcar más del 30 % para llegar al 90 %, o las prendas fuera de categoría salen sin marcar, sumamos a cada modelo la clase "no es de esta categoría", entrenada con fotos de las otras dos.

---

## 2. Spec

### 2.1 Escenarios

#### Happy path

- **Dado** que quien carga tiene la foto de un top de catálogo y la demo abierta
- **Cuando** sube la foto o pega su URL y elige "tops"
- **Entonces** el sistema devuelve cuello, calce y temporada, cada uno con su valor y su confianza
- **Y** marca "revisar" cada campo con confianza debajo del umbral de su atributo
- **Y** la persona confirma o cambia los marcados con un desplegable de valores válidos
- **Y** al guardar, el sistema escribe en `logs/correcciones.csv` una fila por campo: imagen, campo, sugerido, confianza, final, si cambió y hora

#### Casos borde

##### 2.1.1

- **Dado** que la persona eligió "tops"
- **Cuando** sube la foto de un vestido o de una campera
- **Entonces** todos los campos quedan marcados "revisar" y aparece el aviso "Esta foto no parece de la categoría elegida"
- **Le cuesta al usuario:** una ficha inventada, con apariencia de segura, para una prenda que el sistema no conoce. Es el peor caso borde.

##### 2.1.2

- **Dado** que la persona eligió la categoría correcta
- **Cuando** sube una foto de celular, con fondo de casa y sin modelo
- **Entonces** cada campo no marcado coincide con el valor esperado
- **Le cuesta al usuario:** atributos equivocados que se publican sin revisar.

##### 2.1.3

- **Dado** que la persona eligió "pantalones"
- **Cuando** sube una foto de cuerpo entero con top y pantalón
- **Entonces** devuelve solo atributos de pantalones, y cada campo no marcado coincide con el esperado
- **Le cuesta al usuario:** el calce del top cargado como calce del pantalón.

##### 2.1.4

- **Dado** que la demo está abierta
- **Cuando** la persona pega una URL que no responde, sube un archivo que no es imagen o una imagen de menos de 64 px de lado
- **Entonces** el sistema no devuelve atributos y dice cuál de los tres problemas encontró
- **Le cuesta al usuario:** tiempo perdido si no sabe qué pasó.

#### Error del sistema

- **Dado** que la persona subió una imagen válida y eligió la categoría
- **Cuando** el modelo de esa categoría no carga o falla
- **Entonces** el sistema muestra "No pude sugerir esta ficha" con el motivo y deja los campos vacíos con sus desplegables
- **Y** la persona completa la ficha a mano y la guarda
- **Y** el caso queda en `logs/errores.csv`

### 2.2 User experience

1. **Interacción y control.** La persona sube una foto o pega una URL y elige la categoría. El sistema sugiere y nunca guarda solo: guarda la persona. Cualquier campo se puede cambiar, esté marcado o no.

2. **Señal de confianza.** Cada campo muestra su valor y su confianza en porcentaje; los dudosos llevan "revisar" y van primero. La confianza es la probabilidad calibrada de la clase elegida (3.1.1). El umbral de cada atributo es el más bajo con el que, en validación, los no marcados aciertan al menos 90 %. Un "otro" (clases con menos de 100 ejemplos) siempre se marca.

3. **Corrección o aviso.** Cada campo tiene un desplegable con los valores válidos de su atributo. Al guardar, todo queda en `logs/correcciones.csv`, el registro que el canvas (sección 5) pide.

4. **Peor caso borde** (2.1.1). La persona ve arriba de la ficha el aviso "Esta foto no parece de la categoría elegida", con todo marcado. Puede elegir otra categoría, que vuelve a correr la sugerencia, o completar a mano.

### 2.3 Criterios de aceptación

| Criterio | Métrica | Umbral | Conjunto de evaluación | Cómo se chequea |
|---|---|---|---|---|
| Calidad de la respuesta | En los seis atributos, sin temporada: (a) exactitud de los campos no marcados; (b) proporción de campos marcados | (a) ≥ 90 %, del nivel de Confiabilidad del EDA, o la cifra del plan B de 1.4. (b) ≤ 30 % *(H)*: revisar 3 de cada 10 campos es menos que revisarlos todos (canvas, sección 1). Un error no marcado cuesta más que una marca de más: (a) va primero | El congelado, `data/splits/eval_ids.csv`: 15 % de cada categoría, unas 9.090 filas | `python eval/run_eval.py` imprime (a), (b), el detalle por atributo y `PASS` o `FAIL` |
| Latencia (p95) | p95 de `POST /predict-image`, con la imagen en disco, en la CPU de la notebook de la demo | ≤ 2 s *(H)*: las 80 prendas del ejemplo del canvas en menos de 3 minutos | 200 filas de `eval_baseline_ids.csv` | `run_eval.py --latencia` imprime p95 y `PASS` o `FAIL` |
| Cobertura de los casos borde | Imágenes de cada caso en que el sistema hace lo que dice su escenario | 2.1.4 y error del sistema: 100 %. 2.1.1 a 2.1.3: ≥ 4 de 5 *(H)* | `eval/casos_borde.csv`: 5 imágenes por caso con su resultado esperado, fuera del congelado (3.2). El error del sistema se provoca renombrando los pesos | `python eval/run_casos_borde.py` da `PASS` o `FAIL` por imagen |
| Ventaja sobre el *baseline* | Macro-F1 promedio de los seis atributos y p95 de latencia | Macro-F1 propio ≥ el del *baseline* − 2 pp **y** p95 propio ≤ p95 del *baseline* / 10 *(H)*: el canvas (sección 5) pide ganar en costo o velocidad | `eval_baseline_ids.csv`: 1.000 filas del congelado, proporcionales por categoría, semilla 42 | `run_eval.py --comparar-baseline` imprime ambos y `PASS` o `FAIL` |

---

## 3. Diseño

### 3.1 Decisiones de diseño

No probamos ninguna alternativa todavía.

#### 3.1.1 El modelo

**Contexto.** Hay que predecir una clase por atributo a partir de una foto, con una confianza que decida qué se revisa. Hay 60.603 filas reales, un script que entrena ResNet18 (`train_multilabel.py`) y tres semanas. Los pesos actuales vieron filas del futuro congelado, porque el script parte todo al azar (celda L5). Con ese *leakage*, su acierto top-1 en validación fue de 78 % en cuello, 88 % en calce de tops, 80 % en ocasión, 76 % en silueta, 44 % en bolsillos y 77 % en calce de pantalones (`labels.json`), solo como referencia.

**Alternativas evaluadas.**

1. ***Baseline:* modelo generalista con un prompt.** Claude (`claude-sonnet-5-5`) recibe foto y categoría y devuelve un JSON con un valor válido por atributo. Es lo más simple que podría funcionar y la comparación que pide el canvas (sección 3). El prompt no lleva ejemplos, así que ningún caso de evaluación entra en él. Costo y latencia: no sabemos. Su confianza no es una probabilidad calibrada, así que no puede probar el supuesto principal. **Sigue como *baseline*.**
2. **ResNet18 reentrenado por categoría, con una cabeza por atributo.** Parte de ImageNet, se entrena sin el congelado y se calibra con *temperature scaling* en validación. El código existe, y `baseline_v1` se entrenó en una tarde en una GPU de notebook (`labels.json`). Riesgos: sobreajuste a un retailer y sobreconfianza. **Se construye.**
3. **CLIP con una capa lineal.** Podría aguantar mejor fotos distintas, pero es una dependencia nueva. **Se descarta:** no entra en tres semanas con el *baseline*. La clase mayoritaria queda como piso para probar los scripts.

**Decisión.** La 2, medida contra la 1: el supuesto principal necesita una confianza calibrada por campo, y la 2 la da directamente.

**Consecuencias.** Se asume que la calibración alcanza para que "revisar" separe lo dudoso (1.4, riesgo 3). Se resignan `baseline_v1` y CLIP (1.3).

#### 3.1.2 Los datos

**Contexto.** El plan del pipeline del veredicto (sección 4) fija obtención, congelado, limpieza y estructura final. Falta decidir entrenamiento, validación y cómo llegan las imágenes.

**Alternativas evaluadas.**

1. **Todas las filas fuera del congelado, con las imágenes bajadas una vez a 224×224.** Usa todo el volumen y no depende de la red al entrenar. **Se elige.**
2. **Submuestreo balanceado por clase.** Corrige el desbalance, pero tira casi todo tops, donde `Relaxed-Fit` tiene 26.551 filas (celda V1). **Se descarta:** alcanza con "otro" y pesos por clase.
3. **Descarga en cada época.** No ocupa disco, pero depende de un CDN de terceros que puede dejar de responder (datasheet, pregunta 8). **Se descarta.**

**Decisión.** La 1. Del veredicto se mantiene todo: congelado del 15 % con semilla 42, por rango de `id` y agrupado por número de imagen (celdas L2 a L4); clases con menos de 100 ejemplos en "otro" (celda V1); imágenes a 224×224. Cambian tres cosas: entrenamiento y validación (85/15, por grupo) salen solo de fuera del congelado; los duplicados (celda L3) se buscan otra vez entre congelado y entrenamiento; y calibración y umbrales se eligen en validación, nunca en evaluación.

**Consecuencias.** Se asume que las etiquetas tienen menos de 10 % de error (riesgo 2) y que lo aprendido con fotos de un retailer sirve para otras (riesgo 1). Se resigna temporada como criterio (1.3).

#### 3.1.3 Workflow o agente

**Contexto.** Los pasos se conocen de antemano.

**Alternativas evaluadas.**

1. **Workflow fijo:** validar la imagen → modelo de la categoría → calibración → umbral por atributo → aviso si todo quedó marcado. Cada paso se testea solo. **Se elige.**
2. **Un agente** que decide qué hacer (adivinar la categoría, volver a preguntar). Agrega variabilidad y no pone a prueba ningún supuesto. **Se descarta.**

**Decisión.** La 1: un agente no prueba nada que importe acá. Una sola llamada al generalista es el *baseline* (3.1.1).

**Consecuencias.** Se asume que una prenda fuera de categoría da confianza baja en todos sus atributos (riesgo 3). Se resigna un detector aparte (1.3).

#### 3.1.4 Construir o usar hecho

**Contexto.** El repo ya tiene la API, la UI y el script de entrenamiento. Somos cuatro, con tres semanas.

**Alternativas evaluadas.**

1. **Extender lo que existe.** La UI ya habla con la API y se parece al producto. **Se elige.**
2. **Una demo nueva en Gradio.** Se arma más rápido, pero hay que rehacer la conexión con el modelo. **Se descarta.**
3. **Un servicio comercial de etiquetado** (canvas, sección 4). Su taxonomía no es la de la muestra. **Se descarta.**

**Decisión.** La 1. Se construye solo lo que falta (tabla de 3.2).

**Consecuencias.** Se asume que la confianza que muestra la UI es la calibrada (riesgo 3). Se resignan del flujo TinyVGG, K-Means, BLIP y la descripción con Claude (1.3).

#### 3.1.5 Requerimientos no funcionales

**Contexto.** La PoC se valida una vez, en la semana 9, con una persona usando la demo.

**Alternativas evaluadas.**

1. **Reproducibilidad y latencia en una notebook:** semillas fijas, pesos y prompt versionados, evaluación con un comando, modelo en CPU. **Se elige.**
2. **Calidad con modelos más grandes en GPU.** La demo dependería de una GPU y no entra en tres semanas. **Se descarta.**
3. **Escalabilidad con un servicio desplegado.** Hay un solo usuario. **Se descarta.**

**Decisión.** La 1. Confiable: `run_eval.py` da siempre los mismos números. Latencia: la de 2.3. Mantenible: un README para levantar la demo y evaluar.

**Consecuencias.** Se asume que ResNet18 responde en CPU dentro de 2.3; no es un riesgo, porque el prototipo ya lo corre en CPU. Se resignan escalabilidad, usuarios y despliegue (1.3).

### 3.2 Arquitectura

```
[ingesta] → [procesamiento] → [modelo: ResNet18 → calibración → umbral] → [interfaz]
                                   │   baseline: Claude, en paralelo
                              [evaluación]
```

| Pieza | Qué es en este proyecto | Se construye o se usa hecha | Con qué (API, modelo preentrenado, librería) |
|---|---|---|---|
| Ingesta | Cambia: `armar_split.py` escribe el congelado, `eval_baseline_ids.csv` y la partición 85/15; `cachear_imagenes.py` guarda las imágenes a 224×224 fuera del repo | Se construye | pandas, `requests`, Pillow |
| Procesamiento | Cambia: se suma el chequeo de duplicados entre congelado y entrenamiento | Se construye | pandas, `imagehash`, torchvision |
| Modelo o agente | ResNet18 por categoría, una cabeza por atributo, calibrado; el *baseline* en paralelo | Entrenamiento y calibración se construyen; red y *baseline*, hechos | `torchvision.models.resnet18` con pesos de ImageNet, PyTorch; API de Claude (`claude-sonnet-5-5`) |
| Interfaz | Foto o URL, categoría, confianza, "revisar", desplegables y Guardar | Se extiende lo existente | `deep-tagger-ui` (React, MUI), `deep-tagger-api` (FastAPI) |
| Evaluación | Cambia: `run_eval.py` y `run_casos_borde.py`. Casos borde de fuera de la muestra: Fashion Product Images (2.1.1), fotos del equipo (2.1.2), DeepFashion-MultiModal (2.1.3) y archivos armados (2.1.4) | Se construye | scikit-learn, numpy |

Herramientas del agente: no aplica, el sistema es un workflow fijo (3.1.3).

---

## 4. Plan

### 4.1 Plan de construcción

| Semana | Meta | Qué se construye | Quién | Qué se muestra al cierre |
|---|---|---|---|---|
| 7, Workshop 1 | Datos listos | Split y congelado commiteados; imágenes en disco; re-etiquetado y fotos propias (1.4); casos borde; `run_eval.py` con la clase mayoritaria | Jorge (split e imágenes); Nilayvi y Rubén (re-etiquetado, fotos, casos borde); Nicolás (`run_eval.py`) | Conteos por split, cero duplicados cruzados, coincidencia del re-etiquetado y `run_eval.py` corriendo con el piso |
| 8, Workshop 2 | *Baseline* de punta a punta | *Baseline* en la API y la UI, corrido sobre `eval_baseline_ids.csv`; ResNet18 reentrenado y calibrado; umbrales | Nilayvi (*baseline*); Nicolás (entrenamiento); Rubén (API y UI) | Demo con el *baseline*, sus métricas y costo; (a) y (b) del propio en validación |
| 9, Workshop 3 | Evaluación contra los criterios de aceptación | Modelo propio en la API; confianza, "revisar", desplegables, aviso y registro en la UI; casos borde y `run_casos_borde.py` | Rubén (UI); Nicolás (evaluación); Jorge (casos borde); Nilayvi (comparación) | `PASS` o `FAIL` en cada criterio de 2.3, el diagnóstico con fotos propias y la decisión de 1.2 |

Si perdemos una semana, sale primero la UI nueva: 2.3 se mide por la API y la demo usa la UI actual.

---

*Basado en el material de las Clases 5 y 6.*
