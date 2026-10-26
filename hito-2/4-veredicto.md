# Veredicto de viabilidad

**Documento 4 de 4 del Hito 2**

---

**Equipo:** Jorge Federico Flores, Nicolás Velázquez, Nilayvi Barrientos, Rubén Ortigoza

**Datos sobre los que se decidió** (reales, sintéticos o los dos): reales. Los CSV etiquetados de `img-puller` del prototipo [MatLock/UdeSA-computer-vision](https://github.com/MatLock/UdeSA-computer-vision), en [`data/muestra/`](../data/muestra/). No usamos datos sintéticos.

---

## 1. Decisión

Decisión: **Ajustar el alcance.**

## 2. Fundamentos

Fundamentos:

1. **No cumple Completitud, el criterio que más pesó.** La muestra no trae color ni tipo de prenda, los dos atributos que el canvas prioriza porque generan más reclamos (EDA, Completitud, celda C2). Lo que sí trae está completo: 100 % de valores presentes y 100 % de las URLs abren.
2. **No cumple Representatividad.** 60.602 de 60.603 fotos son de un solo retailer de EE. UU., no hay vestidos ni abrigos (0 %), y el dato de marcas argentinas no se consiguió (EDA, Representatividad, celda P5; mapa de fuentes, fuente 1).
3. **Cumple Volumen para los atributos que la muestra trae.** Hay 60.603 filas, y 80 de las 135 clases llegan a 100 ejemplos, con entre el 96,6 % y el 100 % de las filas de cada atributo (EDA, Volumen, celda V1). Alcanza para una PoC sobre esos atributos, no sobre el producto completo.

## 3. Exposición al riesgo

| Propiedad estimada | Riesgo (cifra) | Procedencia de la cifra | Qué haría falta para medirla |
|---|---|---|---|
| Confiabilidad | 0,4 × 10 pp = **4 pp** más de campos a corregir | Probabilidad *(H)* 0,4 de que el error de etiqueta supere el 10 %. En contra: en las 166 fotos con dos etiquetas, `season` coincide solo en el 77,1 % (celda R6). A favor: las etiquetas las pusieron a mano expertos en moda, con revisión humana (datasheet, pregunta 11). Impacto *(H)* 10 pp: un modelo que aprende el ruido lo devuelve como error, sobre el margen de 20 % de campos corregidos del canvas (sección 6) | Que dos personas del equipo re-etiqueten por separado las 60 imágenes de `data/etiquetado/` y comparar contra la etiqueta original (celda R6) |
| Representatividad (usuarios) | 0,7 × 10 pp = **7 pp** menos de acierto con fotos de marcas argentinas | Probabilidad *(H)* 0,7: todas las fotos son de un mismo retailer, con el mismo formato de 1056×1320 px (celdas C3 y P5), y el canvas (sección 7) anticipa que el sistema anda peor con fotos distintas. Impacto *(H)*: una caída de 10 pp, la mitad del margen del canvas | Etiquetar 100 fotos de 3 a 5 marcas argentinas y comparar el acierto del modelo contra el que da sobre esta muestra |
| Vigencia | 0,05 × 5 pp = **0,25 pp** | Probabilidad *(H)* 0,05 de que los productos sean anteriores a 2021: el origen declara que se publicaron entre junio y julio de 2025, y el 99,4 % de los archivos del CDN es de 2021 en adelante (celda T4). Impacto *(H)* 5 pp: los cortes cambian, pero cuello, bolsillos y calce cambian poco | Incluir en el extracto la columna de fecha que trae el dataset original |

## 4. Consecuencias de la decisión

### Plan del pipeline

| | Plan |
|---|---|
| Obtención y conjunto de evaluación | Los tres CSV fijados al commit `b9b3b2b` y sus imágenes, bajadas una sola vez y guardadas fuera del repo. Antes de tocar features se separa y se congela un 15 % para evaluación, con semilla 42 y en `data/splits/eval_ids.csv`. Las reglas: (1) se parte por el rango de `id` dentro de cada categoría, porque `season` depende del `id` (V de Cramér de hasta 0,423, celda L2); (2) la clave de grupo es el número de imagen de la URL, para que las 166 fotos de tops y pants queden del mismo lado (celdas L3 y L4); (3) se buscan duplicados antes y después de partir. Ningún caso de evaluación puede aparecer en los ejemplos de un prompt ni en los documentos que consulte el sistema |
| Limpieza y transformaciones | Se saca la columna `relative_path` repetida y se normalizan los separadores de las etiquetas. Las 55 clases con menos de 100 ejemplos se agrupan en "otro" (celda V1). Se descartan las URLs que no respondan al bajar y se lleva cada imagen a 224×224. `id`, `relative_path`, `image_url` y cualquier texto generado con `describer.py` no entran como features (celda L1). Escalar, imputar y sobremuestrear se hace después de partir |
| Estructura final | Una tabla por imagen con `id`, `categoria`, `ruta_imagen`, `grupo` (número de imagen), una columna por atributo normalizado y `split` |

### Alcance nuevo

Alcance nuevo:

- **Entra en la PoC:** los atributos que la muestra sostiene con volumen. Para tops, cuello y calce; para calzado, silueta y ocasión; para pantalones, bolsillos y calce. `season` también entra, pero es secundario porque su etiqueta es la menos consistente.
- **Queda afuera hasta tener otra fuente:**
  - Color y tipo de prenda: sin verificar con esta muestra. Candidata: Fashion Product Images, fuente 3 del mapa.
  - Vestidos, abrigos y accesorios.
  - Material y talle.
  - Títulos y descripciones: no hay texto orgánico.
  - Cualquier afirmación sobre cómo anda el producto con marcas argentinas.
