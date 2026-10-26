# Datasheet abreviada

**Documento 3 de 4 del Hito 2**

---

**Equipo:** Jorge Federico Flores, Nicolás Velázquez, Nilayvi Barrientos, Rubén Ortigoza

**Muestra** (nombre y ruta en el repo): CSV etiquetados de `img-puller` (tops, shoes y pants), en [`data/muestra/`](../data/muestra/) de [rantezPeperino/UdeSA-computer-vision-plus](https://github.com/rantezPeperino/UdeSA-computer-vision-plus). Salen del prototipo [MatLock/UdeSA-computer-vision](https://github.com/MatLock/UdeSA-computer-vision)

**Fecha:** 2026-10-20

---

## 1. Motivación

**1. ¿Para qué se creó el dataset?**

Respuesta: Las etiquetas se armaron en Lily AI, una empresa de etiquetado de atributos de producto. El objetivo era tener ejemplos etiquetados a mano para automatizar esa misma tarea. En el prototipo se usaron para entrenar los clasificadores de atributos de tops, calzado y pantalones, y en este hito, para decidir si el proyecto es viable con los datos que tenemos.

**2. ¿Quién lo creó, y por encargo de quién?**

Respuesta: Lo etiquetaron expertos en moda de Lily AI, por encargo de sus empresas clientes, que son las dueñas de los productos y las fotos. En el dataset original, alrededor del 70 % de los productos son de Rue Gilt Groupe (RGG, dueña de Rue La La y Gilt) y el resto, de otros retailers como GAP y Foot Locker. En esta muestra, 60.602 de las 60.603 fotos se sirven desde `ruecdn.com`, el CDN de RGG ([EDA](2-eda-de-viabilidad.md), celda P5). Los CSV los aportó Jorge Federico Flores, integrante del equipo, con autorización para usarlos en la materia y publicarlos en este repo.

## 2. Composición

**3. ¿Qué representa cada instancia, y cuántas hay?**

Respuesta: Cada fila es una foto de un producto (una URL), con tres etiquetas: dos atributos propios de la categoría y `season`. Hay 60.603 filas: 36.173 tops, 17.281 calzados y 7.149 pantalones. El detalle por clase está en el EDA, Volumen (celda V1).

**4. ¿Es todo lo que existe o una muestra?**

Respuesta: Es todo lo que tiene el prototipo: los tres CSV completos, sin recortar. Respecto del dataset original de Lily AI es un extracto: solo tres categorías y solo las columnas de id, URL y etiquetas. La fecha, por ejemplo, quedó afuera.

**5. Si es una muestra, ¿de qué, y es representativa?**

Respuesta: De `img-puller/data/` del prototipo, en el commit `b9b3b2b`. Es una copia exacta, verificable por el SHA-256 de cada archivo ([`origen.csv`](../data/muestra/origen.csv)). Respecto del dataset original no es representativa en origen: allá RGG es alrededor del 70 % y acá es prácticamente el 100 %. Sobre la imagen medimos un subconjunto de 300 fotos (100 por categoría, estratificadas por `season`), así que su composición de temporadas reproduce la de cada categoría.

**6. ¿Falta información?**

Respuesta: Sí. Los valores que hay están completos, pero faltan columnas enteras: tipo de prenda, color, material, título, descripción y fecha. Ver EDA, Completitud (celda C2).

**7. ¿Hay errores, ruido o duplicados?**

Respuesta: Sí. Los principales:

- 166 fotos etiquetadas a la vez como top y como pantalón.
- Un encabezado repetido en `shoes_tags.csv`.
- 195 filas con clases que el modelo no reconoce.
- Valores que mezclan `_` y `-`.
- En las fotos con dos etiquetas, `season` coincide solo en el 77,1 %.

Ver EDA, Corrección (celda C3) y Confiabilidad (celda R6).

**8. ¿Depende de recursos externos, con licencias o costos?**

Respuesta: Sí. Las imágenes no están en el repo: se bajan del CDN de RGG, que hoy responde para el 100 % de las URLs chequeadas, pero puede dejar de hacerlo. Los productos y las fotos son de las empresas clientes de Lily AI; el uso en la materia está autorizado, un uso comercial no. El acceso a la fuente está en el [mapa de fuentes](1-mapa-de-fuentes.md), fuente 2.

**9. Si hay personas, ¿se puede identificar a alguien, directa o indirectamente?**

Respuesta: Hay fotos con modelos. En las que miramos, la cara suele aparecer recortada, pero no medimos en cuántas se ve el rostro entero. Los CSV no traen nombres ni datos de las personas.

**10. ¿Hay datos sensibles?**

Respuesta: Los CSV no tienen datos de clientes ni de compradores. Los rostros de modelos, cuando son identificables, son datos personales (Ley 25.326). No hay datos sensibles en el sentido de esa ley (salud, origen étnico, opiniones).

**11. ¿Cómo se consiguió cada dato, y en qué período?**

Respuesta: Las etiquetas las pusieron a mano expertos en moda de Lily AI, con mucha revisión humana en el circuito (*human in the loop*). Los productos se publicaron entre junio y julio de 2025; el dataset original traía esa fecha, pero el extracto no. La fecha de los archivos en el CDN es coherente con eso: el 99,4 % es de 2021 en adelante (EDA, Vigencia, celda T4). Copiamos los CSV del repo del prototipo el 2026-10-20 con `scripts/armar_muestra.py`.

**12. Si es una muestra, ¿con qué criterio se eligió?**

Respuesta: Los CSV no se muestrearon: están completos. Las mediciones sobre imagen usan muestras con semilla 42:

- **Chequeo de URLs:** 500 al azar por categoría.
- **Imágenes descargadas:** 100 por categoría, estratificadas por `season`.
- **Re-etiquetado:** 20 por categoría, tomadas de las descargadas.

Ver `data/muestra/README.md`.

**13. Si hay personas, ¿sabían que se recolectaban sus datos, y lo consintieron?**

Respuesta: Las fotos son de catálogo y las produjeron los retailers para vender sus productos. Si las modelos consintieron que su imagen se use para entrenar un modelo: **No sabemos**.

**14. ¿Hay algo en cómo se armó que afecte sus usos?**

Respuesta: Sí:

- **Fotos repetidas entre categorías:** las 166 fotos que están en tops y en pantalones obligan a partir los datos por foto y no por fila.
- **`season` depende del rango de `id`:** el conjunto de evaluación tiene que separarse por rango de `id`.
- **Color en la URL:** 58 URLs traen el color en el nombre del archivo, así que la URL no puede entrar como feature.
- **Taxonomía recortada:** la del modelo descarta las clases con menos de 20 ejemplos.

Ver EDA, sección 3, celdas L1 a L5.

**15. ¿Para qué tareas no debería usarse?**

Respuesta:

- Para publicar fichas sin revisión humana.
- Para inferir color, tipo de prenda, material o talle: la muestra no los trae.
- Para estimar cómo anda el producto con marcas argentinas, porque todas las fotos son de un retailer de EE. UU.
- Para vestidos, abrigos o accesorios.
- Para reconocer o identificar a las modelos.
- Para un uso comercial: los datos son de las empresas clientes de Lily AI.
