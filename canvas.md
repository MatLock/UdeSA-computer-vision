# AI Opportunity Canvas

> **Equipo:** Deep Tagger
> **Integrantes:** Jorge Federico Flores, Nicolás Velázquez, Nilayvi Barrientos, Rubén Ortigoza
> **Caso:** Carga de fichas de producto y posicionamiento en Google Shopping para e-commerce de moda (lado empresa, no consumidor final)
> **Repositorio del proyecto:** [rantezPeperino/UdeSA-computer-vision-plus](https://github.com/rantezPeperino/UdeSA-computer-vision-plus), fork del prototipo [MatLock/UdeSA-computer-vision](https://github.com/MatLock/UdeSA-computer-vision). Cuando este documento dice "el repo", se refiere al fork. Cuando dice "el prototipo", se refiere a lo que ya estaba en el proyecto original: `deep-tagger-api`, `deep-tagger-ui` e `img-puller`.
> **Versión:** 2 · **Fecha:** 2026-10-26

Convención de este documento: lo marcado como *(H)* es hipótesis nuestra, todavía sin verificar. Lo que no tiene marca tiene una fuente al final o una referencia a un documento del repo. Si algo dice "no sabemos", es que no sabemos.

---

## 1. Problema y contexto

### A quién le pasa

Quien carga el catálogo en una marca de indumentaria argentina chica o mediana: la dueña, o la persona de e-commerce (equipos de 1 a 5). Le pasa cada vez que entra una colección o una reposición y hay que publicar un lote de prendas en la tienda propia y en los lugares donde busca la clienta: un marketplace o Google. Cuántas prendas por lote es un dato que no tenemos *(H: entre decenas y unos pocos cientos)*.

Tamaño del grupo, como proxy: Tiendanube dice tener más de 60.000 tiendas en Argentina (todos los rubros, no solo moda) y la CACE midió 25,1 millones de compradores online en 2025. Cuántas de esas tiendas son de indumentaria, y cuántas anuncian en Google, no sabemos.

### Qué le pasa

El progreso que busca: tener las prendas online rápido, bien descriptas y encontrables por quien las busca, sin prometer algo que la prenda no es y sin gastar en publicidad que no vende.

- Funcional: publicar cada prenda con la ficha completa y consistente (categoría, color, material, talle, título, descripción) en cada canal donde vende, sin pasar horas, y saber qué publicaciones funcionan y cuáles no.
- Social: que la marca se vea prolija frente a sus clientas y frente a la competencia, y que nadie escriba "no era como en la foto".
- Emocional: sacarse de encima una tarea repetitiva; el miedo a haber puesto mal un color o una tela y comerse el reclamo después; y la incomodidad de pagar publicidad sin entender si sirve.

Circunstancia: llega la mercadería, hay fotos, hay una fecha de lanzamiento, y quien carga está sola o con poca gente, en medio de otras tareas (atender consultas, armar pedidos, subir a redes). Una vez publicadas las prendas, mira de reojo cuánto se gasta en anuncios.

Cómo lo diría alguien en esa situación *(texto armado por nosotros como ejemplo, no es un testimonio real)*:

> "Me llegan 80 prendas nuevas un jueves y las quiero online el lunes. Cada una tiene que tener su color, su tela, su categoría, un título y una descripción que no sea copiar la de la remera anterior. Para que aparezcan en Google me piden todavía más datos, y si falta uno me rechazan la prenda y no entiendo por qué. Después pago anuncios y no sé cuáles se ven y cuáles no le interesan a nadie."

Obstáculos: volumen repetitivo; cada canal pide campos y categorías distintas (Google, por ejemplo, pide su propia taxonomía de categorías y, para indumentaria, género, edad, talle y color); hay atributos que se ven en la foto y otros que solo se saben con la etiqueta en la mano; quien carga cambia de una temporada a otra y cada uno describe distinto ("buzo", "hoodie", "canguro"); entender qué anuncio rinde lleva tiempo y conocimiento que una marca chica no suele tener.

Cómo se las arregla hoy: planilla y copiar/pegar, plantillas de descripción, copiar lo que manda el proveedor, delegar en un pasante o un freelance, publicar con la ficha mínima (foto y precio), y para la publicidad, contratar una agencia o no anunciar. Cuánta gente hace cada cosa no lo sabemos.

Calidad para esta persona: que la ficha salga casi hecha y que lo dudoso esté señalado; que lo que manda a Google no vuelva rechazado; y que las publicaciones que no funcionan aparezcan marcadas, sin que nada se pause o se gaste sin que ella lo sepa. Aceptaría revisar algunos campos, pero no revisarlos todos.

### Cuánto le cuesta

No tenemos una medida propia. Lo que hay son proxies, todos de fuentes secundarias y ninguno de Argentina:

- Tiempo por prenda cargada: **no lo sabemos**. Hay que cronometrarlo (ver sección 6).
- Devoluciones: en moda online, entre el 10% y el 20% de las devoluciones se atribuyen a que el artículo no coincide con la foto o con la descripción (10% según un desglose atribuido a DHL; 16% y 20% según otras dos fuentes).
- Fuera de nuestro alcance, pero importante para dimensionar: talle y calce pesan más (26% a 40% de las devoluciones de moda según las mismas fuentes). Ver sección 3.
- Gasto en publicidad y prendas rechazadas por Google: **no lo sabemos**. No tenemos datos de cuánto invierte una marca chica en anuncios ni de cuántas publicaciones le rechazan.

### Evidencia

#### De confirmación

- Las devoluciones por diferencia entre lo publicado y lo recibido existen y están medidas por terceros (fuentes 1 y 2), aunque no en Argentina.
- El e-commerce argentino es grande y sigue creciendo: facturó $34 billones en 2025 (+55% nominal) y vendió 645 millones de unidades (fuente 3).
- Google Merchant Center admite Argentina como país de destino, con precios en pesos (fuente 14). El canal existe para el usuario que describimos.
- *(H)* Pendiente: entrevistar a 8 o 10 dueños o encargados de e-commerce de marcas argentinas y preguntar cómo cargan, cuánto tardan y si anuncian en Google. Todavía no lo hicimos.

#### De refutación

- Según la CACE, en 2025 indumentaria salió de los primeros lugares en facturación y unidades (fuente 4). El rubro pesa menos de lo que suponíamos.
- Los porcentajes de devolución son de España/EE.UU./blogs de logística. Puede que en Argentina el motivo principal sea otro.
- Puede que las marcas chicas vendan sobre todo por Instagram y WhatsApp y que Google no sea un canal que usen. Si es así, la parte de posicionamiento le habla a un grupo más chico que el de la carga.
- Lo que refutaría el problema: que las marcas entrevistadas carguen el catálogo una vez por temporada, les lleve poco, y no lo vivan como un dolor. Hasta no entrevistar, no lo descartamos.

---

## 2. Stakeholders

### Roles

| Papel | Quién es | Qué necesita ver para decir que sí |
|---|---|---|
| Usuario | Quien carga el catálogo y lo publica: encargada/o de e-commerce, community manager, pasante *(H)* | Que la ficha salga casi hecha, que corregir sea más rápido que escribir desde cero, y que lo que manda a Google no vuelva rechazado |
| Influenciador | Fotógrafo/a o diseñador/a de la marca; el referente técnico si la tienda tiene agencia *(H)* | Que no le cambien la forma de nombrar las prendas ni la línea de fotos |
| Recomendador | La agencia que administra la tienda o los anuncios; el dueño/a cuando no es quien paga *(H)* | Ver otra marca parecida que lo use y le vaya bien, y que la herramienta no le quite el control de las campañas |
| Comprador | Dueño/a de la marca o gerente de e-commerce | Un costo mensual menor a lo que hoy gasta en tiempo de carga y en quien le maneja los anuncios |
| Decisor | Dueño/a. En una pyme suele ser el mismo que el comprador *(H)* | Que no publique cosas equivocadas que generen reclamos, y que no se gaste ni se pause nada sin su aprobación |
| Saboteador | Agencias o freelancers que hoy cobran por cargar catálogo o por manejar campañas; Google y las plataformas de e-commerce si suman algo propio *(H)* | Nada que podamos ofrecerles; es un riesgo a monitorear |

Reconocimiento del problema (los tres escalones de la consigna): *(H)* creemos que, para la carga, la mayoría está en el segundo (lo sabe y le duele, pero no hizo nada), y que una minoría ya tiene un apaño (plantillas, delegar). Para la publicidad en Google, creemos que la mayoría de las marcas chicas no anuncia, y las que anuncian ya tienen el apaño de una agencia. Sin entrevistas no podemos afirmarlo.

### Evidencia

#### De confirmación

- Ninguna todavía, más allá de razonamiento propio. Lo que sí sabemos es que hay tiendas chicas en plataformas como Tiendanube, lo que hace plausible que el usuario y el comprador sean la misma persona.
- Para la parte de anuncios, si la marca ya paga publicidad o una agencia, la partida existe: es la evidencia de comprador que pide la consigna para un mercado nuevo. No sabemos cuántas marcas están en ese caso.

#### De refutación

- Si el usuario (quien carga) y el comprador (quien paga) fueran personas distintas y el que paga no viera el costo del tiempo de carga, no habría presupuesto. Hay que averiguar si hoy la marca paga algo por esta tarea (sueldo, pasante, agencia). Si no paga nada y lo hace la dueña "gratis", el comprador no existe todavía.
- La agencia que maneja los anuncios puede ser a la vez recomendador y saboteador: si la herramienta hace parte de su trabajo, tiene motivos para desaconsejarla.

---

## 3. Hipótesis de solución

### Descripción del producto

Una herramienta para equipos de e-commerce de moda que lleva una prenda desde la foto hasta Google Shopping. Se suben las fotos de la prenda junto con los datos que solo la marca conoce (precio, talle, material, stock, link a la tienda). La herramienta devuelve la ficha lista para revisar (tipo de prenda, color, atributos, título y descripción), la convierte al formato de Google Merchant Center, la valida antes de publicarla, la publica y, una vez que la prenda está en una campaña de Shopping, marca las publicaciones que se muestran pero casi nadie clickea y propone reescribir su título. Lo que declara la marca (precio, talle, material, género) no se inventa: se usa tal cual o la fila queda marcada como incompleta.

Lo que ya existe en el repo:

- **Prototipo** (`deep-tagger-api`, `deep-tagger-ui`, `img-puller`): se sube una URL de imagen y devuelve tipo de prenda (10 clases de Fashion-MNIST), color dominante (K-Means con un solo grupo de color), atributos para tops, calzado y pantalones (cuello, calce, bolsillos, silueta, ocasión y temporada), título (BLIP) y descripción (Claude). No predice material, género ni estampa.
- **Pipeline de posicionamiento** (`positioner/`, especificado en `upgrade/`), en cuatro etapas:
  1. Enriquecimiento: corre el prototipo sobre la foto, arma el título con un orden fijo (marca, género, prenda, atributo, color, talle), mapea el tipo de prenda a la taxonomía de Google y guarda el color declarado y el predicho.
  2. Feed: une lo enriquecido con los datos de la marca, lo lleva al esquema de Merchant Center y valida cada fila (campos obligatorios de indumentaria, formato de precio y dígito verificador del GTIN cuando hay uno).
  3. Publicación: carga credenciales desde variables de entorno, apunta por defecto a una cuenta de prueba, sube solo las filas válidas y lee el estado de aprobación de cada producto.
  4. Campañas: crea o reutiliza una campaña de Shopping, trae métricas por producto y aplica una regla (CTR menor a 1% con al menos 100 impresiones) que propone pausar el producto y reescribir el título. Por defecto corre en modo simulación y no cambia nada.

  La lógica de las cuatro etapas está escrita y probada: 28 escenarios de los specs pasan, con las APIs de Google simuladas. Las llamadas reales a Merchant API y a Google Ads API todavía no están implementadas.

Lo que proponemos sumar:

1. **Control de coherencia foto vs. ficha** (nuevo, usa ML). Es lo que más se acerca al problema de las devoluciones por "no coincide" y, en Google, a un anuncio que lleva a una prenda distinta de la que se buscó. El enriquecimiento ya guarda el color declarado y el predicho, que es la base para compararlos.
2. **Confianza por campo y cola de revisión** (nuevo, usa ML). La persona revisa solo lo dudoso antes de publicar.
3. **Registro de lo que pasa después de sugerir**: correcciones de quien revisa, estado de aprobación de Google y métricas por producto. Es el dato que hoy no existe (ver sección 5).
4. **Más atributos y más tipos de prenda**: estampa, tipo de cuello, largo de manga, largo de la prenda, vestidos, abrigos, accesorios. Hoy hay atributos solo para 3 categorías, y la muestra del Hito 2 no trae vestidos ni abrigos.
5. **Vocabulario local**: títulos y descripciones en español rioplatense (remera, buzo, campera, jogger, pollera).
6. **Llamadas reales a Merchant API y Google Ads API**, contra cuentas de prueba primero (esto es integración, no ML).

Qué se resuelve sin ML: mapear el tipo de prenda a la taxonomía de Google, armar el título con una plantilla, validar campos obligatorios, precio y GTIN, publicar, crear campañas y la regla de optimización de la etapa 4, que es un umbral fijo y no un modelo. Decirlo ahorra tiempo.

Qué porción del problema cubre: la carga, el desajuste entre lo publicado y la foto, y la publicación en Google como primer canal. No cubre talle ni calce, que según las fuentes pesan más en las devoluciones de moda, ni la exportación a otros canales (tienda propia, marketplaces). Además, con los datos que tenemos solo se puede medir una parte de la ficha: el veredicto del Hito 2 deja en la PoC cuello, calce, silueta, ocasión, bolsillos y temporada. Tipo de prenda, color, título y descripción los produce el prototipo, pero no tenemos con qué medir si aciertan ([veredicto](hito-2/4-veredicto.md)).

### Acción

Quien carga acepta la ficha sugerida, revisa solo los campos marcados como dudosos, completa los datos de la marca y publica el lote en Google. Corrige las filas que el validador rechaza antes de que lleguen a Google. Con las campañas andando, revisa las publicaciones que la herramienta marcó por bajo CTR y decide si las pausa o les cambia el título. Cambia que deja de completar todo desde cero y pasa a revisar excepciones. Quien decide es la persona de e-commerce, al subir cada lote y al revisar las publicaciones marcadas; la herramienta propone y no ejecuta sola.

### Predicción

Para cada prenda y cada atributo que sale de la foto (tipo, color, cuello, calce, silueta, ocasión, bolsillos, temporada): el valor más probable y, con la confianza por campo (punto 2), qué tan seguro está. Para el control de coherencia: la probabilidad de que un dato declarado en la ficha no corresponda a lo que se ve en la foto.

La marca de bajo CTR de la etapa 4 no es una predicción: es una regla sobre métricas que ya pasaron.

Una advertencia: el material (algodón, lino, poliéster) muchas veces no se distingue en una foto. Por eso el material, el género y el talle los declara la marca y la herramienta no los predice.

### Juicio

Cuánto cuesta publicar un dato equivocado (un reclamo, una devolución, una prenda que no aparece en los filtros, un rechazo de Google, un clic pagado que lleva a una prenda que no era) contra cuánto cuesta revisar un campo de más (unos segundos de quien carga). De eso sale el umbral de confianza a partir del cual la herramienta marca un campo para revisión.

*(H)* Punto de partida: umbral más exigente para tipo de prenda y color (generan más reclamos y son los que Google usa para filtrar) que para temporada y ocasión (más subjetivos).

Para las campañas: en Shopping se paga por clic, así que una publicación con CTR bajo cuesta poca plata; lo que pierde es visibilidad. Pausarla por error saca de la vidriera una prenda que podía venderse. Por eso la regla propone y no ejecuta. *(H)* El umbral de partida (CTR de 1% con al menos 100 impresiones) es nuestro.

Ninguno de los dos umbrales lo validó ni lo firmó alguien de una marca real. Los tiene que fijar la dueña o el gerente de e-commerce, no nosotros.

### Evidencia

#### De confirmación

- La decisión existe hoy: alguien mira la foto (y la etiqueta) y completa la ficha, prenda por prenda. Tiempo y frecuencia: no medido.
- Google publica qué campos pide para indumentaria y en qué casos son obligatorios (fuente 13). La etapa 2 valida contra esa especificación, así que lo que llega a Google ya pasó el control que Google aplica a los campos.
- Existen servicios comerciales de etiquetado automático de moda (se mencionan en sección 4), lo que sugiere que la predicción es posible. Verificar cuánto aciertan y con qué tipo de fotos.
- El prototipo ya devuelve tipo, color, atributos, título y descripción para una URL (hay un video demo en el repo). Las cuatro etapas del pipeline pasan sus 28 escenarios con las APIs simuladas (`python scripts/spec_report.py`).

#### De refutación

- Si quien carga revisa todos los campos igual, porque no confía en la sugerencia, la herramienta no ahorra tiempo y la acción no cambió. Se prueba dándole el prototipo a 3 o 4 personas y contando cuántos campos cambian.
- Un modelo multimodal generalista (por ejemplo Claude, GPT o Gemini con la foto y un prompt) podría hacer lo mismo o mejor sin entrenar nada. Si al probarlo con 100 fotos supera a nuestro pipeline, hay que replantear qué aporta el modelo propio. Todavía no hicimos esa comparación.
- Google ya automatiza las pujas y la distribución del presupuesto en sus campañas. *(H)* Puede que la regla de la etapa 4 no agregue nada sobre lo que Google hace solo; lo único que seguiría siendo nuestro es la reescritura del título.
- Con cuentas de prueba, Google Ads no muestra anuncios ni devuelve métricas reales ([`upgrade/requirements.md`](upgrade/requirements.md)). La regla de la etapa 4 no se puede validar sin gastar en anuncios reales.
- Que nadie quiera fijar los umbrales (cuánto vale un error) sería un problema del proyecto, no técnico.

---

## 4. Alternativas y statu quo

### Qué hace hoy el usuario

Carga a mano en el formulario de la plataforma o en una planilla de carga masiva. Reutiliza la ficha de una prenda parecida. Copia lo que manda el proveedor. Delega en un pasante o freelance. Para aparecer en Google, *(H)* usa la integración que ofrezca la plataforma de su tienda, carga los productos a mano en Merchant Center o se lo encarga a una agencia. *(H)* Es probable que ya use un asistente de chat para redactar descripciones, lo que hay que confirmar preguntando.

Costo de hacerlo así: tiempo por prenda (no medido), fichas inconsistentes entre temporadas, errores que aparecen como reclamos, productos rechazados por Google y la comisión de la agencia, cuando hay una.

Qué tiene de bueno: no cuesta nada extra, tiene la prenda en la mano (puede leer la etiqueta y tocar la tela), y controla lo que dice cada ficha. Si hay agencia, alguien con experiencia decide sobre las campañas.

### Qué otras soluciones existen o podrían aparecer

- Servicios de etiquetado automático de moda vía API (por ejemplo Ximilar, Vue.ai u otros). No verificamos precios, cobertura de atributos ni si funcionan para el mercado argentino.
- Integraciones de las plataformas de e-commerce con Google (Tiendanube, Mercado Libre) que manden el catálogo a Merchant Center. No revisamos qué ofrecen hoy ni qué campos completan.
- Las herramientas de Google: Merchant Center y las campañas automáticas de Google Ads. No revisamos si completan atributos a partir de la foto o de la página del producto.
- Modelos multimodales generalistas usados a mano por el propio usuario.

### Por qué lo nuestro sería suficientemente mejor como para que alguien se mueva

Hipótesis, en orden de qué tan defendibles nos parecen:

1. El control de coherencia foto vs. ficha con confianza por campo. No lo vimos en las alternativas, pero no las revisamos a fondo.
2. Un solo paso de la foto a un feed que Google acepta: la ficha, los campos obligatorios de indumentaria y la validación antes de publicar, sin rechazos que haya que descifrar.
3. Vocabulario local en títulos y descripciones.
4. Aprender de las correcciones de cada marca (su forma de nombrar, su paleta) y del rendimiento de sus títulos.
5. Precio pensado para pymes argentinas. No sabemos qué pagarían.

Nada de esto está demostrado. El primer paso para sostenerlo es la comparación contra un modelo generalista de la sección 3.

### Evidencia

#### De confirmación

- Ninguna propia todavía. Falta observar a una persona cargando 10 prendas y publicándolas en Google, y ver qué hace, con qué herramientas y cuánto tarda.

#### De refutación

- Puede que el statu quo (plantillas + un asistente de chat + la integración de su plataforma con Google) ya sea suficiente para la mayoría y la mejora sea chica. Es el caso a descartar primero.
- Puede que las plataformas o Google ya completen atributos a partir de la foto, con lo cual el hueco no existe.

---

## 5. Hipótesis de datos

Enfoque: aprendizaje supervisado. La respuesta correcta es la ficha que un humano validó para esa foto. Para las campañas, la respuesta llega después: si Google aprobó el producto y cómo rindió el anuncio.

### Dataset

| Dato | Origen | ¿Público? | ¿Lo vimos? | ¿Sensibles? | Sesgo conocido | Comentarios |
|---|---|---|---|---|---|---|
| Fashion-MNIST (70k imágenes, 10 clases) | Zalando Research | Sí | Sí, entrenamos con él | No | Imágenes de 28×28 en gris, sobre fondo limpio, pocas clases; no se parece a una foto de catálogo real | Entrena el clasificador de tipo de prenda |
| CSV etiquetados de `img-puller` (tops, calzado y pantalones): 60.603 imágenes con dos atributos por categoría y temporada, **sin color, tipo de prenda, material, título ni descripción** | Lily AI, sobre productos de sus empresas clientes. Las fotos son casi todas de Rue Gilt Groupe, retailer off-price de EE. UU. Aportados por Jorge Federico Flores | No. Uso autorizado para la materia, no comercial | Sí: es la muestra del Hito 2 ([EDA](hito-2/2-eda-de-viabilidad.md)) | Fotos con modelo | Un solo retailer de EE. UU.; sin vestidos ni abrigos; temporadas del hemisferio norte | Entrena los atributos (cuello, calce, bolsillos, silueta, ocasión, temporada). No sale de Fashion Product Images: ningún `id` coincide ([mapa de fuentes](hito-2/1-mapa-de-fuentes.md), fuente 3) |
| Fashion Product Images: `styles.csv` con 44.424 filas legibles, `gender`, `masterCategory`, `subCategory`, `articleType`, `baseColour`, `season`, `usage`, `year` y `productDisplayName` (título) | Kaggle (paramaggarwal), catálogo de Myntra. Hay una versión chica en Hugging Face (`ashraq/fashion-product-images-small`) | Sí. Kaggle declara licencia MIT, puesta por quien publicó el dataset, no por el retailer | Sí, `styles.csv`; las imágenes no | Fotos con modelo | Retail de India, productos de 2007 a 2019. Títulos cortos con marca ("Turtle Check Men Navy Blue Shirt"), sin descripción larga | Trae color (99,97 % de las filas) y tipo de prenda (143 valores), pero no cumple el nivel de vigencia del EDA (2021 en adelante) |
| H&M: unos 105.000 artículos con `prod_name`, `detail_desc` (descripción), tipo, color e imagen | Competencia de Kaggle "H&M Personalized Fashion Recommendations". En Hugging Face: `Qdrant/hm_ecommerce_products` (completo) y `wbensvage/clothes_desc` (1.000 pares imagen + texto) | Sí, bajo las reglas de la competencia | No | Fotos con modelo. Las transacciones y clientes no los necesitamos | Una sola marca, fast fashion europea, en inglés y con el estilo de redacción de H&M | Es lo más parecido a "foto + ficha escrita por la marca". El espejo de Qdrant dice CC BY 4.0, pero no puede cambiar la licencia de datos que no son suyos: leer las reglas de la competencia |
| Amazon Reviews 2023, categoría Clothing, Shoes & Jewelry: 7,2 millones de ítems con `title`, `description`, `features`, `details` (incluye material) e `images` | McAuley Lab (UCSD), Hugging Face `McAuley-Lab/Amazon-Reviews-2023` | Sí, pensado para investigación. La página no publica licencia | No | Los metadatos no; las reseñas tienen ids de usuario y no las usaríamos | EE. UU., en inglés. Textos escritos por vendedores, de calidad muy despareja y con mucho marketing | Volumen enorme. Hay que filtrar un subconjunto limpio. Es de las pocas fuentes que trae material declarado |
| Fashion-Gen: unas 293.000 imágenes de 1360×1360 con descripción escrita por estilistas; 48 categorías y 121 subcategorías | Element AI y SSENSE (paper de 2018) | Bajo pedido. Licencia no verificada | No | Algunas con modelo | Lujo, fondo uniforme, en inglés | Las descripciones son las más parecidas a una ficha de catálogo. No sabemos si todavía se puede pedir |
| DeepFashion-MultiModal: 44.096 imágenes con atributos manuales de forma, tela (7 clases) y color/estampa (7 clases), más una descripción por imagen | CUHK MMLab | Sí, **solo investigación no comercial** | No | Sí: fotos de personas de cuerpo entero | Todas con modelo, sin fondo de catálogo. Las descripciones salen de plantillas | Sirve para tela y estampa con anotación manual, y para medir cómo anda el prototipo con fotos con modelo. No se puede usar en un producto comercial |
| Fashionpedia: 48.825 imágenes, 27 categorías, 294 atributos finos (cuello, manga, largo, estampa) con máscaras | Google, Cornell y CVDF | Anotaciones CC BY 4.0. Las imágenes tienen la licencia de cada origen (Flickr, Unsplash, Pexels, etc.) | No | Sí: personas en la calle y en eventos | Fotos de la vida real, no de catálogo | Fuente de los atributos que queremos sumar (sección 3, punto 4). No trae títulos ni descripciones |
| Fashion200k: unas 200.000 imágenes con descripción corta y tres niveles de categoría | Han et al. (2017). Espejo en Hugging Face `Marqo/fashion200k` | Sí (el espejo dice Apache 2.0; las imágenes vienen de tiendas online) | No | Algunas con modelo | Retail de EE. UU. | Las descripciones son listas de atributos ("blue denim skinny jeans"): sirven más para títulos que para descripciones |
| Títulos de publicaciones de Mercado Libre en español y portugués, con categoría | MeLi Data Challenge 2019 (competencia pública de Mercado Libre; también en Kaggle) | Sí. Términos a revisar | No | No | Todos los rubros (hay que filtrar indumentaria) y categorías de Mercado Libre. No trae imágenes | La única fuente orgánica en español que encontramos. Muestra cómo se titula en la región ("remera", "buzo", "campera") |
| Títulos y descripciones **sintéticos** para las imágenes de `img-puller` | Generados por nosotros con un modelo multimodal (`img-puller/describer.py`), a partir de la imagen y los tags que ya tiene | Propio | No existe todavía | No, mientras no usemos fotos de clientes | Hereda los errores y el estilo del modelo que los genera. Tiende a descripciones parecidas entre sí | El prompt actual le pide al modelo que describa el material, que no se ve en la foto; antes de generar hay que sacarlo. Ver "Datos orgánicos y sintéticos" abajo |
| Fichas con errores **inyectados** (color, tipo o material cambiados a propósito) | Generadas por nosotros a partir de fichas reales | Propio | No existe todavía | No | El error es artificial: puede no parecerse a los errores que comete una persona | Entrena y evalúa el control de coherencia foto vs. ficha |
| Feed **sintético** de 1.500 productos (`upgrade/data/`) | Generado por nosotros con `upgrade/generate_synthetic_data.py`, a partir de las primeras 500 filas de cada CSV de `img-puller` | Propio | Sí | No | Los campos de la marca (marca, precio, GTIN, género, material, estampa, color, talle) se sortean con semilla fija y no miran la foto: el color declarado no coincide con la foto | Sirve para probar el contrato del feed, la validación y la publicación (etapas 2 a 4). No sirve para entrenar ni para evaluar modelos, ni para el control de coherencia |
| Imágenes de productos de marketplace (el ejemplo del README del prototipo usa una URL de mlstatic.com) | Mercado Libre, si ese es el origen | Visible en la web, pero eso no es lo mismo que libre de usar | Parcialmente | Fotos con modelos = imagen de personas | Marcas grandes y fotos profesionales | Revisar términos de uso antes de seguir usándolas para entrenar |
| Pares foto + ficha cargados por humanos en marcas argentinas | Clientes o marcas piloto | No | **No** | Rostros de modelos | Depende de cada marca | Es el dato del que más depende el proyecto |
| Correcciones de quien revisa (campo, valor sugerido, valor final) | Uso de la herramienta | No | **No existe todavía** | No | — | Hay que empezar a registrarlo desde el primer uso |
| Estado de aprobación de cada producto en Merchant Center, con los motivos de rechazo | Merchant API, al publicar | No | **No existe todavía** | No | Depende de las reglas de Google en cada momento | Dice qué campos hacen fallar una publicación. Se registra desde la primera carga |
| Métricas por producto en Google Ads (impresiones, clics, CTR, conversiones, costo) | Google Ads API, con anuncios reales | No | **No.** Las cuentas de prueba no muestran anuncios ni devuelven métricas reales | No: vienen agregadas, sin datos de cada comprador | Solo existen para marcas que anuncian y gastan | Es lo que alimenta la regla de la etapa 4 y lo que diría qué títulos rinden |
| Devoluciones con motivo | Marcas piloto | No | No | Datos de compradores | — | Para validar impacto a largo plazo |
| Especificación de datos de producto de Google Merchant Center y taxonomía de categorías de Google | Documentación de Google (fuentes 13 y 14) | Sí | Sí: la etapa 2 sigue sus campos obligatorios para indumentaria | No | Es una sola para todos los rubros. En Argentina, color, talle, género y edad son obligatorios para los listados gratuitos de indumentaria y para las prendas que se venden en varios colores o talles | Define la salida del feed. El GTIN es obligatorio solo para productos que lo tienen |

Dato del que más depende: los pares foto + ficha validados por humanos en catálogos reales. Sin ellos no podemos medir si acertamos ni entrenar el control de coherencia. Para la parte de campañas, el dato equivalente son las métricas reales de anuncios, que solo existen con gasto.

Extrapolable entre clientes: parcialmente. Los conceptos (color, tipo) sí, pero cada marca tiene su vocabulario, su paleta y su forma de fotografiar. Las métricas de anuncios dependen de cada marca, de su precio y de su competencia.

Datos personales: los rostros de modelos son datos personales (Ley 25.326). Además, mandar las imágenes de un cliente a un servicio externo para generar la descripción requiere que el cliente lo sepa y lo acepte, y publicar en Google requiere acceso a la cuenta de Merchant Center y de Google Ads de la marca. Hay que definirlo antes de un piloto.

### Datos orgánicos y sintéticos

El hueco concreto: las 60.603 imágenes de `img-puller` tienen categoría y algunos tags, pero no tienen título, descripción ni color. Sin texto escrito para esas fotos no podemos entrenar ni medir la parte que genera la ficha. Proponemos usar los dos tipos de dato, cada uno para una cosa distinta:

- **Orgánicos** (escritos por personas: H&M, Fashion Product Images, Amazon, Fashion-Gen, Mercado Libre y, más adelante, las marcas piloto). Sirven para entrenar donde alcanzan y, sobre todo, para **evaluar**. El set de evaluación es 100% orgánico. Nunca medimos contra texto sintético.
- **Sintéticos** (generados por nosotros). Sirven para tres cosas: (1) completar título y descripción de las imágenes de `img-puller`, en español rioplatense, que es lo que no existe en ningún dataset público con imagen; (2) armar fichas con errores inyectados para el control de coherencia; (3) el feed de prueba de `upgrade/data/`, que solo ejercita el contrato y la validación del feed.

Cómo armaríamos los títulos y descripciones sintéticos *(H, sin probar)*:

1. Un modelo multimodal recibe la imagen y los tags que ya tiene, y escribe título y descripción. Los tags van como restricción: el texto no puede contradecir el tipo ni el color conocidos.
2. Un filtro automático descarta los textos que contradicen los tags o que mencionan material cuando el tag no lo trae. El material casi nunca se ve en la foto (sección 3), y un texto sintético que lo inventa enseña a inventarlo.
3. Una persona del equipo revisa una muestra (*H: unas 300*) y anota cuántos tienen errores. Esa tasa se reporta junto con cualquier resultado que use el dato sintético.
4. Entrenamos con sintético y orgánico mezclados, y medimos solo sobre orgánico.

Lo que hay que tener presente con lo sintético:

- **Circularidad con la sección 3.** Si el texto de entrenamiento lo escribe un modelo generalista, el nuestro como mucho lo imita. La comparación "nuestro pipeline vs. un modelo generalista" queda sesgada a favor del generalista en calidad de texto. Lo que el modelo propio podría aportar es costo, velocidad, vocabulario local y el control de coherencia, y hay que medirlo en esos términos.
- Los modelos multimodales inventan atributos finos (fuente 12). El dato sintético va a tener errores, y por eso se mide su tasa.
- Homogeneización: si todas las descripciones salen del mismo generador, todas suenan igual (sección 7, riesgo social).
- Términos de uso del generador: algunos proveedores restringen usar sus salidas para entrenar otros modelos. Revisarlo antes de generar.
- Idioma: casi todo lo orgánico con imagen está en inglés. El español rioplatense sale de traducción o generación, o sea, también es sintético. Los títulos de Mercado Libre son la única referencia orgánica en español, y no traen imagen.
- El feed de `upgrade/data/` es sintético en todo lo que agrega. Que pase la validación dice que el código funciona, no que los datos de una marca real la vayan a pasar.

Licencias: varios de estos datasets son solo para investigación no comercial (DeepFashion-MultiModal seguro; H&M, Amazon y Fashion-Gen sin confirmar). Para la PoC de la materia alcanza. Para un producto no alcanza, y el dato del que más depende el proyecto sigue siendo el mismo: fichas reales de marcas argentinas.

### Evidencia

#### De confirmación

- Abrimos y usamos Fashion-MNIST y los CSV de `img-puller`. El EDA del Hito 2 midió estos últimos: 60.603 filas, 100 % de valores presentes en las etiquetas y 100 % de las URLs que responden ([EDA](hito-2/2-eda-de-viabilidad.md), celdas V1 y C2).
- Abrimos `styles.csv` de Fashion Product Images: trae color y tipo de prenda casi completos ([mapa de fuentes](hito-2/1-mapa-de-fuentes.md), fuente 3).
- Existen datasets públicos de moda con pares imagen + texto escrito por personas: H&M, Amazon Reviews 2023, Fashion-Gen, DeepFashion-MultiModal y Fashion200k (fuentes 6 a 11). Leímos las fichas de cada uno en Hugging Face, GitHub o el paper, pero **no abrimos ninguno todavía**. Por eso en la tabla dicen "No".
- Generar texto sintético a partir de imágenes de moda es una práctica publicada (fuente 12), así que lo que proponemos es posible.
- Leímos la especificación de Google para indumentaria (fuente 13): los campos que la etapa 2 exige son los que Google pide en Argentina para los listados gratuitos y para las prendas con variantes.

#### De refutación

- Fashion-MNIST no representa fotos reales de catálogo. Es probable que el clasificador de tipo rinda peor con fotos con modelo, fondos complejos o prendas fuera de esas 10 clases. Todavía no lo medimos con fotos reales.
- El cálculo de color descarta píxeles blancos y busca un solo grupo de color. Con una modelo puesta la prenda, la piel, el pelo y el fondo se van a colar. Falta probarlo.
- Los CSV de `img-puller` no traen color ni tipo de prenda, y todas sus fotos son de un retailer de EE. UU. (EDA, celdas C2 y P5). Por eso el veredicto del Hito 2 deja color y tipo fuera de lo que se puede medir.
- Fashion Product Images trae color y tipo, pero sus productos son de 2007 a 2019 y no cumplen el nivel de vigencia del EDA.
- No encontramos ningún dataset público con imagen + ficha en español, ni de marcas argentinas. Todo lo orgánico con imagen es de EE. UU., Europa o India.
- La misma literatura que genera descripciones sintéticas dice que los modelos multimodales sin ajuste inventan o confunden atributos finos (fuente 12). Si la tasa de error del sintético resulta alta, no sirve para entrenar.
- Las licencias de H&M, Amazon y Fashion-Gen no están confirmadas. Si alguna prohíbe el uso fuera de su competencia o fuera de investigación, sale de la tabla.
- Las métricas de anuncios no existen sin gasto real. Durante la materia no vamos a tener ninguna.

---

## 6. Métrica de éxito

### Métrica de negocio

Minutos de trabajo por prenda, desde que la foto está lista hasta que el producto está publicado y aprobado en Google Merchant Center, con la herramienta contra sin ella. Como complemento: porcentaje de campos que la persona modifica antes de publicar, y porcentaje de productos que Google aprueba en el primer envío.

A largo plazo (no medible dentro del trimestre): porcentaje de devoluciones por "no coincide con la descripción/foto" en las marcas piloto, y CTR y ventas desde Google Shopping de las publicaciones cuyo título se reescribió.

### Umbral — por debajo de esto, no vale la pena

*(H) Provisorio, hasta tener la línea de base:*

- Menos de 30% de reducción del tiempo por prenda: no vale la pena.
- Si la persona modifica más del 20% de los campos de tipo y color: sigue revisando todo y no ahorra nada.
- Si Google rechaza más del 10% de los productos en el primer envío: la validación no está haciendo su trabajo.

Estos números están puestos antes de construir, pero salen de nuestra intuición. Se ajustan con la medición de abajo.

### Cómo se mediría dentro del trimestre, aunque sea de forma aproximada

Conseguir 3 o 4 personas que cargan catálogo de moda (marcas conocidas, aunque sea informal). Cronometrar la carga de 20 a 30 prendas a mano y otras 20 a 30 con el prototipo, y anotar cuántos campos cambian. La muestra es chica y no permite conclusiones estadísticas, pero da una línea de base y un orden de magnitud.

El porcentaje de aprobación se mide subiendo esas mismas prendas a una cuenta de prueba de Merchant Center, una vez implementada la llamada real de la etapa 3. El CTR no se puede medir dentro del trimestre: las cuentas de prueba de Google Ads no muestran anuncios.

### Métrica técnica que usaríamos como proxy

F1 por atributo sobre un set de fotos reales etiquetadas por nosotros. Con los datos de hoy, solo para los atributos que el Hito 2 deja en la PoC: cuello, calce, silueta, ocasión, bolsillos y temporada. Para el control de coherencia: precisión y recall de las discrepancias detectadas, sobre un set con errores agregados a propósito (esto es artificial y hay que decirlo cuando se reporte). Para el feed: porcentaje de filas que el validador marca como válidas y motivos de las inválidas. Sirve para saber si el sistema mejora de una semana a otra; no reemplaza a la métrica de negocio.

### Qué se registra de cada uso

Por cada prenda: id de la imagen, valor sugerido y confianza de cada campo, valor final publicado, si fue modificado, tiempo hasta publicar, resultado del validador, estado de aprobación de Google con sus motivos, y, cuando haya campañas, las métricas del producto, la acción que propuso la regla de la etapa 4 y si la persona la aceptó.

### Evidencia

#### De confirmación

- Tiempo por prenda es medible sin construir nada más que el cronometraje. Con el prototipo actual ya se puede correr la parte "con herramienta".
- El validador ya devuelve, para cada fila, si es válida y por qué no (`positioner/feed/report.py`). El porcentaje de filas válidas sale de ahí sin trabajo extra.

#### De refutación

- No hay línea de base: nadie sabe cuánto tarda hoy, y por eso el umbral es todavía un número inventado. Si las marcas no lo miden, la línea de base la armamos nosotros con una muestra chica.
- El rendimiento de los anuncios, que es lo que justifica la etapa 4, no se puede medir sin gasto real. Durante la materia esa parte del producto queda sin métrica de negocio.

---

## 7. Riesgos éticos y de sesgo (preliminar)

Sobre quién decide el sistema aunque no lo use: las compradoras finales. Lo que la herramienta escribe sobre una prenda decide cómo la ven, si la encuentran en Google y qué anuncio les aparece.

**Asignación.** Una prenda mal categorizada o con el color equivocado no aparece en los filtros y no se vende. Para una marca chica es visibilidad perdida. Lo mismo pasa si la regla de la etapa 4 pausa una prenda por CTR bajo con pocas impresiones: la decisión se toma con muy poca información. Si la herramienta hace una tarea que antes se revisaba a mano, el error pasa más desapercibido.

**Calidad de servicio.** Probablemente ande peor con: prendas fuera de las clases de entrenamiento (ropa tradicional, talles grandes, ropa infantil, ropa de trabajo), fotos con modelo o sobre fondo no blanco, estampas y colores fuera del diccionario de colores, y las marcas chicas con fotos caseras. El Hito 2 confirmó que todos los datos de atributos vienen de un solo retailer de EE. UU. *(H)* A medir: acierto por tipo de prenda, tipo de foto y marca.

**Representación.** Las categorías con las que el sistema nombra las prendas (10 clases, "casual/formal") pueden no calzar con la ropa sin género. Google pide un género para indumentaria (masculino, femenino o unisex), así que el canal impone esas tres casillas. Las descripciones generadas pueden reforzar estereotipos ("ideal para una noche elegante").

**Interpersonal.** Las fotos con modelos son imagen de personas; enviarlas a un servicio externo sin acuerdo es un problema. También el uso de imágenes de terceros para entrenar (sección 5). Para publicar, la herramienta necesita credenciales de la cuenta de Merchant Center y de Google Ads de la marca: quien las tenga puede publicar en su nombre y gastar su presupuesto.

**Social.** A escala, los títulos y descripciones se homogenizan y el catálogo de todas las marcas suena igual. Se pierde el trabajo de quien redacta y se puede degradar el posicionamiento en buscadores por contenido repetido. Si los títulos se reescriben para subir el CTR, se premian los títulos llamativos por sobre los que describen bien la prenda. Efecto sobre el empleo de quienes hoy cargan o manejan anuncios: no lo sabemos.

**Proxy.** El origen de las imágenes (marcas grandes con fotos profesionales) puede estar haciendo de proxy de "calidad de foto" o "nivel socioeconómico de la marca". En campañas, el CTR depende del precio y de la foto, no solo del título: una regla que mira el CTR puede castigar a las prendas caras o con fotos caseras.

**Para qué no debería usarse.** Para publicar sin revisión humana, sobre todo en material, talle o composición. Para correr la etapa 4 fuera del modo simulación sin que la dueña apruebe cada acción. Para inventar datos que declara la marca (precio, GTIN, stock): lo que se publica en un anuncio obliga al vendedor. La ley de defensa del consumidor (24.240) exige información veraz sobre el producto, hace vinculante lo que dice la publicidad, y la responsabilidad es del vendedor, no de la herramienta.

**Si se equivoca, ¿cómo se entera la persona?** Antes de publicar, por el validador y la marca de campos dudosos. Después, por el estado de aprobación de Google y por las acciones que propone la etapa 4. Por eso la sección 6 pide registrar correcciones, rechazos y acciones aceptadas, y el diseño incluye confianza por campo.

Regulación: no toca las categorías críticas de la lista (empleo, crédito, biometría, etc.), salvo por las fotos con rostros. Aplican las leyes 25.326 y 24.240, y las políticas de publicidad de Google, que no leímos todavía.

### Evidencia

#### De confirmación

- Fashion-MNIST se creó como reemplazo simple de MNIST, no como representación de catálogos reales.
- Las devoluciones por diferencia entre lo publicado y lo recibido están documentadas (fuentes 1 y 2), así que el costo de un dato mal puesto es real.
- La especificación de Google limita el género de indumentaria a tres valores (fuente 13).

#### De refutación

- Falta buscar casos publicados de sistemas de etiquetado de moda que hayan fallado por sesgo. Hasta ahora no encontramos uno concreto, así que no podemos afirmar que estos riesgos hayan pasado en la práctica.
- Si la herramienta siempre queda con revisión humana antes de publicar, y la etapa 4 solo propone, varios de estos riesgos bajan. Habría que poder decir por qué en cada caso.

---

## Fuentes consultadas (2026-09-28)

1. asest.es, "comercio electrónico con una menor tasa de devoluciones": motivos de devolución (tabla atribuida a DHL). https://asest.es/story/comercio-electronico-con-una-menor-tasa-de-devoluciones/
2. flexfulfillment.eu, "7 formas de reducir devoluciones en moda ecommerce": participación de talle/ajuste y descripción en devoluciones de moda. https://www.flexfulfillment.eu/es/7-formas-reducir-devoluciones-moda-ecommerce/
3. Mercado, "CACE: el estudio anual midió $34.033.238 millones en 2025". https://mercado.com.ar/tendencias/comercio-electronico-el-estudio-anual-de-cace-midio-34-033-238-millones-en-2025
4. iProfesional, "Comercio electrónico vs inflación: ¿quién ganó en 2025?": retroceso de indumentaria. https://www.iprofesional.com/tecnologia/447692-comercio-electronico-vs-inflacion-quien-gano-en-2025
5. Cadena 3, sobre Hot Sale 2025 (cifra de tiendas de Tiendanube). https://www.cadena3.com/noticia/el-dato-confiable/que-fue-lo-mas-vendido-en-hot-sale_422993

Todas son fuentes secundarias (prensa y blogs). Hay que reemplazarlas por los informes originales antes de la entrega final.

### Datasets (consultados el 2026-10-06)

6. H&M Personalized Fashion Recommendations (Kaggle). https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations · Espejos: https://huggingface.co/datasets/Qdrant/hm_ecommerce_products y https://huggingface.co/datasets/wbensvage/clothes_desc
7. Fashion Product Images (Kaggle). https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset · Versión chica: https://huggingface.co/datasets/ashraq/fashion-product-images-small
8. Amazon Reviews 2023 (McAuley Lab). https://amazon-reviews-2023.github.io/ · https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
9. Rostamzadeh et al., "Fashion-Gen: The Generative Fashion Dataset and Challenge" (2018). https://arxiv.org/abs/1806.08317
10. DeepFashion-MultiModal. https://github.com/yumingj/DeepFashion-MultiModal
11. Fashionpedia: https://github.com/cvdfoundation/fashionpedia · Fashion200k: https://huggingface.co/datasets/Marqo/fashion200k · MeLi Data Challenge 2019: https://www.kaggle.com/datasets/fredericods/mercado-libre-data-challenge
12. "RA-CoA: Training-free Fashion Image Captioning via Retrieval-Augmented Chain-of-Attributes". https://arxiv.org/html/2609.14100

### Google (consultadas el 2026-10-26)

13. Google Merchant Center, "Especificación de datos de producto": campos obligatorios para indumentaria (color, talle, género, edad), GTIN y valores de disponibilidad. https://support.google.com/merchants/answer/7052112?hl=es-419
14. Google Merchant Center, "Supported languages and currencies": Argentina como país de destino, con pesos argentinos (ARS). https://support.google.com/merchants/answer/160637
15. Google Ads API, cuentas de prueba. https://developers.google.com/google-ads/api/docs/best-practices/test-accounts · Merchant API, cuentas de prueba. https://developers.google.com/merchant/api/guides/accounts/test-accounts

---

## Bitácora de revisiones

| Fecha | Sección | Qué cambió | Qué lo motivó |
|---|---|---|---|
| 2026-09-28 | Todas | Primera versión | Clase 02, README del repo, búsqueda de proxies de devoluciones y e-commerce |
| 2026-09-28 | Encabezado y sección 3 | Referencia al repo que extendemos (MatLock/UdeSA-computer-vision) | Dejar explícito a qué proyecto se refiere "el repo" |
| 2026-10-06 | Sección 5 | Datasets públicos candidatos, estrategia de datos orgánicos y sintéticos, nueva evidencia y fuentes 6 a 12 | Las ~50.000 imágenes del repo no tienen título ni descripción; búsqueda en Hugging Face, Kaggle y papers |
| 2026-10-20 | Sección 5 | Corrección de la fila de img-puller: las celdas de "Origen" y "¿Público?" estaban corridas | Hito 2 (mapa de fuentes) |
| 2026-10-20 | Encabezado | Integrantes actualizados: se suman Nilayvi Barrientos y Rubén Ortigoza | Cambio en la composición del equipo |
| 2026-10-26 | Todas | Versión 2: sumamos Google Shopping al caso (publicar y anunciar las prendas) y el repo pasa a ser el fork | Pasamos todo el proyecto al fork, que ya tenía el pipeline para Google |
| 2026-10-26 | Secciones 3 y 5 | Material, género y talle los carga la marca, no el modelo. Sumamos lo medido en el Hito 2 | El prototipo no predice esos campos; resultados del Hito 2 |
| 2026-10-26 | Fuentes | Fuentes 13 a 15, de Google | Chequear qué campos pide Google y si acepta Argentina |
