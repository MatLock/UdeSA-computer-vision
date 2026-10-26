# Propuesta de cambio: posicionamiento en Google Shopping

## Por qué

El prototipo genera, a partir de una foto, el tipo de prenda, el color dominante, algunos atributos (cuello, calce, bolsillos, silueta, ocasión, temporada), un título y una descripción. Eso cubre la primera de las cuatro capas que hacen falta para que una prenda aparezca y se promocione en Google:

```
Capa 1 · Contenido del producto       ← lo que hace el prototipo
Capa 2 · Feed de Google Merchant      ← no existía
Capa 3 · Publicación                  ← no existía
Capa 4 · Campañas y optimización      ← no existía
```

Para el usuario del [canvas](../canvas.md) (quien carga el catálogo en una marca argentina chica o mediana), cada capa que falta es trabajo manual: completar los campos que pide Google, corregir los rechazos y mirar qué anuncios rinden.

## Qué cambia

Se suman cuatro capacidades nuevas, sin modificar el comportamiento del prototipo:

1. **product-enrichment**: corre el prototipo sobre la foto, arma un título con orden fijo y mapea el tipo de prenda a la taxonomía de Google. Los datos que solo conoce la marca pasan tal cual.
2. **merchant-feed**: une lo enriquecido con los datos de la marca, lo lleva al esquema de Merchant Center y valida cada fila antes de publicar.
3. **merchant-publishing**: carga las credenciales, apunta a una cuenta de prueba por defecto, sube solo las filas válidas y lee el estado de aprobación.
4. **ads-optimization**: crea o reutiliza una campaña de Shopping, trae métricas por producto y aplica una regla de CTR que propone pausar el producto y reescribir su título, en modo simulación por defecto.

### Decisión de diseño: qué se infiere y qué declara la marca

| Tipo de campo | Campos | De dónde sale |
|---|---|---|
| Inferido de la foto | `google_product_category`, color, atributo del título, `title`, `description` | El pipeline (prototipo + plantillas + Claude) |
| Declarado por la marca | `price`, `gtin`, `availability`, `link`, `brand`, `condition`, `gender`, `material`, `pattern`, `size`, `age_group` | El CSV de la marca. **Nunca se inventa** |

El material, el género y el talle no se ven de forma confiable en una foto, y ningún modelo del prototipo los predice. El GTIN es opcional: Google lo pide solo para los productos que lo tienen, y una prenda de marca propia en general no lo tiene. Una fila se puede publicar solo si tiene los campos obligatorios y pasa la validación.

## Impacto

- Código nuevo en el paquete `positioner/`. El endpoint `/predict-image` y la CLI de `img-puller` siguen funcionando igual.
- Dependencias externas: cliente de Merchant API, cliente de Google Ads API y autenticación con cuenta de servicio. Las llamadas reales todavía no están implementadas.
- Requiere un proyecto de Google Cloud y cuentas de prueba ([`requirements.md`](requirements.md)).

## Fuera de alcance

- Reemplazar los modelos del prototipo (TinyVGG, ResNet, BLIP, K-Means): se reutilizan.
- Inferir material, género, estampa o talle a partir de la foto.
- Garantizar la aprobación de Google para datos inventados: el feed sintético de [`data/`](data/) es solo para cuentas de prueba.
- Gastar en anuncios durante el desarrollo: todo se prueba con cuentas de prueba.
- Ejecutar acciones sobre campañas sin aprobación humana: la etapa 4 propone y la persona decide.
