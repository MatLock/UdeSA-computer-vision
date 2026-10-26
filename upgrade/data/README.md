# Feed sintético

Tres CSV con 500 productos cada uno (1.500 en total), armados a partir de las primeras 500 filas de cada CSV de `img-puller` (`pants_tags.csv`, `shoes_tags.csv`, `tops_tags.csv`). A cada fila se le agregan los campos que pediría Google Merchant Center.

**Todo lo agregado es sintético.** Se sortea con semilla fija (42) y no mira la foto: el color, el material o el género declarados no tienen por qué coincidir con la prenda de la imagen. Sirve para probar el contrato del feed, la validación y la publicación en una cuenta de prueba (etapas 2 a 4). No sirve para entrenar ni evaluar modelos, ni para el control de coherencia foto vs. ficha ([canvas](../../canvas.md), sección 5).

## Archivos

| Archivo | Categoría | Filas |
|---|---|---|
| `pants_google_500.csv` | Pantalones | 500 |
| `shoes_google_500.csv` | Calzado | 500 |
| `tops_google_500.csv` | Tops | 500 |

## Columnas

Orden en los tres archivos: `id`, `image_url`, los dos atributos de la categoría, `season`, `relative_path`, `brand`, `price`, `gtin`, `availability`, `link`, `condition`, `gender`, `material`, `pattern`, `color`, `size`, `age_group`, `google_product_category`, `title`, `description` (21 columnas).

| Columna | Origen | Valores |
|---|---|---|
| `id`, `image_url`, `season` | CSV de `img-puller` | Sin cambios |
| `relative_path` | CSV de `img-puller` | Ruta local de la imagen después de bajarla con `img-puller` |
| Atributos de la categoría | CSV de `img-puller` | Pantalones: `pocket_details`, `fit_silhouette`. Calzado: `wearing_occasion`, `style_silhouette`. Tops: `neck_style`, `fit_silhouette` |
| `brand` | Sintético | StyleCo, FashionHub, TrendWear, ClassicStyle, ModernFit, ElegantLine (marcas inventadas) |
| `price` | Sintético | Entre 15.000,00 y 150.000,00, con formato `45999.00 ARS` |
| `gtin` | Sintético | La mitad de las filas tiene un GTIN-13 con prefijo argentino (779) y dígito verificador correcto; la otra mitad no tiene, como una prenda de marca propia |
| `availability` | Sintético | Siempre `in_stock` |
| `link` | Sintético | `https://store.example.com/<categoría>/<id>` |
| `condition` | Sintético | Siempre `new` |
| `gender` | Sintético | `female`, `male`, `unisex` |
| `material` | Sintético | Cotton, Polyester, Cotton Blend, Denim, Linen, Wool, Silk |
| `pattern` | Sintético | Solid, Striped, Checkered, Floral, Abstract, Geometric |
| `color` | Sintético | black, navy blue, white, gray, beige, khaki, blue, red, green |
| `size` | Sintético | Pantalones y tops: XS a XXL. Calzado: 35 a 44 |
| `age_group` | Sintético | Siempre `adult` |
| `google_product_category` | Mapeo | Pantalones: `Apparel & Accessories > Clothing > Pants`. Calzado: `Apparel & Accessories > Shoes`. Tops: `Apparel & Accessories > Clothing > Tops`. Son los mismos caminos que usa `positioner/enrichment/category_mapper.py` |
| `title` | Plantilla | `marca + prenda + género + material + color + talle`, en español. En calzado, la prenda sale de `style_silhouette` (Botas, Mocasines, Sandalias, etc.) |
| `description` | Plantilla | Prenda, material, color, estampa y, si la temporada es conocida, para qué temporada está pensada |

Ejemplo (calzado):

```
10097473,https://asset1.ruecdn.com/images/product/131395/1313956101_RLLDTH_1.jpg,Casual,Slides,Tropical-Vacation,images/shoes/10097473.jpg,ElegantLine,130123.16 ARS,7790392405141,in_stock,https://store.example.com/shoes/10097473,new,male,Denim,Solid,khaki,38,adult,Apparel & Accessories > Shoes,ElegantLine Chinelas hombre de denim caqui talle 38,"Chinelas de denim en color caqui, liso. Pensado para vacaciones en clima cálido."
```

## Cómo se usa en cada etapa

| Etapa | Uso |
|---|---|
| 1 · Enriquecimiento | Las columnas de la marca entran como datos declarados; el pipeline completa lo que se infiere de la foto |
| 2 · Feed | Todas las filas pasan el contrato y la validación; las que no tienen GTIN prueban el caso de marca propia |
| 3 · Publicación | Se suben a una cuenta de prueba de Merchant Center ([`requirements.md`](../requirements.md)) |
| 4 · Campañas | Sirven para crear una campaña de prueba; las métricas de una cuenta de prueba vuelven en cero |

## Cómo se regenera

Desde la raíz del repo:

```bash
python upgrade/generate_synthetic_data.py
```

Con la semilla fija, el resultado es siempre el mismo. **No usar en producción:** los GTIN, precios, links y marcas no son reales.
