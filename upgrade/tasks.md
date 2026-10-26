# Tareas por etapa

Cada etapa se cierra con todos sus escenarios en verde (`python scripts/spec_report.py upgrade/specs/<etapa>/spec.md`) antes de pasar a la siguiente.

## Etapa 1 · Enriquecimiento

- [x] 1.1 Definir el modelo del producto enriquecido (`EnrichedProduct`)
- [x] 1.2 Reutilizar el prototipo (TinyVGG, ResNet, K-Means, BLIP, Claude) para producir los campos inferidos
- [x] 1.3 Pasar tal cual los campos que declara la marca (género, material, estampa) y guardar el color declarado y el predicho
- [x] 1.4 Armar el título con orden fijo (`marca + género + prenda + atributo + color + talle`)
- [x] 1.5 Mapear el tipo de prenda a `google_product_category`
- [x] 1.6 Marcar como `partial` la fila sin imagen y como `needs_category_review` el tipo sin mapeo
- [x] 1.7 Tests (ver `specs/product-enrichment/spec.md`)

## Etapa 2 · Feed

- [x] 2.1 Definir el contrato del CSV de entrada (campos de la marca y campos inferidos)
- [x] 2.2 Unir lo enriquecido con los campos de la marca
- [x] 2.3 Mapear al esquema de Merchant Center
- [x] 2.4 Validar campos obligatorios de indumentaria, formato de precio y GTIN cuando lo hay
- [x] 2.5 Reporte de validación (filas válidas e inválidas con sus motivos)
- [x] 2.6 Tests (ver `specs/merchant-feed/spec.md`)

## Etapa 3 · Publicación

- [x] 3.1 Cargar credenciales desde variables de entorno
- [x] 3.2 Elegir ambiente (prueba por defecto, producción a pedido)
- [x] 3.3 Subir solo las filas válidas
- [x] 3.4 Seguir ante fallas parciales y devolver el resultado de cada producto
- [x] 3.5 Leer el estado de aprobación de cada producto
- [x] 3.6 Tests con el cliente simulado (ver `specs/merchant-publishing/spec.md`)
- [ ] 3.7 Llamadas reales a la Merchant API (`_upload_product`, `_get_status`)

## Etapa 4 · Campañas

- [x] 4.1 Crear o actualizar una campaña de Shopping
- [x] 4.2 Traer métricas por producto (impresiones, clics, CTR, conversiones, costo)
- [x] 4.3 Regla de optimización: CTR bajo con impresiones suficientes → proponer pausa y reescritura del título
- [x] 4.4 Modo simulación por defecto
- [x] 4.5 Tests con el cliente simulado (ver `specs/ads-optimization/spec.md`)
- [ ] 4.6 Llamadas reales a la Google Ads API (`_find_campaign`, `_create_campaign`, `_update_campaign`, `pause_product`, `_fetch_metrics`)

## Transversales

- [x] X.1 Documentar el entorno gratis y el pago (`requirements.md`) y enlazarlo desde el README del repo
- [x] X.2 No commitear credenciales (`.gitignore`; las credenciales se leen solo de variables de entorno)
- [ ] X.3 Prueba de integración contra las cuentas de prueba de Google, opcional y salteada en CI
