# 🚀 Automated Google Ads Positioning Upgrade — Summary

## Overview: ¿Qué se propone?

Sistema integrado de **4 milestones independientes** para automatizar posicionamiento en Google Ads:

1. **Enriquecer productos** con campos inferibles (título, género, material)
2. **Validar y mapear** a esquema Google Merchant
3. **Publicar feeds** seguros en Google Merchant Center
4. **Optimizar campañas** con métricas y reglas automáticas

**Riesgo**: Bajo. Totalmente testeable en ambiente sandbox gratis.  
**Reuso**: Modelos CV existentes (TinyVGG, ResNet, K-Means, BLIP).  
**Nuevas APIs**: Google Merchant API + Google Ads API (credentials vía Service Account).

---

## 📋 OpenSpecs Añadidos: Qué aporta cada uno

| OpenSpec | Propósito | Valor |
|----------|-----------|-------|
| **proposal.md** | Análisis de impacto y viabilidad | Justifica la inversión técnica |
| **requirements.md** | Tabla FREE vs PAID (sandbox vs producción) | Clarifica ruta desarrollo sin gastos |
| **tasks.md** | Checklist de 25+ tareas por milestone | Orienta implementación paso a paso |
| **README.md** | Navegación y contexto general | Acceso rápido para nuevos contribuyentes |
| **INDEX.md** | Índice detallado + preguntas frecuentes | Referencia completa para consultas |

---

## 🏗️ Especificaciones por Milestone

### Milestone 1: Product Enrichment
📁 `specs/product-enrichment/spec.md`

**¿Qué hace?**  
Completa campos inferibles usando pipeline CV existente + LLM.

| Campo | Fuente | Aporte |
|-------|--------|--------|
| **title** | Image + tags + LLM | Título estructurado: Marca+Género+Producto |
| **gender** | CV classifier + LLM | Segmentación demográfica para campañas |
| **material** | Image features (K-Means) | Filtrado por atributo de comprador |
| **pattern** | ResNet + image analysis | Diferenciador visual en feed Google |
| **color** | Image processing + CV | Variante clave para búsqueda usuarios |
| **size** | Tags + LLM inference | Compatibilidad con filtros Google Ads |
| **age_group** | Gender + material + category | Compliance COPPA automático |
| **google_product_category** | product_type → mapping table | Requerido por Google Merchant API |

**Tests**: 6 escenarios (imagen completa, missing, ambigua, etc.)  
**Datos entrada**: `upgrade/data/pants_google_500.csv` (500 pantalones)

---

### Milestone 2: Merchant Feed
📁 `specs/merchant-feed/spec.md`

**¿Qué hace?**  
Fusiona campos enriquecidos + datos negocio → validación Google schema.

| Validación | Por qué | Impacto |
|-----------|--------|---------|
| **GTIN checksum** | Google rechaza GTINs inválidos | Evita upload fallido de lotes |
| **Price format** | Requiere moneda + dos decimales | Previene errores de parsing |
| **Required fields** | Title, brand, condition, availability | Cumplimiento mínimo Merchant API |
| **Link activo** | Evita 404 en producto final | Reduce bounce rate de usuarios |

**Campos entrada (del negocio):**
- `brand`, `price`, `gtin`, `availability`, `link`, `condition`

**Campos output (para Google):**
- 21 columnas → esquema Google Merchant schema

**Tests**: 7 escenarios (campos faltantes, inválidos, parciales, etc.)

---

### Milestone 3: Merchant Publishing
📁 `specs/merchant-publishing/spec.md`

**¿Qué hace?**  
Carga feeds validados a Google Merchant Center, gestiona credenciales seguras.

| Componente | Función | Aporte |
|-----------|----------|--------|
| **Credential loader** | Lee env vars / service account | Zero-secretos en repositorio |
| **Environment selector** | Test (sandbox) vs Production | Control riesgo en desarrollo |
| **Merchant API client** | Upload de filas válidas solamente | Robustez ante fallos parciales |
| **Status readback** | Consulta aprobación/rechazo por fila | Feedback automático calidad |

**Modos de operación:**
- **FREE**: Merchant Center sandbox (test account)
- **PAID**: Merchant Center real (dominio verificado)

**Tests**: 6 escenarios con cliente mockeado

---

### Milestone 4: Ads Optimization
📁 `specs/ads-optimization/spec.md`

**¿Qué hace?**  
Crea campañas Shopping/Performance Max, extrae métricas, auto-optimiza.

| Regla | Condición | Acción |
|------|-----------|--------|
| **Pausar bajo CTR** | CTR < 1% por 7 días | Libera presupuesto para mejores |
| **Down-bid bajo ROAS** | ROAS < 3:1 | Reduce CPC gradualmente |
| **Flag título débil** | Impresiones altas, clicks bajos | Sugiere reescritura de titulo |
| **Rebalance presupuesto** | Conversión/día trending up/down | Traslada presupuesto dinámicamente |

**Modos operación:**
- **Dry-run**: Simula cambios sin mutar
- **Execute**: Aplica cambios reales (producción)

**Tests**: 7 escenarios (sin métricas, ROAS border, etc.)

---

## 📊 Datos Sintéticos: 1,500 Productos Completos

### Estructura

```
Campos originales (6):      id, image_url, relative_path, [tags por tipo]
Campos negocio (6):         brand, price, gtin, availability, link, condition
Campos inferibles (9):      title, description, gender, material, pattern,
                            color, size, age_group, google_product_category
─────────────────────────────────────────────────────────────────────
Total por fila:             21 columnas
Total productos:            1,500 (500 pants + 500 shoes + 500 tops)
Tamaño total:               ~664 KB
```

### Archivos CSVs

| Archivo | Filas | Propósito |
|---------|-------|-----------|
| `pants_google_500.csv` | 500 | Test M1–M4, categoría base |
| `shoes_google_500.csv` | 500 | Test M1–M4, diversidad prenda |
| `tops_google_500.csv` | 500 | Test M1–M4, mix completo |

**Características:**
- Datos **claramente sintéticos** (precios ficticios, GTINs inventados)
- Esquema **idéntico** en los 3 CSVs → validación uniforme
- Reproducible: `python3 upgrade/generate_synthetic_data.py`
- Seguro para test accounts (sin validación dominio real)

---

## 🎯 Decisiones Clave de Diseño

### 1️⃣ Campos "Negocio" vs "Inferibles"
- **Negocio** (usuario proporciona): precio, GTIN, link, brand, availability
- **Inferibles** (pipeline genera): título, descripción, género, material, pattern
- **Aplicación**: Milestone 2 valida esta división → tabla de mapping clara

### 2️⃣ "ADDED Only" — Cero Modificaciones
- Proyecto actual (`deep-tagger-api`, `deep-tagger-ui`, `img-puller`) **intacto**
- Nuevas APIs se construyen **encima**, reusando modelos CV existentes
- Riesgo mínimo: si upgrade falla, proyecto sigue funcionando

### 3️⃣ Desarrollo Gratis + Producción Opcional
| Etapa | Costo | Objetivo |
|-------|-------|----------|
| **Development** | $0 | Sandbox Google (test account) |
| **QA** | $0 | Merchant test + Ads test |
| **Production** | $$ | Merchant real + Ads real con billing |

Validación de integración completa sin gastar un peso.

### 4️⃣ Milestones Secuenciales con Dependencias Claras
```
M1 (Enrich) → M2 (Validate) → M3 (Publish) → M4 (Optimize)
   ↓ output       ↓ output       ↓ output
   input ────────→input ────────→input ────────→input
```
Cada uno es **independientemente testeable** con datos de `upgrade/data/`.

---

## 📖 Ruta de Lectura Recomendada

### Para stakeholders / decisores:
1. `upgrade/proposal.md` (2 min)
2. `upgrade/requirements.md` (3 min) — especialmente tabla FREE vs PAID
3. Retorno: entender riesgo + costo + aporte

### Para implementadores (por milestone):
```
1. Lee: upgrade/specs/<milestone>/spec.md
2. Input: upgrade/data/pants_google_500.csv (u otro CSV)
3. Implementa + testa
4. Cuando todos los tests ✅ → next milestone
```

### Para referencia:
- `upgrade/README.md` — visión general
- `upgrade/INDEX.md` — índice completo + FAQ
- `upgrade/tasks.md` — checklist implementación
- `upgrade/data/MANIFEST.md` — detalles de esquema
- `upgrade/data/README.md` — guía uso datos

---

## 🔑 Notas Importantes

✅ **Todos los 3 CSVs** tienen estructura idéntica (21 columnas)  
✅ **Datos realistas pero claramente sintéticos** → seguro para sandbox  
✅ **Regenerable**: `python3 upgrade/generate_synthetic_data.py`  
✅ **Zero credenciales** en repositorio (placeholders, .gitignore en prod)  
✅ **Estructura de proyecto preservada**: código existente intacto, todo es ADDED  

---

## 🚀 Próximos Pasos

1. ✅ Revisar `proposal.md` + `requirements.md` (5 min total)
2. ⚙️ Configurar ambiente gratis Google (Cloud project + test accounts, ~15 min)
3. 🔨 Empezar Milestone 1 (product enrichment)
4. 🧪 Testear contra `upgrade/data/pants_google_500.csv`
5. 🔄 Iterar milestones 2 → 3 → 4

**¿Dudas?** → Ver `upgrade/INDEX.md` (Q&A completo)  
**¿Implementación?** → Ver `upgrade/tasks.md` (checklist detallado)

---

**Última actualización**: Octubre 2026  
**Estado**: Especificación completa, listo para desarrollo  
**Riesgo**: ✅ Bajo (sandbox, datos sintéticos, reuso CV existente)
