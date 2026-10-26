# Requisitos: cuentas de prueba (gratis) y producción (pago)

Para correr el pipeline contra Google hacen falta cuentas, credenciales y datos de producto. Google ofrece cuentas de prueba que permiten validar la integración completa sin costo. La producción con pago solo hace falta para mostrar anuncios reales y medir su rendimiento.

## Tabla de requisitos

| Requisito | Gratis (cuenta de prueba) | Pago (producción) |
|---|---|---|
| Cuenta de Google Merchant Center | Cuenta de prueba, aislada de los datos reales; no muestra productos a nadie | Cuenta real verificada (dominio y datos del negocio) |
| Cuenta de Google Ads | Cuenta de prueba: **no muestra anuncios** ni cobra | Cuenta activa con **facturación** configurada |
| Acceso a la Google Ads API | Nivel Test: 15.000 operaciones por día, solo sobre cuentas de prueba. Se gestiona desde Google Cloud Console; los developer tokens se dieron de baja el 2026-09-09 | Niveles Explorer, Basic (15.000 operaciones por día) o Standard (sin límite) |
| Credenciales | Cuenta de servicio (evita los tokens OAuth manuales) | OAuth2 o cuenta de servicio, más Customer ID y Merchant ID |
| Proyecto de Google Cloud | Necesario (gratis) para habilitar las APIs | Necesario |
| Datos de producto | Sirven datos inventados (precios, GTIN). No se publica nada | **Datos reales obligatorios**: GTIN válido si el producto lo tiene, precio real, link activo, stock |
| Anuncios visibles | No: solo se valida que el código corre | Sí: se muestran y compiten en subastas |
| Costo por clic o impresión | $0, no hay subasta real | Costo real por clic o por conversión |
| Qué se puede validar | Que el código sube el feed, crea campañas y lee estados y métricas (en cero) | Posicionamiento real y métricas reales (CTR, ROAS, conversiones) |

## Notas

- **Desarrollo gratis completo:** una cuenta de prueba de Merchant Center, una de Google Ads y un proyecto de Google Cloud alcanzan para validar el flujo de punta a punta (feed → publicación → campaña → estado) sin gasto y sin anuncios públicos.
- **Límite del modo gratis:** las cuentas de prueba no muestran anuncios ni devuelven métricas reales. Validan que la integración funciona, no el posicionamiento. La regla de la etapa 4 solo se puede probar de verdad en producción, con gasto.
- **Merchant API:** la Content API for Shopping se dio de baja el 2026-08-18. El pipeline apunta a la Merchant API, que tiene sus propias cuentas de prueba.
- **Estado del código:** las llamadas reales a las dos APIs todavía no están implementadas. Hoy los tests corren con las APIs simuladas ([`README.md`](README.md), sección Estado).

## Orden recomendado para armar el entorno gratis

1. Crear un proyecto en Google Cloud Console.
2. Habilitar **Merchant API** y **Google Ads API**.
3. Crear una **cuenta de prueba** en Merchant Center.
4. Crear una **cuenta de prueba** en Google Ads.
5. Configurar una **cuenta de servicio** y bajar su clave JSON.
6. Guardar las credenciales fuera del repo (variables de entorno o un gestor de secretos) y nunca commitear claves.

Variables de entorno que lee `positioner/publishing/` (`credential_loader.py` las dos primeras, `environment.py` la tercera):

```bash
GOOGLE_SERVICE_ACCOUNT_PATH=/ruta/a/la/clave.json
MERCHANT_ID=123456789
MERCHANT_ENV=test          # o "production"
```

## Fuentes (consultadas el 2026-10-26)

- [Google Ads API: cuentas de prueba](https://developers.google.com/google-ads/api/docs/best-practices/test-accounts)
- [Google Ads API: niveles de acceso](https://developers.google.com/google-ads/api/docs/api-policy/access-levels)
- [Merchant API: cuentas de prueba](https://developers.google.com/merchant/api/guides/accounts/test-accounts)
- [Content API for Shopping: baja](https://developers.google.com/shopping-content/guides/sunset)
- [Nueva experiencia de alta en la Google Ads API (2026-09-10)](https://ads-developers.googleblog.com/2026/09/new-onboarding-experience-for-google-ads-api.html)

Precios y políticas pueden cambiar: antes de pasar a producción hay que contrastarlos con la documentación oficial de Google.
