# Deep Tagger

Aplicación web para el etiquetado de imágenes con IA, desarrollada como parte del proyecto del curso de Computer Vision en la Universidad de San Andrés (UdeSA).

Los usuarios envían la URL de una imagen de producto y la aplicación devuelve predicciones generadas por IA, que incluyen el título, el tipo, la descripción y las etiquetas visuales superpuestas sobre la imagen.

## Características

- Entrada de URL de imagen con validación
- Predicción de producto con IA (título, tipo, descripción, etiquetas)
- Superposición animada de etiquetas sobre la imagen
- Interfaz responsive con Material Design

## Stack Tecnológico

- **React** 19 con React Router
- **Material-UI (MUI)** v9 para componentes y temas
- **Emotion** para estilos CSS-in-JS
- **Create React App** como herramienta de build

## Primeros Pasos

### Requisitos previos

- Node.js (se recomienda v16 o superior)
- npm

### Instalación

```bash
cd deep-tagger-ui
npm install
```

### Desarrollo

```bash
npm start
```

Abre la aplicación en [http://localhost:3000](http://localhost:3000).

### Build de producción

```bash
npm run build
```

## Estructura del Proyecto

```
deep-tagger-ui/
├── src/
│   ├── pages/
│   │   ├── Landing/    # Página de inicio con entrada de URL
│   │   └── Results/    # Resultados de predicción con etiquetas superpuestas
│   ├── components/
│   │   ├── Loading/    # Spinner de carga
│   │   └── Toast/      # Notificaciones tipo toast
│   ├── services/
│   │   └── api.js      # Cliente de la API de predicción
│   ├── theme.js        # Tema personalizado de MUI
│   └── App.js          # Componente raíz con routing
```
