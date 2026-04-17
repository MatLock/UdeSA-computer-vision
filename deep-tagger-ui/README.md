# Deep Tagger

A web application for AI-powered image tagging built as part of a Computer Vision course project at Universidad de San Andrés (UdeSA).

Users submit a product image URL, and the application returns AI-generated predictions including the product title, type, description, and visual tags overlaid on the image.

## Features

- Image URL input with validation
- AI-powered product prediction (title, type, description, tags)
- Animated tag overlays displayed on the image
- Responsive Material Design UI

## Tech Stack

- **React** 19 with React Router
- **Material-UI (MUI)** v9 for components and theming
- **Emotion** for CSS-in-JS styling
- **Create React App** for build tooling

## Getting Started

### Prerequisites

- Node.js (v16 or higher recommended)
- npm

### Installation

```bash
cd deep-tagger-ui
npm install
```

### Development

```bash
npm start
```

Opens the app at [http://localhost:3000](http://localhost:3000).

### Production Build

```bash
npm run build
```

## Project Structure

```
deep-tagger-ui/
├── src/
│   ├── pages/
│   │   ├── Landing/    # Home page with URL input
│   │   └── Results/    # Prediction results with tag overlays
│   ├── components/
│   │   ├── Loading/    # Loading spinner
│   │   └── Toast/      # Notification toasts
│   ├── services/
│   │   └── api.js      # Prediction API client
│   ├── theme.js        # MUI custom theme
│   └── App.js          # Root component with routing
```
