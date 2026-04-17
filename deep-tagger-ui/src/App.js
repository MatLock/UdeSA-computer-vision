import { Routes, Route, Navigate } from 'react-router-dom';
import Landing from './pages/Landing/Landing';
import Results from './pages/Results/Results';
import './App.css';

function App() {
  return (
    <div className="App">
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/results" element={<Results />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}

// example of usage https://www.segutecnica.com/images/000000000001756628580remera-azul-segutecnica.png
export default App;
