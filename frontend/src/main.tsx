import { createRoot } from 'react-dom/client'
import './styles/global.css'
import App from './App.tsx'

// StrictMode disabled: PDF rendering via canvas has incompatible double-render behavior
createRoot(document.getElementById('root')!).render(<App />)
