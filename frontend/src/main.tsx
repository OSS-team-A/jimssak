import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './app/App';
import '@fontsource/pretendard/400.css';
import '@fontsource/pretendard/600.css';
import '@fontsource/pretendard/700.css';
import './styles/tokens.css';
import './style.css';

createRoot(document.getElementById('root')!).render(<StrictMode><App /></StrictMode>);
