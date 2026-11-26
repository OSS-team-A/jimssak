import { useEffect, useState } from 'react';
import { HomePage } from '../pages/HomePage';
import { GuidePage } from '../pages/GuidePage';
import { AuthPage } from '../pages/AuthPage';
import { HistoryPage } from '../pages/HistoryPage';
import { CapturePage } from '../pages/CapturePage';
import { MoveInfoPage } from '../pages/MoveInfoPage';
import { AnalysisPage } from '../pages/AnalysisPage';
import { ChatPage, InventoryPage, ResultPage } from '../pages/ReviewPages';
import { routes } from './routes';

export function App() {
  const [path, setPath] = useState(window.location.pathname);
  useEffect(() => {
    const sync = () => setPath(window.location.pathname);
    window.addEventListener('popstate', sync);
    return () => window.removeEventListener('popstate', sync);
  }, []);

  if (path === routes.capture) return <CapturePage />;
  if (path === routes.moveInfo) return <MoveInfoPage />;
  if (path === routes.analysis) return <AnalysisPage />;
  if (path === routes.chat) return <ChatPage />;
  if (path === routes.inventory) return <InventoryPage />;
  if (path === routes.result) return <ResultPage />;
  if (path === routes.home) return <HomePage />;
  if (path === routes.guide) return <GuidePage />;
  if (path === routes.login) return <AuthPage mode="login" />;
  if (path === routes.signup) return <AuthPage mode="signup" />;
  if (path === routes.history) return <HistoryPage />;
  return <HomePage />;
}
