import { useEffect, useState } from 'react';
import { AnalysisDemoPage } from '../pages/AnalysisDemoPage';
import { HomePage } from '../pages/HomePage';
import { GuidePage } from '../pages/GuidePage';
import { AuthPage } from '../pages/AuthPage';
import { HistoryPage } from '../pages/HistoryPage';
import { routes } from './routes';

export function App() {
  const [path, setPath] = useState(window.location.pathname);
  useEffect(() => {
    const sync = () => setPath(window.location.pathname);
    window.addEventListener('popstate', sync);
    return () => window.removeEventListener('popstate', sync);
  }, []);

  if (path === routes.demo) return <AnalysisDemoPage />;
  if (path === routes.home) return <HomePage />;
  if (path === routes.guide) return <GuidePage />;
  if (path === routes.login) return <AuthPage mode="login" />;
  if (path === routes.signup) return <AuthPage mode="signup" />;
  if (path === routes.history) return <HistoryPage />;
  return <main className="app-page"><h1>화면을 찾을 수 없어요</h1><a href={routes.home}>처음으로 돌아가기</a></main>;
}
