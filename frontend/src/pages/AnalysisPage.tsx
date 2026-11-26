import { useEffect } from 'react';
import { BrandHeader } from '../components/common/BrandHeader';
import { Button } from '../components/common/Button';
import { navigate, routes } from '../app/routes';
import { useCurrentMove } from '../hooks/useCurrentMove';
import mascot from '../assets/analysis-mascot.webp';
import failedMascot from '../assets/analysis-failed.webp';
import failedIcon from '../assets/analysis-error.svg';

export function AnalysisPage() {
  const { snapshot, loading, error, refresh } = useCurrentMove();
  const status = snapshot?.data.analysis.status;
  const completed = status === 'completed' && snapshot?.data.inventory.status === 'ready';
  useEffect(() => { if (completed) navigate(routes.chat); }, [completed]);
  useEffect(() => {
    if (status !== 'queued' && status !== 'running') return;
    const timer = window.setInterval(() => void refresh(), 5000);
    return () => window.clearInterval(timer);
  }, [status, refresh]);
  return <main className="app-page analysis-page">
    <BrandHeader />
    <div className="analysis-content">
      <h1>이사 정보가 저장됐어요</h1>
      <p>이삿짐 목록을 마무리하고 있어요.<br />분석이 끝나면 목록 확인으로 이동해요.</p>
      <img className="analysis-mascot" src={status === 'failed' ? failedMascot : mascot} alt="" />
      {loading && !snapshot ? <div className="analysis-state" role="status">이사 상태 확인 중</div> : error ? <div className="analysis-state analysis-state--failed" role="alert">{error}</div> : status === 'failed' ? <div className="analysis-state analysis-state--failed" role="alert"><img src={failedIcon} alt="" />영상 분석 실패!{snapshot?.data.analysis.error && <small>{snapshot.data.analysis.error}</small>}</div> : completed ? <div className="analysis-state analysis-state--done" role="status">영상 분석 완료</div> : status === 'queued' || status === 'running' ? <div className="analysis-state" role="status">영상 분석 중이에요.</div> : <div className="analysis-state" role="status">영상 분석 기능이 아직 연결되지 않았어요.</div>}
      {(status === 'not_started' || status === 'empty' || !snapshot) && <p className="analysis-detail">입력한 정보는 그대로 유지돼요.</p>}
      {status === 'failed' && <p className="analysis-detail">다시 분석 요청하는 기능은 아직 연결되지 않았어요.</p>}
      <div className="analysis-actions"><Button variant="secondary" onClick={() => navigate(routes.moveInfo)}>이사 정보 확인하기</Button><Button onClick={() => void refresh()} disabled={loading}>상태 다시 확인하기</Button></div>
    </div>
  </main>;
}
