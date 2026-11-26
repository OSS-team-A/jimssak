import { useEffect, useState } from 'react';
import { BrandHeader } from '../components/common/BrandHeader';
import { Button } from '../components/common/Button';
import { getMove } from '../api/moves';
import { navigate, routes } from '../app/routes';
import readyMascot from '../assets/analysis-mascot-ready.png';
import failedMascot from '../assets/analysis-mascot-failed.png';
import checkIcon from '../assets/analysis-check.svg';
import warningIcon from '../assets/analysis-warning.svg';
import type { MoveSnapshot } from '../types/move';

type AnalysisScreen = 'waiting' | 'complete' | 'failed';

export function AnalysisPage() {
  const [snapshot, setSnapshot] = useState<MoveSnapshot | null>(null);
  const [error, setError] = useState('');
  const saved = sessionStorage.getItem('jimssak-member-move') || sessionStorage.getItem('jimssak-guest-move');
  let move: { move_id: string; guest_token: string } | null = null;
  try { move = saved ? JSON.parse(saved) : null; } catch { /* Invalid session data is treated as missing. */ }
  const moveId = move?.move_id;
  const guestToken = move?.guest_token;

  useEffect(() => {
    if (!moveId) return;
    let active = true;
    const refresh = () => {
      void getMove(moveId, guestToken).then(value => { if (active) { setSnapshot(value); setError(''); } })
        .catch(() => { if (active) setError('분석 상태를 불러오지 못했어요. 잠시 후 다시 확인해주세요.'); });
    };
    refresh();
    const timer = window.setInterval(refresh, 5000);
    return () => { active = false; window.clearInterval(timer); };
  }, [moveId, guestToken]);

  const status = snapshot?.data.analysis.status;
  const screen: AnalysisScreen = status === 'completed' ? 'complete' : status === 'failed' ? 'failed' : 'waiting';

  if (!moveId) return <main className="app-page analysis-page"><BrandHeader />
    <div className="analysis-content"><h1>이사 정보를 찾을 수 없어요</h1><Button onClick={() => navigate(routes.capture)}>촬영 준비로 돌아가기</Button></div>
  </main>;

  return <main className="app-page analysis-page">
    <BrandHeader />
    <div className="analysis-content">
      <h1>이사 정보가 저장됐어요</h1>
      <p className="analysis-intro">이삿짐 목록을 마무리하고 있어요.<br />분석이 끝나면 목록 확인으로 이동해요.</p>
      <div className="analysis-mascot-frame"><img src={screen === 'failed' ? failedMascot : readyMascot} alt="" /></div>
      <div className={`analysis-state analysis-state--${screen}`} role="status">
        {screen === 'waiting' ? <><span className="analysis-spinner" aria-hidden="true" />영상 분석 중</>
          : screen === 'complete' ? <><img src={checkIcon} alt="" />영상 분석 완료!</>
          : <><img src={warningIcon} alt="" />영상 분석 실패!</>}
      </div>
      {screen === 'failed'
        ? <Button variant="secondary" onClick={() => navigate(routes.capture)}>영상 다시 촬영하기</Button>
        : <p className="analysis-detail">입력한 정보는 그대로 유지돼요.</p>}
      {error && <p className="analysis-error" role="alert">{error}</p>}
    </div>
  </main>;
}
