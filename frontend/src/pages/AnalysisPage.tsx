import { useEffect, useState } from 'react';
import { BrandHeader } from '../components/common/BrandHeader';
import { Button } from '../components/common/Button';
import { getMove } from '../api/moves';
import { ApiError } from '../api/client';
import { navigate, routes } from '../app/routes';
import readyMascot from '../assets/analysis-mascot-ready.png';
import failedMascot from '../assets/analysis-mascot-failed.png';
import checkIcon from '../assets/analysis-check.svg';
import warningIcon from '../assets/analysis-warning.svg';
import type { MoveSnapshot } from '../types/move';

type AnalysisScreen = 'waiting' | 'complete' | 'failed';

export function AnalysisPage() {
  const params = new URLSearchParams(window.location.search);
  const afterMove = params.get('mode') === 'after';
  const [snapshot, setSnapshot] = useState<MoveSnapshot | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [retryKey, setRetryKey] = useState(0);
  const saved = sessionStorage.getItem('jimssak-member-move') || sessionStorage.getItem('jimssak-guest-move');
  let move: { move_id: string; guest_token: string } | null = null;
  try { move = saved ? JSON.parse(saved) : null; } catch { /* Invalid session data is treated as missing. */ }
  const moveId = move?.move_id;
  const guestToken = move?.guest_token;
  const retry = () => { setLoading(true); setRetryKey(key => key + 1); };

  useEffect(() => {
    if (!moveId || afterMove) return;
    let active = true;
    const refresh = () => {
      void getMove(moveId, guestToken).then(value => { if (active) { setSnapshot(value); setError(''); } })
        .catch(cause => { if (active) setError(cause instanceof ApiError && cause.status === 401 ? '로그인이 만료됐어요. 다시 로그인해주세요.' : '분석 상태를 불러오지 못했어요. 잠시 후 다시 확인해주세요.'); })
        .finally(() => { if (active) setLoading(false); });
    };
    refresh();
    const timer = window.setInterval(refresh, 3000);
    return () => { active = false; window.clearInterval(timer); };
  }, [moveId, guestToken, afterMove, retryKey]);

  const status = snapshot?.data.analysis.status;
  const screen: AnalysisScreen = status === 'completed' ? 'complete' : status === 'failed' ? 'failed' : 'waiting';

  useEffect(() => {
    if (status !== 'completed' || snapshot?.data.analysis.demo || afterMove) return;
    const timer = window.setTimeout(() => navigate(routes.chat), 2500);
    return () => window.clearTimeout(timer);
  }, [status, snapshot?.data.analysis.demo, afterMove]);

  if (!moveId) return <main className="app-page analysis-page"><BrandHeader />
    <div className="analysis-content"><h1>이사 정보를 찾을 수 없어요</h1><Button onClick={() => navigate(routes.capture)}>촬영 준비로 돌아가기</Button></div>
  </main>;

  if (loading && !snapshot && !afterMove) return <main className="app-page analysis-page"><BrandHeader />
    <div className="analysis-content"><h1>분석 상태를 확인하고 있어요</h1></div>
  </main>;

  if (afterMove) return <main className="app-page analysis-page"><BrandHeader />
    <div className="analysis-content"><h1>이사 후 비교를 준비하고 있어요</h1><p className="analysis-intro">이사 전후 비교 결과 API가 연결되면 결과를 확인할 수 있어요.</p><Button variant="secondary" onClick={() => navigate(routes.history)}>나의 이사 기록으로 돌아가기</Button></div>
  </main>;

  if (snapshot?.data.room_video.status === 'empty') return <main className="app-page analysis-page"><BrandHeader />
    <div className="analysis-content"><h1>촬영한 영상이 없어요</h1><p className="analysis-intro">방 영상을 촬영하거나 업로드하면 분석을 시작할 수 있어요.</p><Button variant="secondary" onClick={() => navigate(routes.capture)}>촬영 준비로 돌아가기</Button></div>
  </main>;

  if (snapshot?.data.analysis.demo) return <main className="app-page analysis-page"><BrandHeader />
    <div className="analysis-content"><h1>실제 분석 결과를 기다리고 있어요</h1><p className="analysis-intro">데모 분석 결과는 이삿짐 목록으로 표시하지 않아요.</p><Button variant="secondary" onClick={retry}>다시 확인하기</Button></div>
  </main>;

  if (error && !snapshot) return <main className="app-page analysis-page"><BrandHeader />
    <div className="analysis-content"><h1>분석 상태를 불러오지 못했어요</h1><p className="analysis-intro">{error}</p><Button variant="secondary" onClick={() => error.includes('만료') ? navigate(routes.login) : retry()}>{error.includes('만료') ? '로그인하기' : '다시 확인하기'}</Button></div>
  </main>;

  return <main className="app-page analysis-page">
    <BrandHeader />
    <div className="analysis-content">
      <h1>{afterMove ? screen === 'waiting' ? '이사 후 영상을 분석하고 있어요' : screen === 'complete' ? '이사 전후 영상 비교를 마쳤어요' : '영상을 분석하지 못했어요' : '이사 정보가 저장됐어요'}</h1>
      <p className="analysis-intro">{afterMove ? screen === 'failed' ? '일시적인 문제로 이사 후 영상을 확인하지 못했어요. 다시 시도해주세요.' : screen === 'complete' ? '상태가 달라 보이거나 영상에서 찾지 못한 물품이 있는지 목록에서 확인해주세요.' : '이사 전 영상과 비교해 상태가 달라 보이거나 영상에서 찾지 못한 물품을 살펴보고 있어요.' : <>이삿짐 목록을 마무리하고 있어요.<br />분석이 끝나면 목록 확인으로 이동해요.</>}</p>
      <div className="analysis-mascot-frame"><img src={screen === 'failed' ? failedMascot : readyMascot} alt="" /></div>
      <div className={`analysis-state analysis-state--${screen}`} role="status">
        {screen === 'waiting' ? <><span className="analysis-spinner" aria-hidden="true" />영상 분석 중</>
          : screen === 'complete' ? <><img src={checkIcon} alt="" />영상 분석 완료!</>
          : <><img src={warningIcon} alt="" />영상 분석 실패!</>}
      </div>
      {screen === 'failed'
        ? <Button variant="secondary" onClick={() => navigate(routes.capture)}>영상 다시 촬영하기</Button>
        : !afterMove && <p className="analysis-detail">입력한 정보는 그대로 유지돼요.</p>}
      {error && <p className="analysis-error" role="alert">{error}</p>}
    </div>
  </main>;
}
