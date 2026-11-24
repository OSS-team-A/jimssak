import { useEffect, useState } from 'react';
import { BrandHeader } from '../components/common/BrandHeader';
import { Button } from '../components/common/Button';
import { getMove } from '../api/moves';
import { navigate, routes } from '../app/routes';
import mascot from '../assets/analysis-mascot.webp';
import type { MoveSnapshot } from '../types/move';

export function AnalysisPage() {
  const [snapshot, setSnapshot] = useState<MoveSnapshot | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const saved = sessionStorage.getItem('jimssak-member-move') || sessionStorage.getItem('jimssak-guest-move');
  let move: { move_id: string; guest_token: string } | null = null;
  try { move = saved ? JSON.parse(saved) : null; } catch { /* Bad local data is handled as missing. */ }
  const moveId = move?.move_id;
  const guestToken = move?.guest_token;
  const refresh = () => {
    if (!moveId) return;
    setLoading(true); setError('');
    void getMove(moveId, guestToken).then(setSnapshot)
      .catch(() => setError('상태를 불러오지 못했어요. 다시 시도해주세요.'))
      .finally(() => setLoading(false));
  };
  useEffect(() => { refresh(); }, [moveId, guestToken]);
  return <main className="app-page analysis-page">
    <BrandHeader />
    <h1>이사 정보가 저장됐어요</h1>
    <p>이삿짐 목록을 마무리하고 있어요.<br />분석이 끝나면 목록 확인으로 이동해요.</p>
    <img className="analysis-mascot" src={mascot} alt="" />
    <div className="analysis-state" role="status">{loading ? '이사 상태 확인 중' :
      snapshot?.data.analysis.status === 'not_started' ? '영상 분석 기능이 아직 연결되지 않았어요.' :
      error || '영상 분석 대기 중'}</div>
    <p className="analysis-detail">입력한 정보는 그대로 유지돼요.</p>
    <div className="analysis-actions">
      <Button variant="secondary" onClick={() => navigate(routes.moveInfo)}>이사 정보 확인하기</Button>
      <Button onClick={refresh} disabled={loading}>상태 다시 확인하기</Button>
    </div>
  </main>;
}
