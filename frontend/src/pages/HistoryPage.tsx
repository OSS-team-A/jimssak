import { useEffect, useState } from 'react';
import { navigate, routes } from '../app/routes';
import { Button } from '../components/common/Button';
import { getMove } from '../api/moves';
import type { MoveSnapshot } from '../types/move';

type Filter = 'all' | 'progress' | 'done';

export function HistoryPage() {
  const [filter, setFilter] = useState<Filter>('all');
  const [record, setRecord] = useState<MoveSnapshot | null>(null);
  const [error, setError] = useState('');
  useEffect(() => {
    const saved = sessionStorage.getItem('jimssak-member-move');
    if (!saved) return;
    try {
      const { move_id, guest_token } = JSON.parse(saved) as { move_id: string; guest_token: string };
      void getMove(move_id, guest_token).then(setRecord).catch(() => setError('이사 기록을 불러오지 못했어요.'));
    } catch { sessionStorage.removeItem('jimssak-guest-move'); }
  }, []);
  const visible = record && filter !== 'done';

  return <main className="app-page history-page">
    <header className="sub-header"><strong>나의 이사 기록</strong><button type="button" aria-label="닫기" onClick={() => navigate(routes.home)}>×</button></header>
    <div className="history-content"><span className="eyebrow">이사 기록</span>
      <h1>차곡차곡 쌓인 나의 이사</h1>
      <p>이사 준비부터 이사 후 물품 확인까지,<br />진행 중인 이사를 이어가고 지난 결과를 확인해요.</p>
      <div className="history-filters" aria-label="이사 기록 필터">
        {([['all', '전체'], ['progress', '진행중'], ['done', '완료']] as const).map(([key, label]) =>
          <button type="button" key={key} aria-pressed={filter === key} className={filter === key ? 'active' : ''} onClick={() => setFilter(key)}>{label}</button>)}
      </div>
      {error && <p role="alert">{error}</p>}
      {visible ? <article className="history-card">
        <div className="history-status"><span>진행중</span><strong>이사 준비중</strong></div>
        <h2>{record.data.schedule.value?.move_date ?? '이사 날짜 입력 전'}</h2><hr />
        <strong className="history-route">{record.data.origin.value?.road_address ?? '출발지 입력 전'} → {record.data.destination.value?.road_address ?? '도착지 입력 전'}</strong>
        <p>이사 정보를 이어서 입력해주세요.</p>
        <Button onClick={() => navigate(routes.moveInfo)}>이사 기록 보기</Button>
      </article> : <p className="history-empty">이사 기록이 없어요.</p>}
    </div>
  </main>;
}
