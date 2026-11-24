import { useState } from 'react';
import { navigate, routes } from '../app/routes';
import { Button } from '../components/common/Button';

type Filter = 'all' | 'progress' | 'done';

export function HistoryPage() {
  const [filter, setFilter] = useState<Filter>('all');
  // 한 기록의 현재 상태 한 건만 표시한다. 디자인의 네 카드는 같은 기록의 상태별 예시다.
  const record = { status: 'progress' as const, date: '2026년 11월 20일 이사', route: '서울 마포구 → 서울 성동구', count: '이삿짐 5개 · 원룸' };
  const visible = filter === 'all' || filter === record.status;

  return <main className="app-page history-page">
    <header className="sub-header"><strong>나의 이사 기록</strong><button type="button" aria-label="닫기" onClick={() => navigate(routes.home)}>×</button></header>
    <div className="history-content"><span className="eyebrow">이사 기록</span>
      <h1>차곡차곡 쌓인 나의 이사</h1>
      <p>이사 준비부터 이사 후 물품 확인까지,<br />진행 중인 이사를 이어가고 지난 결과를 확인해요.</p>
      <div className="history-filters" aria-label="이사 기록 필터">
        {([['all', '전체'], ['progress', '진행중'], ['done', '완료']] as const).map(([key, label]) =>
          <button type="button" key={key} aria-pressed={filter === key} className={filter === key ? 'active' : ''} onClick={() => setFilter(key)}>{label}</button>)}
      </div>
      {visible ? <article className="history-card">
        <div className="history-status"><span>진행중</span><strong>이사 준비중</strong></div>
        <h2>{record.date}</h2><hr />
        <strong className="history-route">{record.route}</strong>
        <p>{record.count}</p><p>명세서를 업체에 보내 견적을 문의해보세요.</p>
        <Button disabled>이사 기록 보기</Button>
      </article> : <p className="history-empty">해당하는 이사 기록이 없어요.</p>}
    </div>
  </main>;
}
