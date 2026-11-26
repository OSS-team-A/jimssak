import { useState } from 'react';
import { BrandHeader } from '../components/common/BrandHeader';
import { Button } from '../components/common/Button';
import { navigate, routes } from '../app/routes';
import { useCurrentMove } from '../hooks/useCurrentMove';
import type { ComparisonItem } from '../types/move';
import avatar from '../assets/chatAvatar.svg';

const label: Record<ComparisonItem['status'], string> = { unchanged: '변화 없음', suspected: '상태 변화 의심', not_visible: '확인 필요' };
type Decision = 'checked' | 'not_found';

export function ComparisonPage() {
  const { move, snapshot, loading, error, refresh } = useCurrentMove();
  const [open, setOpen] = useState(false);
  const [expanded, setExpanded] = useState<string | null>(null);
  const key = move ? `jimssak-comparison-draft-${move.id}` : '';
  const [decisions, setDecisions] = useState<Record<string, Decision>>(() => {
    try { return JSON.parse(sessionStorage.getItem(key) || '{}') as Record<string, Decision>; }
    catch { return {}; }
  });
  const items = snapshot?.postmove?.status === 'ready' ? snapshot.postmove.items || [] : [];
  const choose = (id: string, choice: Decision) => {
    const next = { ...decisions, [id]: choice }; setDecisions(next); if (key) sessionStorage.setItem(key, JSON.stringify(next));
  };
  const issues = items.filter(item => item.status !== 'unchanged' && decisions[item.id] !== 'checked');
  return <main className="app-page comparison-page">
    <BrandHeader />
    <div className="comparison-scroll">
      <div className="chat-speaker"><img src={avatar} alt="" /><strong>짐싹</strong></div>
      <p className="chat-prompt">이사 전후 영상을 비교했어요.</p>
      {loading ? <p className="inventory-feedback" role="status">비교 결과를 불러오는 중이에요.</p> : error ? <div className="inventory-feedback" role="alert">{error}<button type="button" onClick={() => void refresh()}>다시 시도하기</button></div> : snapshot?.postmove?.status === 'failed' ? <div className="inventory-feedback" role="alert">{snapshot.postmove.error || '비교에 실패했어요.'}<button type="button" onClick={() => navigate(routes.postmove)}>다시 촬영하기</button></div> : !items.length ? <div className="inventory-feedback" role="status">아직 확인할 전후 비교 결과가 없어요.<button type="button" onClick={() => navigate(routes.postmove)}>이사 후 촬영으로 돌아가기</button></div> : <>
        <p className="comparison-summary">{issues.length ? `확인이 필요한 물품 ${issues.length}개가 있어요.` : '확인이 필요한 물품이 없어요.'}</p>
        <button className="chat-summary" type="button" onClick={() => setOpen(true)}><strong>이사 후 이삿짐 목록 <span>›</span></strong><hr />{items.slice(0, 5).map(item => <span className="chat-summary-row" key={item.id}>{item.name}<b>{label[item.status]}</b></span>)}<hr /><small>누르면 자세히 볼 수 있어요</small></button>
        <Button onClick={() => setOpen(true)}>목록 확인하기</Button>
      </>}
    </div>
    {open && <div className="comparison-document" role="dialog" aria-modal="true" aria-label="이사 후 이삿짐 목록"><header className="inventory-header"><strong>이삿짐 목록</strong><button type="button" aria-label="닫기" onClick={() => setOpen(false)}>×</button></header><div className="inventory-body"><h1>이사 후 이삿짐 목록</h1><p className="inventory-subtitle">이사 후 촬영한 영상 분석을 바탕으로 작성한 목록이에요.</p>{items.map(item => <div className={`comparison-item ${item.status !== 'unchanged' && decisions[item.id] !== 'checked' ? 'comparison-item--attention' : ''}`} key={item.id}><button type="button" onClick={() => setExpanded(expanded === item.id ? null : item.id)} aria-expanded={expanded === item.id}><strong>{item.name}</strong><span>{decisions[item.id] === 'checked' ? '직접 확인' : label[item.status]}　{expanded === item.id ? '⌃' : '⌄'}</span></button>{expanded === item.id && <div className="comparison-details">{item.note && <p>{item.note}</p>}{(item.before_photo_url || item.after_photo_url) && <div className="comparison-photos">{item.before_photo_url && <figure><img src={item.before_photo_url} alt={`${item.name} 이사 전`} /><figcaption>이사 전</figcaption></figure>}{item.after_photo_url && <figure><img src={item.after_photo_url} alt={`${item.name} 이사 후`} /><figcaption>이사 후</figcaption></figure>}</div>}{item.status !== 'unchanged' && <div className="comparison-decisions"><button type="button" onClick={() => choose(item.id, 'checked')}>직접 확인했어요</button><button type="button" onClick={() => choose(item.id, 'not_found')}>아직 못 찾았어요</button></div>}</div>}</div>)}<p className="inventory-unsaved" role="status">확인 결과는 이 브라우저에 임시 보관돼요. 서버 저장·명세서 확정 기능은 아직 연결되지 않았어요.</p></div></div>}
  </main>;
}
