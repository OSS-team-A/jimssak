import { useEffect, useState } from 'react';
import { closeHistory, navigate, routes } from '../app/routes';
import { Button } from '../components/common/Button';
import { BrandHeader } from '../components/common/BrandHeader';
import { hasSession } from '../api/auth';
import { getMove, listAllMoves } from '../api/moves';
import { ApiError } from '../api/client';
import type { MoveSummary } from '../types/move';
import { moveDate, shortPlace } from '../features/historyAdapter';
import closeIcon from '../assets/history-close.svg';
import chatAvatar from '../assets/chat-avatar.svg';
import listIcon from '../assets/chat-list.svg';
import downloadIcon from '../assets/chat-download-active.svg';
import sendIcon from '../assets/chat-send.svg';

type Filter = 'all' | 'progress' | 'done';
type Stage = MoveSummary['stage'];
const copy: Record<Stage, { status: string; badge: string; description: string; button: string }> = {
  preparing: { status: '이사 준비중', badge: '진행중', description: '명세서를 업체에 보내 견적을 문의해보세요.', button: '이사 기록 보기' },
  after_move: { status: '이사 후 확인 전', badge: '진행중', description: '이사를 마쳤다면 이사 후 물품을 촬영해보세요.', button: '이사 후 확인 시작하기' },
  comparing: { status: '물품 검토 중', badge: '진행중', description: '이사 후 물품을 확인해주세요.', button: '물품 검토 이어가기' },
  done: { status: '검토 완료', badge: '완료', description: '확인한 물품을 다시 살펴보세요.', button: '검토 결과 보기' },
};

function HistoryCard({ record, onOpen }: { record: MoveSummary; onOpen: () => void }) {
  const state = copy[record.stage];
  return <article className="history-card">
    <div className="history-status"><span className={record.stage === 'done' ? 'history-status-done' : ''}>{state.badge}</span><strong>{state.status}</strong></div>
    <h2>{moveDate(record.move_date)}</h2><hr />
    <strong className="history-route">{shortPlace(record.origin_address)} → {shortPlace(record.destination_address)}</strong>
    <p className="history-items">이삿짐 목록은 기록에서 확인할 수 있어요.</p>
    <p className="history-description">{state.description}</p>
    <Button onClick={onOpen}>{state.button}</Button>
  </article>;
}

export function HistoryPage() {
  const [filter, setFilter] = useState<Filter>('all');
  const [records, setRecords] = useState<MoveSummary[]>([]);
  const [loading, setLoading] = useState(hasSession());
  const [error, setError] = useState('');
  const [opening, setOpening] = useState(false);
  const [retryKey, setRetryKey] = useState(0);

  useEffect(() => {
    if (!hasSession()) return;
    let active = true;
    async function load() {
      setLoading(true);
      setError('');
      try {
        const all = await listAllMoves();
        if (active) setRecords(all);
      } catch (cause) { if (active) setError(cause instanceof ApiError && cause.status === 401 ? '로그인이 만료됐어요. 다시 로그인해주세요.' : '이사 기록을 불러오지 못했어요. 잠시 후 다시 시도해주세요.'); }
      finally { if (active) setLoading(false); }
    }
    void load();
    return () => { active = false; };
  }, [retryKey]);

  async function openRecord(record: MoveSummary) {
    if (opening) return;
    setOpening(true);
    try {
      const snapshot = await getMove(record.move_id);
      sessionStorage.setItem('jimssak-member-move', JSON.stringify({ move_id: record.move_id, guest_token: '' }));
      if (record.stage !== 'preparing') window.history.pushState({}, '', `${routes.historyDetail}?moveId=${record.move_id}`);
      else if (snapshot.data.room_video.status === 'empty') navigate(routes.capture);
      else if (snapshot.missing.some(step => step !== 'room_video')) navigate(routes.moveInfo);
      else if (snapshot.data.analysis.status === 'completed') navigate(routes.chat);
      else navigate(routes.analysis);
      if (record.stage !== 'preparing') window.dispatchEvent(new PopStateEvent('popstate'));
    } catch { setError('선택한 이사 기록을 열지 못했어요. 잠시 후 다시 시도해주세요.'); }
    finally { setOpening(false); }
  }

  const visible = records.filter(record => filter === 'all' || (filter === 'done' ? record.stage === 'done' : record.stage !== 'done'));
  return <main className={`app-page history-page${!loading && !hasSession() ? ' history-page--guest' : ''}`}>
    <header className="sub-header"><strong>나의 이사 기록</strong><button type="button" aria-label="닫기" onClick={closeHistory}><img src={closeIcon} alt="" /></button></header>
    <div className="history-content"><span className="eyebrow">이사 기록</span>
      <h1>차곡차곡 쌓인 나의 이사</h1>
      <p>이사 준비부터 이사 후 물품 확인까지,<br />진행 중인 이사를 이어가고 지난 결과를 확인해요.</p>
      <div className="history-filters" aria-label="이사 기록 필터">
        {([['all', '전체'], ['progress', '진행중'], ['done', '완료']] as const).map(([key, label]) =>
          <button type="button" key={key} aria-pressed={filter === key} className={filter === key ? 'active' : ''} onClick={() => setFilter(key)}>{label}</button>)}
      </div>
      {error && <><p className="history-error" role="alert">{error}</p><Button variant="secondary" onClick={() => error.includes('만료') ? navigate(routes.login) : setRetryKey(key => key + 1)}>{error.includes('만료') ? '로그인하기' : '다시 시도하기'}</Button></>}
      {loading ? <p className="history-empty">이사 기록을 불러오고 있어요.</p>
        : !hasSession() ? <div className="history-empty history-guest"><p>로그인하면 나의 이사 기록을 볼 수 있어요.</p><Button onClick={() => navigate(routes.login)}>로그인하기</Button></div>
        : visible.length ? <div className="history-card-list">{visible.map(record => <HistoryCard key={record.move_id} record={record} onOpen={() => void openRecord(record)} />)}</div>
        : error ? null
        : <p className="history-empty">{filter === 'all' ? '이사 기록이 없어요.' : `${filter === 'done' ? '완료된' : '진행 중인'} 이사 기록이 없어요.`}</p>}
    </div>
  </main>;
}

export function HistoryDetailPage() {
  const moveId = new URLSearchParams(location.search).get('moveId');
  const [stage, setStage] = useState<Stage>('preparing');
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState(!moveId ? '이사 기록을 찾을 수 없어요.' : '');
  const [notice, setNotice] = useState<'review' | 'pdf' | null>(null);
  const [retryKey, setRetryKey] = useState(0);
  useEffect(() => {
    if (!moveId) return;
    let active = true;
    setLoaded(false);
    setError('');
    void Promise.all([getMove(moveId), listAllMoves()]).then(([, records]) => {
      if (!active) return;
      const record = records.find(item => item.move_id === moveId);
      if (!record) { setError('이사 기록을 찾을 수 없어요.'); return; }
      setStage(record.stage);
      sessionStorage.setItem('jimssak-member-move', JSON.stringify({ move_id: moveId, guest_token: '' }));
      setLoaded(true);
    }).catch(cause => { if (active) setError(cause instanceof ApiError && cause.status === 401 ? '로그인이 만료됐어요. 다시 로그인해주세요.' : '이사 기록을 불러오지 못했어요.'); });
    return () => { active = false; };
  }, [moveId, retryKey]);

  async function download() {
    navigate(routes.result);
  }

  return <main className="app-page history-detail-page">
    <BrandHeader />
    <div className="history-detail-scroll">
      {error && <><p role="alert">{error}</p><Button variant="secondary" onClick={() => !moveId ? navigate(routes.history) : error.includes('만료') ? navigate(routes.login) : setRetryKey(key => key + 1)}>{!moveId ? '기록으로 돌아가기' : error.includes('만료') ? '로그인하기' : '다시 시도하기'}</Button></>}
      {!loaded && !error && <p>이사 기록을 불러오고 있어요.</p>}
      {loaded && <>
        <div className="review-avatar"><img src={chatAvatar} alt="" /><strong>짐싹</strong></div>
        <h1>{stage === 'after_move' ? '이사는 잘 마치셨나요?' : stage === 'comparing' ? '물품을 함께 확인해요' : '이사 기록을 확인해요'}</h1>
        <p>{stage === 'after_move' ? <>이사 후 방을 촬영하면 이사 전에 찍어둔 영상과<br />비교해드려요. 상태가 달라 보이거나<br />영상에서 찾지 못한 물품을 알려드릴게요.<br />결과를 보고 직접 확인할 수 있어요.</> : stage === 'comparing' ? '이사 전후 물품의 상태를 검토해 주세요.' : '확인한 물품과 이사 기록을 다시 살펴볼 수 있어요.'}</p>
        <Button onClick={() => { if (stage === 'after_move') { window.history.pushState({}, '', `${routes.capture}?mode=after`); window.dispatchEvent(new PopStateEvent('popstate')); } else if (stage === 'preparing') navigate(routes.chat); else navigate(routes.afterReview); }}>{stage === 'after_move' ? '이사 후 물품 확인하기' : copy[stage].button}</Button>
      </>}
    </div>
    <div className="review-chat-bottom history-detail-bottom">
      <div className="review-footer-tabs"><button type="button" onClick={() => navigate(routes.inventory)}><img src={listIcon} alt="" />이사전 확정 목록 열기</button><button type="button" className="review-pdf-ready" onClick={() => void download()}><img src={downloadIcon} alt="" />이삿짐 명세서 PDF 받기</button></div>
      <div className="review-chat-input"><span>수정할 내용을 알려주세요</span><button type="button" aria-label="채팅으로 돌아가기" onClick={() => navigate(routes.chat)}><img src={sendIcon} alt="" /></button></div>
      <p>문서는 목록과 이사 조건을 확인한 뒤 만들 수 있어요</p>
    </div>
    {notice && <div className="review-dialog-backdrop" onClick={() => setNotice(null)}><div className="review-dialog" role="dialog" aria-modal="true" aria-label="안내" onClick={event => event.stopPropagation()}><h2>{notice === 'review' ? '물품 검토는 준비 중이에요' : '이삿짐 명세서를 받았어요'}</h2><p>{notice === 'review' ? '이사 후 영상 비교 결과를 받으면 이 기록에서 물품 검토를 이어갈 수 있어요.' : '명세서를 업체에 보내 견적을 문의해보세요.'}</p><Button onClick={() => setNotice(null)}>확인</Button></div></div>}
  </main>;
}
