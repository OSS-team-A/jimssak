import { useEffect, useState } from 'react';
import { navigate, routes } from '../app/routes';
import { Button } from '../components/common/Button';
import { BrandHeader } from '../components/common/BrandHeader';
import { hasSession } from '../api/auth';
import { getMove, listMoves } from '../api/moves';
import { ApiError } from '../api/client';
import type { MoveSummary } from '../types/move';
import { downloadReviewPdf } from '../features/reviewPdf';
import type { ReviewDraft } from '../features/review';
import closeIcon from '../assets/history-close.svg';
import chatAvatar from '../assets/chat-avatar.svg';
import listIcon from '../assets/chat-list.svg';
import downloadIcon from '../assets/chat-download-active.svg';
import sendIcon from '../assets/chat-send.svg';

type Filter = 'all' | 'progress' | 'done';
type Stage = MoveSummary['stage'];
const stages: Stage[] = ['preparing', 'after_move', 'comparing', 'done'];
const isPreview = () => import.meta.env.DEV && new URLSearchParams(location.search).get('preview') === '1';
const previewStage = (): Stage => stages.find(stage => stage === new URLSearchParams(location.search).get('state')) ?? 'preparing';
const copy: Record<Stage, { status: string; badge: string; description: string; button: string }> = {
  preparing: { status: '이사 준비중', badge: '진행중', description: '명세서를 업체에 보내 견적을 문의해보세요.', button: '이사 기록 보기' },
  after_move: { status: '이사 후 확인 전', badge: '진행중', description: '이사를 마쳤다면 이사 후 물품을 촬영해보세요.', button: '이사 후 확인 시작하기' },
  comparing: { status: '물품 검토 중', badge: '진행중', description: '상태 변화 의심 1개 · 영상 미확인 1개', button: '물품 검토 이어가기' },
  done: { status: '검토 완료', badge: '완료', description: '직접 확인 1개 · 아직 확인 필요 1개', button: '검토 결과 보기' },
};

function shortPlace(address: string | null) {
  if (!address) return '주소 확인 전';
  const [province, district] = address.split(' ');
  return `${province.replace(/특별시|광역시|특별자치시|특별자치도|도$/, '')} ${district || ''}`.trim();
}
function moveDate(date: string | null) {
  if (!date) return '이사 날짜 확인 전';
  const [year, month, day] = date.split('-').map(Number);
  return `${year}년 ${month}월 ${day}일 이사`;
}

function HistoryCard({ record, preview, onOpen }: { record: MoveSummary; preview: boolean; onOpen: () => void }) {
  const state = copy[record.stage];
  return <article className="history-card">
    <div className="history-status"><span className={record.stage === 'done' ? 'history-status-done' : ''}>{state.badge}</span><strong>{state.status}</strong></div>
    <h2>{moveDate(record.move_date)}</h2><hr />
    <strong className="history-route">{shortPlace(record.origin_address)} → {shortPlace(record.destination_address)}</strong>
    <p className="history-items">{preview ? '이삿짐 5개 · 원룸' : '이삿짐 목록은 기록에서 확인할 수 있어요.'}</p>
    <p className="history-description">{record.stage === 'preparing' || preview ? state.description : '이사 후 물품 상태를 확인해 주세요.'}</p>
    <Button onClick={onOpen}>{state.button}</Button>
  </article>;
}

export function HistoryPage() {
  const preview = isPreview();
  const [filter, setFilter] = useState<Filter>('all');
  const [records, setRecords] = useState<MoveSummary[]>([]);
  const [loading, setLoading] = useState(!preview && hasSession());
  const [error, setError] = useState('');
  const [opening, setOpening] = useState(false);

  useEffect(() => {
    if (preview) {
      let active = true;
      void import('../features/historyPreview').then(({ historyPreviewMove }) => { if (active) setRecords([historyPreviewMove(previewStage())]); });
      return () => { active = false; };
    }
    if (!hasSession()) return;
    let active = true;
    async function load() {
      try {
        const all: MoveSummary[] = [];
        while (true) {
          const page = await listMoves(100, all.length);
          all.push(...page.moves);
          if (all.length >= page.total || page.moves.length === 0) break;
        }
        if (active) setRecords(all);
      } catch (cause) { if (active) setError(cause instanceof ApiError && cause.status === 401 ? '로그인이 만료됐어요. 다시 로그인해주세요.' : '이사 기록을 불러오지 못했어요. 잠시 후 다시 시도해주세요.'); }
      finally { if (active) setLoading(false); }
    }
    void load();
    return () => { active = false; };
  }, [preview]);

  async function openRecord(record: MoveSummary) {
    if (opening) return;
    if (preview) {
      if (record.stage === 'preparing') window.history.pushState({}, '', `${routes.chat}?preview=1`);
      else window.history.pushState({}, '', `${routes.historyDetail}?preview=1&state=${record.stage}&moveId=${record.move_id}`);
      window.dispatchEvent(new PopStateEvent('popstate'));
      return;
    }
    setOpening(true);
    try {
      const snapshot = await getMove(record.move_id);
      sessionStorage.setItem('jimssak-member-move', JSON.stringify({ move_id: record.move_id, guest_token: '' }));
      if (record.stage !== 'preparing') window.history.pushState({}, '', `${routes.historyDetail}?moveId=${record.move_id}&state=${record.stage}`);
      else if (snapshot.data.room_video.status === 'empty') navigate(routes.capture);
      else if (snapshot.missing.some(step => step !== 'room_video')) navigate(routes.moveInfo);
      else if (snapshot.data.analysis.status === 'completed') navigate(routes.chat);
      else navigate(routes.analysis);
      if (record.stage !== 'preparing') window.dispatchEvent(new PopStateEvent('popstate'));
    } catch { setError('선택한 이사 기록을 열지 못했어요. 잠시 후 다시 시도해주세요.'); }
    finally { setOpening(false); }
  }

  const visible = records.filter(record => filter === 'all' || (filter === 'done' ? record.stage === 'done' : record.stage !== 'done'));
  return <main className="app-page history-page">
    <header className="sub-header"><strong>나의 이사 기록</strong><button type="button" aria-label="닫기" onClick={() => navigate(routes.home)}><img src={closeIcon} alt="" /></button></header>
    <div className="history-content"><span className="eyebrow">이사 기록</span>
      <h1>차곡차곡 쌓인 나의 이사</h1>
      <p>이사 준비부터 이사 후 물품 확인까지,<br />진행 중인 이사를 이어가고 지난 결과를 확인해요.</p>
      <div className="history-filters" aria-label="이사 기록 필터">
        {([['all', '전체'], ['progress', '진행중'], ['done', '완료']] as const).map(([key, label]) =>
          <button type="button" key={key} aria-pressed={filter === key} className={filter === key ? 'active' : ''} onClick={() => setFilter(key)}>{label}</button>)}
      </div>
      {error && <p className="history-error" role="alert">{error}</p>}
      {loading ? <p className="history-empty">이사 기록을 불러오고 있어요.</p>
        : !preview && !hasSession() ? <div className="history-empty"><p>로그인하면 나의 이사 기록을 볼 수 있어요.</p><Button onClick={() => navigate(routes.login)}>로그인하기</Button></div>
        : visible.length ? <div className="history-card-list">{visible.map(record => <HistoryCard key={record.move_id} record={record} preview={preview} onOpen={() => void openRecord(record)} />)}</div>
        : error ? null
        : <p className="history-empty">{filter === 'all' ? '이사 기록이 없어요.' : `${filter === 'done' ? '완료된' : '진행 중인'} 이사 기록이 없어요.`}</p>}
    </div>
  </main>;
}

export function HistoryDetailPage() {
  const preview = isPreview();
  const stage = previewStage() === 'preparing' ? 'after_move' : previewStage();
  const moveId = new URLSearchParams(location.search).get('moveId');
  const [loaded, setLoaded] = useState(preview);
  const [error, setError] = useState(!preview && !moveId ? '이사 기록을 찾을 수 없어요.' : '');
  const [notice, setNotice] = useState<'capture' | 'review' | 'pdf' | null>(null);
  useEffect(() => {
    if (preview || !moveId) return;
    let active = true;
    void getMove(moveId).then(() => { if (active) setLoaded(true); }).catch(() => { if (active) setError('이사 기록을 불러오지 못했어요.'); });
    return () => { active = false; };
  }, [moveId, preview]);

  async function download() {
    if (!preview) { navigate(routes.result); return; }
    const { reviewPreview } = await import('../features/reviewPreview');
    const draft: ReviewDraft = { analysisId: reviewPreview.analysis_id!, step: 5, quantities: {}, added: [], notes: [], answers: [], selectedMoveType: 'full' };
    try { await downloadReviewPdf({ moveId: moveId ?? '__preview__', guestToken: '', preview: true, draft, items: reviewPreview.items, added: [] }); setNotice('pdf'); }
    catch { setError('PDF를 만들지 못했어요. 다시 시도해주세요.'); }
  }

  return <main className="app-page history-detail-page">
    <BrandHeader />
    <div className="history-detail-scroll">
      {error && <p role="alert">{error}</p>}
      {!loaded && !error && <p>이사 기록을 불러오고 있어요.</p>}
      {loaded && <>
        <div className="review-avatar"><img src={chatAvatar} alt="" /><strong>짐싹</strong></div>
        <h1>{stage === 'after_move' ? '이사는 잘 마치셨나요?' : stage === 'comparing' ? '물품을 함께 확인해요' : '이사 기록을 확인해요'}</h1>
        <p>{stage === 'after_move' ? <>이사 후 방을 촬영하면 이사 전에 찍어둔 영상과<br />비교해드려요. 상태가 달라 보이거나<br />영상에서 찾지 못한 물품을 알려드릴게요.<br />결과를 보고 직접 확인할 수 있어요.</> : stage === 'comparing' ? '이사 전후 물품의 상태를 검토해 주세요.' : '확인한 물품과 이사 기록을 다시 살펴볼 수 있어요.'}</p>
        <Button onClick={() => setNotice(stage === 'after_move' ? 'capture' : 'review')}>{stage === 'after_move' ? '이사 후 물품 확인하기' : copy[stage].button}</Button>
      </>}
    </div>
    <div className="review-chat-bottom history-detail-bottom">
      <div className="review-footer-tabs"><button type="button" onClick={() => { if (preview) { window.history.pushState({}, '', `${routes.inventory}?preview=1`); window.dispatchEvent(new PopStateEvent('popstate')); } else navigate(routes.inventory); }}><img src={listIcon} alt="" />이사전 확정 목록 열기</button><button type="button" className="review-pdf-ready" onClick={() => void download()}><img src={downloadIcon} alt="" />이삿짐 명세서 PDF 받기</button></div>
      <div className="review-chat-input"><span>수정할 내용을 알려주세요</span><button type="button" aria-label="채팅으로 돌아가기" onClick={() => { if (preview) { window.history.pushState({}, '', `${routes.chat}?preview=1`); window.dispatchEvent(new PopStateEvent('popstate')); } else navigate(routes.chat); }}><img src={sendIcon} alt="" /></button></div>
      <p>문서는 목록과 이사 조건을 확인한 뒤 만들 수 있어요</p>
    </div>
    {notice && <div className="review-dialog-backdrop" onClick={() => setNotice(null)}><div className="review-dialog" role="dialog" aria-modal="true" aria-label="안내" onClick={event => event.stopPropagation()}><h2>{notice === 'capture' ? '이사 후 촬영은 준비 중이에요' : notice === 'review' ? '물품 검토는 준비 중이에요' : '이삿짐 명세서를 받았어요'}</h2><p>{notice === 'capture' ? '이사 전 영상은 보관됩니다. 이사 후 영상 비교 기능이 연결되면 이 기록에서 촬영을 시작할 수 있어요.' : notice === 'review' ? '이사 후 영상 비교 결과를 받으면 이 기록에서 물품 검토를 이어갈 수 있어요.' : '명세서를 업체에 보내 견적을 문의해보세요.'}</p><Button onClick={() => setNotice(null)}>확인</Button></div></div>}
  </main>;
}
