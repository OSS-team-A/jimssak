import { useEffect, useRef, useState, type FormEvent } from 'react';
import { BrandHeader } from '../components/common/BrandHeader';
import { Button } from '../components/common/Button';
import { navigate, routes } from '../app/routes';
import { dimensions, type ReviewDraft } from '../features/review';
import { downloadReviewPdf } from '../features/reviewPdf';
import type { AnalysisResult } from '../types/move';
import { ChatFooter, PdfNotice } from './ReviewPages';
import chatAvatar from '../assets/chat-avatar.svg';
import sendIcon from '../assets/chat-send.svg';
import chevronIcon from '../assets/chat-chevron.svg';
import rowChevronIcon from '../assets/chat-row-chevron.svg';
import closeIcon from '../assets/inventory-close.svg';
import downIcon from '../assets/inventory-chevron-down.svg';
import upIcon from '../assets/inventory-chevron-up.svg';
import minusIcon from '../assets/inventory-minus.svg';
import plusIcon from '../assets/inventory-plus.svg';
import warningIcon from '../assets/after-review-warning.svg';
import confirmIcon from '../assets/after-review-confirm.svg';
import questionIcon from '../assets/after-review-question-bg.svg';
import questionMark from '../assets/after-review-question-mark.svg';

type Confirmation = 'confirmed' | 'missing' | null;
type AfterReviewState = { step: 0 | 1 | 2; chairConfirmed: boolean; wardrobe: Confirmation; notes: string[]; quantities: Record<string, number> };
const initial: AfterReviewState = { step: 0, chairConfirmed: false, wardrobe: null, notes: [], quantities: { 'preview-bed': 1, 'preview-desk': 1, 'preview-chair': 1, 'preview-wardrobe': 1, 'preview-monitor': 1 } };
const preview = () => import.meta.env.DEV && new URLSearchParams(location.search).get('preview') === '1';
const storageKey = 'jimssak-after-review-preview';
const load = (): AfterReviewState => {
  if (!preview()) return initial;
  try {
    const params = new URLSearchParams(location.search);
    if (params.get('reset') === '1') sessionStorage.removeItem(storageKey);
    const saved = sessionStorage.getItem(storageKey);
    if (saved) return { ...initial, ...JSON.parse(saved) as AfterReviewState };
    const step = params.get('state') === 'saved' ? 2 : params.get('state') === 'followup' ? 1 : 0;
    return { ...initial, step };
  } catch { return initial; }
};
function go(view: 'chat' | 'list', status?: 'normal' | 'needs' | 'saved', fromList = false) {
  const query = new URLSearchParams();
  if (preview()) query.set('preview', '1');
  if (view === 'list') query.set('view', 'list');
  if (status) query.set('state', status);
  if (fromList) query.set('from', 'list');
  window.history.pushState({}, '', `${routes.afterReview}?${query}`);
  window.dispatchEvent(new PopStateEvent('popstate'));
}
function Avatar() { return <div className="review-avatar"><img src={chatAvatar} alt="" /><strong>짐싹</strong></div>; }

export function AfterMoveReviewPage() {
  const [query, setQuery] = useState(location.search);
  const params = new URLSearchParams(query);
  const isPreview = preview();
  const [state, setState] = useState<AfterReviewState>(load);
  const [expanded, setExpanded] = useState<string | null>(params.get('view') === 'list' ? 'preview-chair' : null);
  const [message, setMessage] = useState('');
  const [toast, setToast] = useState(() => params.get('from') === 'list' ? '목록을 확인했어요' : params.get('view') === 'list' && !state.chairConfirmed ? '상태 변화가 의심돼요' : '');
  const [pdfNotice, setPdfNotice] = useState<string | null>(null);
  const [sample, setSample] = useState<AnalysisResult | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const listView = params.get('view') === 'list';
  const normalPreview = params.get('state') === 'normal';
  const shown = (sample?.items ?? []).map(item => ({ ...item, quantity: state.quantities[item.item_id] ?? item.quantity }));
  const roomPhoto = sample?.items.find(item => item.snapshots[0]?.image_url)?.snapshots[0]?.image_url ?? '';
  const total = shown.reduce((sum, item) => sum + item.quantity, 0);
  const concerns = (state.chairConfirmed ? 0 : 1) + (state.wardrobe === 'confirmed' ? 0 : 1);

  useEffect(() => { const sync = () => setQuery(location.search); window.addEventListener('popstate', sync); return () => window.removeEventListener('popstate', sync); }, []);
  useEffect(() => { if (isPreview) sessionStorage.setItem(storageKey, JSON.stringify(state)); }, [isPreview, state]);
  useEffect(() => { if (!isPreview) return; let active = true; void import('../features/reviewPreview').then(({ reviewPreview }) => { if (active) setSample(reviewPreview); }); return () => { active = false; }; }, [isPreview]);
  useEffect(() => { if (!toast) return; const timer = window.setTimeout(() => setToast(''), 2600); return () => window.clearTimeout(timer); }, [toast]);
  useEffect(() => { if (!listView && scrollRef.current && state.step > 0) scrollRef.current.scrollTop = scrollRef.current.scrollHeight; }, [listView, state.step]);

  function openList() { go('list', concerns === 0 ? 'normal' : 'needs'); }
  function closeList() {
    if (state.step === 0) sessionStorage.setItem(storageKey, JSON.stringify({ ...state, step: 1 }));
    go('chat', state.step === 2 ? 'saved' : undefined, true);
    if (state.step === 0) setState(current => ({ ...current, step: 1 }));
    setToast('목록을 확인했어요');
  }
  function selectConfirmation(value: Confirmation) {
    setState(current => ({ ...current, wardrobe: value }));
    if (value === 'missing') setToast('영상에서 확인되지 않았어요');
    else setToast('');
  }
  async function onPdf() {
    if (!isPreview) return;
    const draft: ReviewDraft = { analysisId: sample?.analysis_id ?? 'figma-sc16-preview', step: 5, quantities: state.quantities, added: [], notes: [], answers: [], selectedMoveType: 'full' };
    try {
      await downloadReviewPdf({ moveId: '__preview__', guestToken: '', preview: true, draft, items: shown, added: [] });
      setPdfNotice('');
    } catch (cause) { setPdfNotice(cause instanceof Error ? cause.message : '잠시 후 다시 시도해주세요.'); }
  }
  function send(event: FormEvent) {
    event.preventDefault();
    const value = message.trim();
    if (!value) return;
    setState(current => ({ ...current, step: Math.max(1, current.step) as 1 | 2, notes: [...current.notes, value] }));
    setMessage('');
  }

  if (!isPreview) return <main className="app-page review-chat-page"><BrandHeader /><div className="review-gate"><p>이사 후 영상 비교 결과 API가 연결되면 물품 검토 결과를 확인할 수 있어요.</p><Button variant="secondary" onClick={() => navigate(routes.history)}>나의 이사 기록으로 돌아가기</Button></div></main>;

  if (listView) return <main className="app-page review-inventory-page after-review-list">
    <header className="sub-header"><strong>이삿짐 목록</strong><button type="button" onClick={closeList} aria-label="목록 닫기"><img src={closeIcon} alt="" /></button></header>
    <div className="review-inventory-body">
      <div className="review-inventory-heading"><h1>이삿짐 목록</h1><span className={normalPreview || concerns === 0 ? 'after-badge-normal' : 'after-badge-concern'}>{normalPreview || concerns === 0 ? '이상 없음' : '확인 필요'}</span></div>
      <p>이사 후 촬영한 영상 분석을 바탕으로 작성한 목록이에요.<br />파손되거나 분실된 물품을 확인해요.</p>
      <div className="review-inventory-stats"><span>품목 수 <strong>{shown.filter(item => item.quantity > 0).length}종</strong></span><span>총 수량 <strong>{total}개</strong></span></div>
      <div className="review-inventory-table"><div className="review-inventory-table-head"><span>번호</span><span>품목</span><span>수량</span><span>예상 크기 (cm)</span></div>
        {shown.map((item, index) => {
          const concern = !normalPreview && ((item.item_id === 'preview-chair' && !state.chairConfirmed) || (item.item_id === 'preview-wardrobe' && state.wardrobe !== 'confirmed'));
          const opened = expanded === item.item_id;
          return <div className={`review-inventory-item${concern && opened ? ' after-item-concern' : ''}`} key={item.item_id}>
            <div className="review-inventory-row"><small>{String(index + 1).padStart(2, '0')}</small><strong>{item.label_ko}</strong><div className="review-quantity"><button type="button" aria-label={`${item.label_ko} 수량 줄이기`} onClick={() => setState(current => ({ ...current, quantities: { ...current.quantities, [item.item_id]: Math.max(0, item.quantity - 1) } }))}><img src={minusIcon} alt="" /></button><b>{item.quantity}</b><button type="button" aria-label={`${item.label_ko} 수량 늘리기`} onClick={() => setState(current => ({ ...current, quantities: { ...current.quantities, [item.item_id]: item.quantity + 1 } }))}><img src={plusIcon} alt="" /></button></div><span>{dimensions(item)}</span><button className="review-expand" type="button" aria-label={`${item.label_ko} 상세 ${opened ? '닫기' : '보기'}`} onClick={() => { setExpanded(opened ? null : item.item_id); if (concern) setToast(item.item_id === 'preview-chair' ? '상태 변화가 의심돼요' : '영상에서 확인되지 않았어요'); }}><img src={opened ? upIcon : downIcon} alt="" /></button></div>
            {opened && <div className="after-review-detail">
              {item.item_id === 'preview-wardrobe' && concern && <p>촬영 범위 밖이거나 다른 물품에 가려졌을 수 있어요.</p>}
              {(item.item_id === 'preview-chair' || item.item_id === 'preview-wardrobe' || item.snapshots[0]?.image_url) && <div className="after-review-photo-stack">
                {item.item_id === 'preview-chair' && concern && <><span>before</span><div className="after-review-photo"><img src={roomPhoto} alt="이사 전 의자" /></div><span>after</span></>}
                <div className={`after-review-photo${item.item_id === 'preview-wardrobe' ? ' after-wardrobe-photo' : ''}`}><img src={item.snapshots[0]?.image_url || roomPhoto} alt={`${item.label_ko} 이사 후 사진`} /></div>
              </div>}
              {item.item_id === 'preview-wardrobe' && concern ? <div className="after-review-confirm"><button type="button" className={state.wardrobe === 'confirmed' ? 'selected' : ''} onClick={() => selectConfirmation('confirmed')}><img src={confirmIcon} alt="" />직접 확인했어요</button><button type="button" className={state.wardrobe === 'missing' ? 'selected' : ''} onClick={() => selectConfirmation('missing')}><span className="after-question-icon"><img src={questionIcon} alt="" /><img src={questionMark} alt="" /></span>아직 못 찾았어요</button></div> : item.item_id === 'preview-chair' && concern ? null : <dl><dt>크기</dt><dd>{dimensions(item)} cm</dd><dt>예상 무게</dt><dd>확인 필요</dd><dt>분리 여부</dt><dd>{item.disassembly_required ? '분리 필요' : '다리 분리 가능 여부 확인'}</dd><dt>위치</dt><dd>창가 왼쪽 · 모니터 아래</dd></dl>}
            </div>}
          </div>;
        })}
      </div>
      <div className="review-inventory-total"><strong>합계</strong><strong>{total}개</strong></div>
      <div className="review-inventory-help"><strong>확인해 주세요</strong><p>• 크기는 가로 × 깊이 × 높이 순서의 추정값이에요.<br />• 무게와 분리 여부는 실제 물품 확인이 필요해요.<br />• 누락된 물품이나 두고 갈 물품은 채팅으로 알려주세요.</p><small>확정된 목록 · 검토 후 명세서로 저장할 수 있어요.</small></div>
    </div>{toast && <div className="after-review-toast" role="status"><img src={warningIcon} alt="" />{toast}</div>}
  </main>;

  return <main className="app-page review-chat-page after-review-chat"><BrandHeader />
    <div className="review-chat-scroll" ref={scrollRef}>
      <section className="review-chat-turn"><Avatar /><h1>확인이 필요한 물품이 있어요</h1><p className="review-chat-intro">이사 전후 영상을 비교했어요.<br />상태가 달라 보이는 물품 1개와<br />이사 후 영상에서 찾지 못한 물품 1개가 있어요.<br />목록 탭을 눌러 직접 확인해주세요.</p>
        <button className="review-summary" type="button" onClick={openList}><span className="review-summary-title">이삿짐 목록<img src={chevronIcon} alt="" /></span>{shown.map(item => <span className="review-summary-row" key={item.item_id}><span>{item.label_ko}</span><strong>{item.quantity}개</strong><img src={rowChevronIcon} alt="" /></span>)}<span className="review-summary-count after-concern-count">총 {total}개 · 확인이 필요한 물품 {concerns}개</span></button>
        {state.step === 0 && <Button onClick={() => { setState(current => ({ ...current, step: 1 })); setToast('목록을 확인했어요'); }}>확인했어요</Button>}
      </section>
      {state.step > 0 && <><div className="review-user-bubble">목록을 확인했어요</div><section className="review-chat-turn"><Avatar /><h1>물품은 직접 확인해보셨나요?</h1><p className="review-chat-intro">상태가 달라 보이거나 영상에서 찾지 못한 물품을<br />목록에서 확인해주세요. 직접 확인한 결과가 있다면<br />채팅으로 알려주셔도 돼요.</p><div className="review-example"><strong>예를 들어 이렇게 알려주세요</strong><p>“의자는 괜찮아요. 옷장은 아직 못 찾았어요.”</p></div><Button onClick={() => { const next = { ...state, step: 2 as const }; sessionStorage.setItem(storageKey, JSON.stringify(next)); setState(next); go('chat', 'saved'); }}>검토 결과 저장하기</Button><p className="review-local-note">확인하지 못한 물품은 확인 필요로 저장돼요</p></section></>}
      {state.notes.map((note, index) => <div className="review-user-bubble" key={`${note}-${index}`}>{note}</div>)}
      {state.step === 2 && <section className="review-chat-turn"><Avatar /><h1>이사 후 물품 확인을 마쳤어요</h1><p className="review-chat-intro">확인한 내용을 이사 기록에 저장했어요.<br />아직 찾지 못한 물품은 ‘확인필요’로 남겨뒀어요.<br />나의 이사 기록에서 결과를 언제든 다시 볼 수 있어요.</p><Button variant="secondary" onClick={() => navigate(routes.home)}>홈으로 돌아가기</Button></section>}
    </div>
    <div className="review-chat-bottom"><ChatFooter canDownload onPdf={onPdf} result onInventory={openList} /><form className="review-chat-input" onSubmit={send}><input value={message} onChange={event => setMessage(event.target.value)} placeholder="확인한 내용을 알려주세요" aria-label="확인한 내용" /><button type="submit" aria-label="내용 보내기" disabled={!message.trim()}><img src={sendIcon} alt="" /></button></form><p>문서는 목록과 이사 조건을 확인한 뒤 만들 수 있어요</p></div>
    {toast && <div className="after-review-toast" role="status"><img src={warningIcon} alt="" />{toast}</div>}
    {pdfNotice !== null && <PdfNotice guest={false} error={pdfNotice} onClose={() => setPdfNotice(null)} />}
  </main>;
}
