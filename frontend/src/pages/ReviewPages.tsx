import { useEffect, useRef, useState, type FormEvent } from 'react';
import { BrandHeader } from '../components/common/BrandHeader';
import { Button } from '../components/common/Button';
import { navigate, routes } from '../app/routes';
import { dimensions, isReviewPreview, navigateReview, useReview, type ReviewDraft } from '../features/review';
import { downloadReviewPdf } from '../features/reviewPdf';
import chatAvatar from '../assets/chat-avatar.svg';
import listIcon from '../assets/chat-list.svg';
import sendIcon from '../assets/chat-send.svg';
import downloadIcon from '../assets/chat-download.svg';
import activeDownloadIcon from '../assets/chat-download-active.svg';
import chevronIcon from '../assets/chat-chevron.svg';
import rowChevronIcon from '../assets/chat-row-chevron.svg';
import inventoryCloseIcon from '../assets/inventory-close.svg';
import inventoryChevronDown from '../assets/inventory-chevron-down.svg';
import inventoryChevronUp from '../assets/inventory-chevron-up.svg';
import inventoryMinusIcon from '../assets/inventory-minus.svg';
import inventoryPlusIcon from '../assets/inventory-plus.svg';

const questions = [
  { title: '빠진 물품이 있나요?', description: '영상에 나오지 않은 물품을 목록에서 추가해주세요.', example: '“모니터 27인치 추가해줘.”', skip: '빠진 물품 없어요' },
  { title: '목록에서 제외할 물품이 있나요?', description: '이사하지 않는 물품은 목록에서 수량을 0개로 바꿔주세요.', example: '“세탁기는 이사하지 않아요.”', skip: '제외할 물품 없어요' },
  { title: '이사할 때 알아두면 좋을 점이 있나요?', description: '무거운 물품이나 트럭 진입 제한, 짐을 빼는 날과 받는 날이 다른 경우를 알려주세요.', example: '“침대가 돌침대예요. 짐을 빼고 이틀 뒤에 받아야 해요.”', skip: '특이사항 없어요' },
];
const objectParticle = (name: string) => {
  const last = name.charCodeAt(name.length - 1);
  return last >= 0xac00 && last <= 0xd7a3 && (last - 0xac00) % 28 === 0 ? '를' : '을';
};

function ReviewGate({ loading, error, hasMove, completed }: { loading: boolean; error: string; hasMove: boolean; completed: boolean }) {
  return <div className="review-gate">
    <p>{loading ? '분석 결과를 불러오고 있어요.' : error || (!hasMove ? '저장된 이사 정보가 없어요.' : !completed ? '영상 분석이 완료되면 이삿짐 목록을 확인할 수 있어요.' : '')}</p>
    {!loading && <Button variant="secondary" onClick={() => navigate(hasMove ? routes.analysis : routes.capture)}>{hasMove ? '분석 화면으로 돌아가기' : '촬영 준비로 돌아가기'}</Button>}
  </div>;
}

export function ChatFooter({ canDownload, onPdf, result = false, onInventory }: { canDownload: boolean; onPdf: () => void; result?: boolean; onInventory?: () => void }) {
  return <div className="review-chat-footer">
    <div className="review-footer-tabs">
      <button type="button" onClick={onInventory ?? (() => navigateReview(routes.inventory))}><img src={listIcon} alt="" />{result ? '이사전 확정 목록 열기' : '이삿짐 목록 보러가기'}</button>
      <button type="button" className={canDownload ? 'review-pdf-ready' : ''} onClick={onPdf} disabled={!canDownload}><img src={canDownload ? activeDownloadIcon : downloadIcon} alt="" />이삿짐 명세서 PDF 받기</button>
    </div>
  </div>;
}

export function PdfNotice({ guest, error, onClose }: { guest: boolean; error: string; onClose: () => void }) {
  return <div className="review-dialog-backdrop" onClick={onClose}><div className="review-dialog review-pdf-dialog" role="dialog" aria-modal="true" aria-labelledby="review-pdf-title" onClick={event => event.stopPropagation()}>
    <h2 id="review-pdf-title">{error ? '명세서를 만들지 못했어요' : '이삿짐 명세서를 받았어요'}</h2>
    <p>{error || (guest ? <>업체에 보내 견적을 문의해보세요.<br />이사 후에도 확인하려면 로그인해 이사 기록을 저장해 주세요.</> : <>명세서를 업체에 보내 견적을 문의해보세요.<br />이사 후에는 나의 이사 기록에서 물품을 확인할 수 있어요.</>)}</p>
    <button type="button" onClick={onClose}>확인</button>
  </div></div>;
}

function EstimateConversation({ preview, total, selected, onSelect, onPdf }: {
  preview: boolean;
  total: number;
  selected: ReviewDraft['selectedMoveType'];
  onSelect: (type: NonNullable<ReviewDraft['selectedMoveType']>) => void;
  onPdf: () => void;
}) {
  return <>
    <div className="review-avatar"><img src={chatAvatar} alt="" /><strong>짐싹</strong></div>
    <h1>이사 유형별 예상 비용</h1><p>확인한 이삿짐 목록과 이사 조건을 기준으로 비교했어요.</p>
    <div className="review-result-summary"><strong>{preview ? '서울 강남구 → 서울 마포구' : `검토한 이삿짐 ${total}개`}</strong><p>{preview ? '길찾기 약 18km · 출발 3층 · 도착 5층' : '목록과 이사 조건을 확인하고 있어요.'}</p>{preview && <small>화면의 주소와 거리는 피그마 예시 데이터예요.</small>}</div>
    <div className="review-result-card"><strong>이사 유형</strong>
      {([
        { id: 'standard', name: '일반 이사', price: '약 18만~25만원', description: '운반 중심 · 포장과 정리는 직접 해요' },
        { id: 'semi', name: '반포장 이사', price: '약 27만~36만원', description: '큰 짐 포장과 운반 · 잔짐 정리는 직접 해요' },
        { id: 'full', name: '포장 이사', price: '약 39만~52만원', description: '포장부터 운반·정리까지 맡겨요' },
      ] as const).map(option => <button type="button" key={option.id} className={`review-move-option${selected === option.id ? ' selected' : ''}`} aria-pressed={selected === option.id} onClick={() => onSelect(option.id)}>
        <span><strong>{option.name}</strong><b>{preview ? option.price : '비용 계산 전'}</b></span><small>{option.description}</small>
      </button>)}
      <p>{preview ? '화면의 금액은 예시예요. 실제 비용은 업체 제안에 따라 달라져요.' : '예상 비용 API가 연결되면 이사 유형별 금액을 보여드려요. 실제 비용은 이사업체와 확인해주세요.'}</p>
    </div>
    <div className="review-result-help"><strong>같은 짐을 다시 입력하지 않아도 돼요</strong><p>확인한 명세서를 내려받아 여러 업체에 전달하고 실제 금액을 문의하세요.</p></div>
    <Button disabled={!selected} onClick={onPdf}>이삿짐 명세서 PDF 받기</Button>
    <button className="review-text-button" type="button" onClick={() => navigate(routes.history)}>나의 이사 기록에서 이사 완료하기↗︎</button>
  </>;
}

export function ChatPage() {
  const { move, analysis, draft, loading, error, items, added, total, preview, updateDraft } = useReview();
  const [message, setMessage] = useState('');
  const [pdfNotice, setPdfNotice] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const step = Math.min(draft?.step ?? 0, 5);
  const ready = analysis?.status === 'completed' && !!draft;
  const canDownload = ready && step === 5 && !!draft.selectedMoveType;

  async function onPdf() {
    if (!move || !draft || !canDownload) return;
    try {
      await downloadReviewPdf({ moveId: move.move_id, guestToken: move.guest_token, preview, draft, items, added });
      setPdfNotice('');
    } catch (cause) { setPdfNotice(cause instanceof Error ? cause.message : '잠시 후 다시 시도해주세요.'); }
  }

  useEffect(() => {
    if (!ready) return;
    const frame = requestAnimationFrame(() => {
      const scroll = scrollRef.current;
      if (scroll) scroll.scrollTop = scroll.scrollHeight;
    });
    return () => cancelAnimationFrame(frame);
  }, [ready, step, draft?.notes.length]);

  function next() {
    updateDraft(current => ({
      ...current,
      answers: current.step >= 1 && current.step <= 3
        ? [...current.answers, { step: current.step, text: questions[current.step - 1].skip }]
        : current.answers,
      step: Math.min(current.step + 1, 4),
    }));
  }
  function send(event: FormEvent) {
    event.preventDefault();
    const text = message.trim();
    if (!text || !ready || step === 0 || step === 4) return;
    updateDraft(current => ({ ...current, notes: [...current.notes, { step, text }], answers: [...current.answers, { step, text }], step: step === 5 ? 5 : Math.min(current.step + 1, 4) }));
    setMessage('');
  }

  return <main className="app-page review-chat-page">
    <BrandHeader />
    {ready ? <div className="review-chat-scroll" ref={scrollRef}>
      <section className="review-chat-turn">
        <div className="review-avatar"><img src={chatAvatar} alt="" /><strong>짐싹</strong></div>
        <p className="review-chat-intro">영상에서 찾은 이삿짐을 정리했어요.<br />목록 탭을 눌러 맞는지 확인해주세요.</p>
        <button className="review-summary" type="button" onClick={() => navigateReview(routes.inventory)}>
          <span className="review-summary-title">이삿짐 목록<img src={chevronIcon} alt="" /></span>
          {items.filter(item => item.quantity > 0).slice(0, 5).map(item => <span className="review-summary-row" key={item.item_id}><span>{item.label_ko}</span><strong>{item.quantity}개</strong><img src={rowChevronIcon} alt="" /></span>)}
          {added.filter(item => item.quantity > 0).slice(0, Math.max(0, 5 - items.length)).map(item => <span className="review-summary-row" key={item.id}><span>{item.label}</span><strong>{item.quantity}개</strong><img src={rowChevronIcon} alt="" /></span>)}
          {total === 0 && <span className="review-summary-empty">분석에서 찾은 물품이 없어요. 목록에서 직접 추가할 수 있어요.</span>}
          <span className="review-summary-count">총 {total}개 · 누르면 자세히 볼 수 있어요</span>
        </button>
        {step === 0 && <Button onClick={next}>확인했어요</Button>}
      </section>
      {step > 0 && <div className="review-user-bubble">목록을 확인했어요</div>}
      {questions.slice(0, Math.min(step, 3)).map((question, index) => {
        const questionStep = index + 1;
        const completed = step > questionStep;
        const answer = draft.answers.find(entry => entry.step === questionStep)?.text ?? question.skip;
        return <section className="review-chat-turn" key={questionStep}>
          <div className="review-avatar"><img src={chatAvatar} alt="" /><strong>짐싹</strong></div>
          <h1>{question.title}</h1>
          <p className="review-chat-intro">{question.description}</p>
          <div className="review-example"><strong>예를 들어 이렇게 알려주세요</strong><p>{question.example}</p></div>
          {completed ? <div className="review-user-bubble">{answer}</div> : <>
            <Button onClick={next}>{question.skip}</Button>
            <p className="review-local-note">입력한 내용은 이 브라우저에 임시 보관돼요. 목록 수정은 이삿짐 목록에서 해주세요.</p>
          </>}
        </section>;
      })}
      {step === 4 && <section className="review-chat-turn">
        <div className="review-avatar"><img src={chatAvatar} alt="" /><strong>짐싹</strong></div>
        <h1>이 내용으로 견적을 비교할까요?</h1>
        <p className="review-chat-intro">확인한 이삿짐과 이사 조건을 바탕으로 이사 유형별 예상 비용을 비교해요. 실제 견적은 업체와 상담할 때 확인해주세요.</p>
        <Button onClick={() => updateDraft(current => ({ ...current, step: 5 }))}>예상 비용 비교하기</Button>
      </section>}
      {step === 5 && <section className="review-chat-turn review-estimate-turn">
        <EstimateConversation preview={preview} total={total} selected={draft.selectedMoveType} onSelect={type => updateDraft(current => ({ ...current, selectedMoveType: type }))} onPdf={onPdf} />
      </section>}
      {step === 5 && draft.notes.filter(note => note.step === 5).map((note, index) => <div className="review-user-bubble" key={index}>{note.text}</div>)}
    </div> : <ReviewGate loading={loading} error={error} hasMove={!!move} completed={analysis?.status === 'completed'} />}
    <div className="review-chat-bottom">
      <ChatFooter canDownload={canDownload} onPdf={onPdf} result={step === 5} />
      <form className="review-chat-input" onSubmit={send}>
        <input value={message} onChange={event => setMessage(event.target.value)} placeholder="수정할 내용을 알려주세요" aria-label="검토 내용" disabled={!ready || step === 0 || step === 4} />
        <button type="submit" aria-label="내용 보내기" disabled={!message.trim() || !ready || step === 0 || step === 4}><img src={sendIcon} alt="" /></button>
      </form>
      <p>문서는 목록과 이사 조건을 확인한 뒤 만들 수 있어요</p>
    </div>
    {pdfNotice !== null && <PdfNotice guest={!!move?.guest_token} error={pdfNotice} onClose={() => setPdfNotice(null)} />}
  </main>;
}

export function InventoryPage() {
  const { move, analysis, draft, loading, error, items, added, total, preview, updateDraft } = useReview();
  const [expanded, setExpanded] = useState<string | null>(() => isReviewPreview() ? 'preview-chair' : null);
  const [pendingDelete, setPendingDelete] = useState<{ id: string; label: string; added: boolean } | null>(null);
  const ready = analysis?.status === 'completed' && !!draft;
  const visibleItems = items.filter(item => item.quantity > 0);

  function decrease(id: string, label: string, quantity: number, isAdded: boolean) {
    if (quantity === 1) {
      setPendingDelete({ id, label, added: isAdded });
      return;
    }
    if (isAdded) updateDraft(current => ({ ...current, added: current.added.map(entry => entry.id === id ? { ...entry, quantity: entry.quantity - 1 } : entry) }));
    else updateDraft(current => ({ ...current, quantities: { ...current.quantities, [id]: quantity - 1 } }));
  }

  function confirmDelete() {
    if (!pendingDelete) return;
    if (pendingDelete.added) updateDraft(current => ({ ...current, added: current.added.filter(entry => entry.id !== pendingDelete.id) }));
    else updateDraft(current => ({ ...current, quantities: { ...current.quantities, [pendingDelete.id]: 0 } }));
    setExpanded(current => current === pendingDelete.id ? null : current);
    setPendingDelete(null);
  }

  return <main className="app-page review-inventory-page">
    <header className="sub-header"><strong>이삿짐 목록</strong><button type="button" onClick={() => navigateReview(routes.chat)} aria-label="목록 닫기"><img src={inventoryCloseIcon} alt="" /></button></header>
    {ready ? <div className="review-inventory-body">
      <div className="review-inventory-heading"><h1>이삿짐 목록</h1><span>검토중</span></div>
      <p>영상 분석을 바탕으로 작성한 목록이에요.<br />확인한 내용은 명세서에 반영돼요.</p>
      <div className="review-inventory-stats"><span>품목 수 <strong>{visibleItems.length + added.filter(item => item.quantity > 0).length}종</strong></span><span>총 수량 <strong>{total}개</strong></span></div>
      <div className="review-inventory-table"><div className="review-inventory-table-head"><span>번호</span><span>품목</span><span>수량</span><span>예상 크기 (cm)</span></div>
        {visibleItems.map((item, index) => <div className="review-inventory-item" key={item.item_id}>
          <div className="review-inventory-row"><small>{String(index + 1).padStart(2, '0')}</small><strong>{item.label_ko}</strong><div className="review-quantity"><button type="button" onClick={() => decrease(item.item_id, item.label_ko, item.quantity, false)} aria-label={`${item.label_ko} 수량 줄이기`}><img src={inventoryMinusIcon} alt="" /></button><b>{pendingDelete?.id === item.item_id ? 0 : item.quantity}</b><button type="button" onClick={() => updateDraft(current => ({ ...current, quantities: { ...current.quantities, [item.item_id]: item.quantity + 1 } }))} aria-label={`${item.label_ko} 수량 늘리기`}><img src={inventoryPlusIcon} alt="" /></button></div><span>{dimensions(item)}</span><button className="review-expand" type="button" onClick={() => setExpanded(expanded === item.item_id ? null : item.item_id)} aria-label={`${item.label_ko} 상세 ${expanded === item.item_id ? '닫기' : '보기'}`}><img src={expanded === item.item_id ? inventoryChevronUp : inventoryChevronDown} alt="" /></button></div>
          {expanded === item.item_id && <div className="review-inventory-detail">{item.snapshots[0]?.image_url && <div className={`review-inventory-photo${preview ? ' review-inventory-photo--preview' : ''}`}><img src={item.snapshots[0].image_url} alt={`${item.label_ko} 분석 사진`} /></div>}<dl><dt>크기</dt><dd>{dimensions(item)} cm</dd><dt>예상 무게</dt><dd>확인 필요</dd><dt>분리 여부</dt><dd>{item.disassembly_required == null ? '확인 필요' : item.disassembly_required ? '분리 필요' : '분리 불필요'}</dd><dt>검토</dt><dd>{item.needs_review ? item.review_reasons.join(' · ') || '확인 필요' : '확인 완료'}</dd></dl></div>}
        </div>)}
        {added.filter(item => item.quantity > 0).map((item, index) => <div className="review-inventory-row review-inventory-added" key={item.id}><small>{String(visibleItems.length + index + 1).padStart(2, '0')}</small><strong>{item.label}</strong><div className="review-quantity"><button type="button" onClick={() => decrease(item.id, item.label, item.quantity, true)} aria-label={`${item.label} 수량 줄이기`}><img src={inventoryMinusIcon} alt="" /></button><b>{pendingDelete?.id === item.id ? 0 : item.quantity}</b><button type="button" onClick={() => updateDraft(current => ({ ...current, added: current.added.map(entry => entry.id === item.id ? { ...entry, quantity: entry.quantity + 1 } : entry) }))} aria-label={`${item.label} 수량 늘리기`}><img src={inventoryPlusIcon} alt="" /></button></div><span>직접 추가</span></div>)}
      </div>
      <div className="review-inventory-total"><strong>합계</strong><strong>{total}개</strong></div>
      <div className="review-inventory-help"><strong>확인해 주세요</strong><p>• 크기는 가로 × 깊이 × 높이 순서의 추정값이에요.<br />• 무게와 분리 여부는 실제 물품 확인이 필요해요.<br />• 누락된 물품이나 두고 갈 물품은 채팅으로 알려주세요.</p><small>수정한 목록은 이 브라우저에 임시로 보관돼요.</small></div>
    </div> : <ReviewGate loading={loading} error={error} hasMove={!!move} completed={analysis?.status === 'completed'} />}
    {pendingDelete && <div className="review-delete-backdrop" onClick={() => setPendingDelete(null)}><div className="review-delete-dialog" role="dialog" aria-modal="true" aria-labelledby="review-delete-title" onClick={event => event.stopPropagation()}><h2 id="review-delete-title">{pendingDelete.label}{objectParticle(pendingDelete.label)} 삭제할까요?</h2><p>{pendingDelete.label}{objectParticle(pendingDelete.label)} 목록에서 삭제할까요?</p><div><button type="button" onClick={() => setPendingDelete(null)}>아니요</button><button type="button" onClick={confirmDelete}>예</button></div></div></div>}
  </main>;
}

export function ResultPage() {
  const { move, analysis, draft, loading, error, total, preview, items, added, updateDraft } = useReview();
  const [pdfNotice, setPdfNotice] = useState<string | null>(null);
  const ready = analysis?.status === 'completed' && !!draft;
  const canDownload = ready && !!draft.selectedMoveType;
  async function onPdf() {
    if (!move || !draft || !canDownload) return;
    try {
      await downloadReviewPdf({ moveId: move.move_id, guestToken: move.guest_token, preview, draft, items, added });
      setPdfNotice('');
    } catch (cause) { setPdfNotice(cause instanceof Error ? cause.message : '잠시 후 다시 시도해주세요.'); }
  }
  return <main className="app-page review-result-page"><BrandHeader />
    {ready ? <div className="review-result-body">
      <EstimateConversation preview={preview} total={total} selected={draft.selectedMoveType} onSelect={type => updateDraft(current => ({ ...current, selectedMoveType: type }))} onPdf={onPdf} />
    </div> : <ReviewGate loading={loading} error={error} hasMove={!!move} completed={analysis?.status === 'completed'} />}
    <div className="review-chat-bottom review-result-bottom">
      <ChatFooter canDownload={canDownload} onPdf={onPdf} result />
      <div className="review-chat-input"><span>수정할 내용을 알려주세요</span><button type="button" onClick={() => navigateReview(routes.chat)} aria-label="채팅으로 돌아가기"><img src={sendIcon} alt="" /></button></div>
      <p>문서는 목록과 이사 조건을 확인한 뒤 만들 수 있어요</p>
    </div>
    {pdfNotice !== null && <PdfNotice guest={!!move?.guest_token} error={pdfNotice} onClose={() => setPdfNotice(null)} />}
  </main>;
}
