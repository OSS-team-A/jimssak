import { useState, type FormEvent } from 'react';
import { BrandHeader } from '../components/common/BrandHeader';
import { Button } from '../components/common/Button';
import { navigate, routes } from '../app/routes';
import { dimensions, useReview } from '../features/review';
import chatAvatar from '../assets/chat-avatar.svg';
import listIcon from '../assets/chat-list.svg';
import sendIcon from '../assets/chat-send.svg';
import downloadIcon from '../assets/chat-download.svg';
import chevronIcon from '../assets/chat-chevron.svg';
import rowChevronIcon from '../assets/chat-row-chevron.svg';

const questions = [
  { title: '빠진 물품이 있나요?', description: '영상에 나오지 않은 물품을 목록에서 추가해주세요.', example: '“모니터 27인치 추가해줘.”', skip: '빠진 물품 없어요' },
  { title: '목록에서 제외할 물품이 있나요?', description: '이사하지 않는 물품은 목록에서 수량을 0개로 바꿔주세요.', example: '“세탁기는 이사하지 않아요.”', skip: '제외할 물품 없어요' },
  { title: '이사할 때 알아두면 좋을 점이 있나요?', description: '무거운 물품이나 트럭 진입 제한, 짐을 빼는 날과 받는 날이 다른 경우를 알려주세요.', example: '“침대가 돌침대예요. 짐을 빼고 이틀 뒤에 받아야 해요.”', skip: '특이사항 없어요' },
];

function ReviewGate({ loading, error, hasMove, completed }: { loading: boolean; error: string; hasMove: boolean; completed: boolean }) {
  return <div className="review-gate">
    <p>{loading ? '분석 결과를 불러오고 있어요.' : error || (!hasMove ? '저장된 이사 정보가 없어요.' : !completed ? '영상 분석이 완료되면 이삿짐 목록을 확인할 수 있어요.' : '')}</p>
    {!loading && <Button variant="secondary" onClick={() => navigate(hasMove ? routes.analysis : routes.capture)}>{hasMove ? '분석 화면으로 돌아가기' : '촬영 준비로 돌아가기'}</Button>}
  </div>;
}

function ChatFooter({ canEdit }: { canEdit: boolean }) {
  const [pdfOpen, setPdfOpen] = useState(false);
  return <div className="review-chat-footer">
    <div className="review-footer-tabs">
      <button type="button" onClick={() => navigate(routes.inventory)}><img src={listIcon} alt="" />이삿짐 목록 보러가기</button>
      <button type="button" onClick={() => setPdfOpen(true)}><img src={downloadIcon} alt="" />이삿짐 명세서 PDF 받기</button>
    </div>
    {!canEdit && <p>문서는 목록과 이사 조건을 확인한 뒤 만들 수 있어요</p>}
    {pdfOpen && <div className="review-dialog-backdrop" onClick={() => setPdfOpen(false)}><div className="review-dialog" role="dialog" aria-modal="true" aria-label="명세서 안내" onClick={event => event.stopPropagation()}><h2>명세서를 아직 만들 수 없어요</h2><p>목록 저장과 PDF 생성 기능이 연결되면 내려받을 수 있어요.</p><Button onClick={() => setPdfOpen(false)}>확인</Button></div></div>}
  </div>;
}

export function ChatPage() {
  const { move, analysis, draft, loading, error, items, added, total, updateDraft } = useReview();
  const [message, setMessage] = useState('');
  const step = Math.min(draft?.step ?? 0, 4);
  const ready = analysis?.status === 'completed' && !!draft;
  const question = step >= 1 && step <= 3 ? questions[step - 1] : null;

  function next() { updateDraft(current => ({ ...current, step: Math.min(current.step + 1, 4) })); }
  function send(event: FormEvent) {
    event.preventDefault();
    const text = message.trim();
    if (!text || !ready || step === 0 || step === 4) return;
    updateDraft(current => ({ ...current, notes: [...current.notes, { step, text }], step: Math.min(current.step + 1, 4) }));
    setMessage('');
  }

  return <main className="app-page review-chat-page">
    <BrandHeader />
    {ready ? <div className="review-chat-scroll">
      {step > 0 && <div className="review-user-bubble">목록을 확인했어요</div>}
      {draft.notes.length > 0 && <div className="review-user-bubble">{draft.notes[draft.notes.length - 1].text}</div>}
      <div className="review-avatar"><img src={chatAvatar} alt="" /><strong>짐싹</strong></div>
      {step === 0 ? <>
        <p className="review-chat-intro">영상에서 찾은 이삿짐을 정리했어요.<br />목록 탭을 눌러 맞는지 확인해주세요.</p>
        <button className="review-summary" type="button" onClick={() => navigate(routes.inventory)}>
          <span className="review-summary-title">이삿짐 목록<img src={chevronIcon} alt="" /></span>
          {items.filter(item => item.quantity > 0).slice(0, 5).map(item => <span className="review-summary-row" key={item.item_id}><span>{item.label_ko}</span><strong>{item.quantity}개</strong><img src={rowChevronIcon} alt="" /></span>)}
          {added.filter(item => item.quantity > 0).slice(0, Math.max(0, 5 - items.length)).map(item => <span className="review-summary-row" key={item.id}><span>{item.label}</span><strong>{item.quantity}개</strong><img src={rowChevronIcon} alt="" /></span>)}
          {total === 0 && <span className="review-summary-empty">분석에서 찾은 물품이 없어요. 목록에서 직접 추가할 수 있어요.</span>}
          <span className="review-summary-count">총 {total}개 · 누르면 자세히 볼 수 있어요</span>
        </button>
        <Button onClick={next}>확인했어요</Button>
      </> : question ? <>
        <h1>{question.title}</h1>
        <p className="review-chat-intro">{question.description}</p>
        <div className="review-example"><strong>예를 들어 이렇게 알려주세요</strong><p>{question.example}</p></div>
        <Button onClick={next}>{question.skip}</Button>
        <p className="review-local-note">입력한 내용은 이 브라우저에 임시 보관돼요. 목록 수정은 이삿짐 목록에서 해주세요.</p>
      </> : <>
        <h1>이 내용으로 견적을 비교할까요?</h1>
        <p className="review-chat-intro">확인한 이삿짐과 이사 조건을 바탕으로 이사 유형별 예상 비용을 비교해요. 실제 견적은 업체와 상담할 때 확인해주세요.</p>
        <Button onClick={() => navigate(routes.result)}>예상 비용 비교하기</Button>
      </>}
    </div> : <ReviewGate loading={loading} error={error} hasMove={!!move} completed={analysis?.status === 'completed'} />}
    <div className="review-chat-bottom">
      <ChatFooter canEdit={false} />
      <form className="review-chat-input" onSubmit={send}>
        <input value={message} onChange={event => setMessage(event.target.value)} placeholder="수정할 내용을 알려주세요" aria-label="검토 내용" disabled={!ready || step === 0 || step === 4} />
        <button type="submit" aria-label="내용 보내기" disabled={!message.trim() || !ready || step === 0 || step === 4}><img src={sendIcon} alt="" /></button>
      </form>
      <p>문서는 목록과 이사 조건을 확인한 뒤 만들 수 있어요</p>
    </div>
  </main>;
}

export function InventoryPage() {
  const { move, analysis, draft, loading, error, items, added, total, updateDraft } = useReview();
  const [expanded, setExpanded] = useState<string | null>(null);
  const [newLabel, setNewLabel] = useState('');
  const ready = analysis?.status === 'completed' && !!draft;

  function addItem(event: FormEvent) {
    event.preventDefault();
    const label = newLabel.trim();
    if (!label) return;
    updateDraft(current => ({ ...current, added: [...current.added, { id: crypto.randomUUID(), label, quantity: 1 }] }));
    setNewLabel('');
  }

  return <main className="app-page review-inventory-page">
    <header className="sub-header"><strong>이삿짐 목록</strong><button type="button" onClick={() => navigate(routes.chat)} aria-label="목록 닫기">×</button></header>
    {ready ? <div className="review-inventory-body">
      <div className="review-inventory-heading"><h1>이삿짐 목록</h1><span>검토중</span></div>
      <p>영상 분석을 바탕으로 작성한 목록이에요.<br />확인한 내용을 검토해 주세요.</p>
      <div className="review-inventory-stats"><span>품목 수 <strong>{items.filter(item => item.quantity > 0).length + added.filter(item => item.quantity > 0).length}종</strong></span><span>총 수량 <strong>{total}개</strong></span></div>
      <div className="review-inventory-table"><div className="review-inventory-table-head"><span>번호</span><span>품목</span><span>수량</span><span>예상 크기 (cm)</span></div>
        {items.map((item, index) => <div className="review-inventory-item" key={item.item_id}>
          <div className="review-inventory-row"><small>{String(index + 1).padStart(2, '0')}</small><strong>{item.label_ko}</strong><div className="review-quantity"><button type="button" onClick={() => updateDraft(current => ({ ...current, quantities: { ...current.quantities, [item.item_id]: Math.max(0, item.quantity - 1) } }))} aria-label={`${item.label_ko} 수량 줄이기`}>−</button><b>{item.quantity}</b><button type="button" onClick={() => updateDraft(current => ({ ...current, quantities: { ...current.quantities, [item.item_id]: item.quantity + 1 } }))} aria-label={`${item.label_ko} 수량 늘리기`}>+</button></div><span>{dimensions(item)}</span><button className="review-expand" type="button" onClick={() => setExpanded(expanded === item.item_id ? null : item.item_id)} aria-label={`${item.label_ko} 상세 ${expanded === item.item_id ? '닫기' : '보기'}`}>{expanded === item.item_id ? '⌃' : '⌄'}</button></div>
          {expanded === item.item_id && <div className="review-inventory-detail"><dl><dt>크기</dt><dd>{dimensions(item)} cm</dd><dt>분리 여부</dt><dd>{item.disassembly_required == null ? '확인 필요' : item.disassembly_required ? '분리 필요' : '분리 불필요'}</dd><dt>검토</dt><dd>{item.needs_review ? item.review_reasons.join(' · ') || '확인 필요' : '확인 완료'}</dd></dl></div>}
        </div>)}
        {added.map((item, index) => <div className="review-inventory-row review-inventory-added" key={item.id}><small>{String(items.length + index + 1).padStart(2, '0')}</small><strong>{item.label}</strong><div className="review-quantity"><button type="button" onClick={() => updateDraft(current => ({ ...current, added: current.added.map(entry => entry.id === item.id ? { ...entry, quantity: Math.max(0, entry.quantity - 1) } : entry) }))} aria-label={`${item.label} 수량 줄이기`}>−</button><b>{item.quantity}</b><button type="button" onClick={() => updateDraft(current => ({ ...current, added: current.added.map(entry => entry.id === item.id ? { ...entry, quantity: entry.quantity + 1 } : entry) }))} aria-label={`${item.label} 수량 늘리기`}>+</button></div><span>직접 추가</span></div>)}
      </div>
      <form className="review-add-item" onSubmit={addItem}><label htmlFor="review-add-name">빠진 물품 추가</label><div><input id="review-add-name" value={newLabel} onChange={event => setNewLabel(event.target.value)} placeholder="물품 이름" /><button type="submit" disabled={!newLabel.trim()}>추가</button></div></form>
      <div className="review-inventory-total"><strong>합계</strong><strong>{total}개</strong></div>
      <div className="review-inventory-help"><strong>확인해 주세요</strong><p>크기는 가로 × 깊이 × 높이 순서의 추정값이에요.<br />무게와 분리 여부는 실제 물품 확인이 필요해요.</p><p>수정한 목록은 이 브라우저에 임시로 보관돼요.</p></div>
      <Button onClick={() => navigate(routes.chat)}>채팅으로 돌아가기</Button>
    </div> : <ReviewGate loading={loading} error={error} hasMove={!!move} completed={analysis?.status === 'completed'} />}
  </main>;
}

export function ResultPage() {
  const { move, analysis, draft, loading, error, total } = useReview();
  const ready = analysis?.status === 'completed' && !!draft;
  return <main className="app-page review-result-page"><BrandHeader />
    {ready ? <div className="review-result-body"><div className="review-avatar"><img src={chatAvatar} alt="" /><strong>짐싹</strong></div><h1>이사 유형별 예상 비용</h1><p>확인한 이삿짐 목록과 이사 조건을 기준으로 비교해요.</p><div className="review-result-card"><strong>검토한 이삿짐 {total}개</strong><p>예상 비용을 계산하는 서버 기능이 아직 연결되지 않았어요. 실제 금액은 이사업체와 상담해 확인해주세요.</p></div><Button variant="secondary" onClick={() => navigate(routes.inventory)}>이삿짐 목록 다시 확인하기</Button><button className="review-text-button" onClick={() => navigate(routes.chat)}>채팅으로 돌아가기</button></div> : <ReviewGate loading={loading} error={error} hasMove={!!move} completed={analysis?.status === 'completed'} />}
  </main>;
}
