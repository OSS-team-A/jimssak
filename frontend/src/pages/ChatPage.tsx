import { useState } from 'react';
import { BrandHeader } from '../components/common/BrandHeader';
import { Button } from '../components/common/Button';
import { navigate, routes } from '../app/routes';
import { useCurrentMove } from '../hooks/useCurrentMove';
import avatar from '../assets/chatAvatar.svg';
import sendIcon from '../assets/chatSend.svg';
import listIcon from '../assets/chatList.svg';

type Question = 'review' | 'missing' | 'excluded' | 'special' | 'quote';
const next: Record<Exclude<Question, 'quote'>, Question> = { review: 'missing', missing: 'excluded', excluded: 'special', special: 'quote' };
const prompts: Record<Question, string> = {
  review: '영상에서 찾은 이삿짐을 정리했어요.\n목록 탭을 눌러 맞는지 확인해주세요.',
  missing: '빠진 물품이 있나요?',
  excluded: '이사할 때 두고 갈 물품이 있나요?',
  special: '이삿짐에 대해 더 알려주실 특이사항이 있나요?',
  quote: '목록 확인이 끝났어요. 예상 비용을 비교해볼까요?',
};
const choice: Record<Exclude<Question, 'quote'>, string> = { review: '확인했어요', missing: '빠진 물품 없어요', excluded: '두고 갈 물품 없어요', special: '특이사항 없어요' };

export function ChatPage() {
  const { move, snapshot, loading, error, refresh } = useCurrentMove();
  const ready = snapshot?.data.inventory.status === 'ready' && Boolean(snapshot.data.inventory.items?.length);
  const key = move ? `jimssak-chat-${move.id}` : '';
  const [question, setQuestion] = useState<Question>(() => {
    try { return (JSON.parse(sessionStorage.getItem(key) || '{}') as { question?: Question }).question || 'review'; }
    catch { return 'review'; }
  });
  const [messages, setMessages] = useState<string[]>(() => {
    try { return (JSON.parse(sessionStorage.getItem(key) || '{}') as { messages?: string[] }).messages || []; }
    catch { return []; }
  });
  const [input, setInput] = useState('');
  const [modal, setModal] = useState(false);
  const save = (step: Question, sent: string[]) => { setQuestion(step); setMessages(sent); if (key) sessionStorage.setItem(key, JSON.stringify({ question: step, messages: sent })); };
  const choose = () => {
    if (question === 'quote') { navigate(routes.result); return; }
    save(next[question], [...messages, choice[question]]);
  };
  const send = () => {
    if (!input.trim()) return;
    save(question, [...messages, input.trim()]); setInput('');
  };
  const items = snapshot?.data.inventory.items || [];
  return <main className="app-page chat-page">
    <BrandHeader />
    <div className="chat-scroll">
      {loading ? <p className="chat-loading" role="status">이삿짐 목록을 불러오는 중이에요.</p> : error ? <div className="inventory-feedback" role="alert">{error}<button type="button" onClick={() => void refresh()}>다시 시도하기</button></div> : !ready ? <div className="inventory-feedback" role="status">{move ? '분석된 이삿짐 목록이 아직 없어요.' : '진행 중인 이사 기록이 없어요.'}<button type="button" onClick={() => navigate(move ? routes.analysis : routes.home)}>{move ? '분석 상태 확인하기' : '촬영 시작하기'}</button></div> : <>
        <div className="chat-speaker"><img src={avatar} alt="" /><strong>짐싹</strong></div>
        <p className="chat-prompt">{prompts[question]}</p>
        {question === 'review' && <button type="button" className="chat-summary" onClick={() => navigate(routes.inventory)}><strong>이삿짐 목록 <span>›</span></strong><hr />{items.slice(0, 5).map(item => <span className="chat-summary-row" key={item.id}>{item.name}<b>{item.quantity}개　›</b></span>)}<hr /><small>총 {items.reduce((n, item) => n + item.quantity, 0)}개 · 누르면 자세히 볼 수 있어요</small></button>}
        {question === 'missing' && <p className="chat-example">예: “모니터 27인치 추가해줘.”</p>}
        {messages.map((message, index) => <div className="chat-user-message" key={`${index}-${message}`}>{message}</div>)}
        {messages.some(message => !Object.values(choice).includes(message)) && <p className="chat-pending" role="status">입력한 내용은 이 브라우저에 임시 보관돼요. 자동 해석·목록 반영 기능은 아직 연결되지 않았어요. 목록에서 직접 수정할 수 있어요.</p>}
        <Button onClick={choose}>{question === 'quote' ? '예상 비용 비교하기' : choice[question]}</Button>
      </>}
    </div>
    <div className="chat-bottom"><div className="chat-shortcuts"><button type="button" onClick={() => navigate(routes.inventory)}><img src={listIcon} alt="" /> 이삿짐 목록 보러가기</button><button type="button" onClick={() => setModal(true)}>↓ 이삿짐 명세서 PDF 받기</button></div>
      <form className="chat-input" onSubmit={event => { event.preventDefault(); send(); }}><input value={input} onChange={event => setInput(event.target.value)} placeholder="수정할 내용을 알려주세요" aria-label="수정할 내용" disabled={!ready} /><button type="submit" aria-label="전송" disabled={!ready || !input.trim()}><img src={sendIcon} alt="" /></button></form>
      <small>문서는 목록과 이사 조건을 확인한 뒤 만들 수 있어요</small>
    </div>
    {modal && <div className="chat-modal-backdrop" role="presentation" onClick={() => setModal(false)}><div className="chat-modal" role="dialog" aria-modal="true" aria-label="명세서 발급 안내" onClick={event => event.stopPropagation()}><h2>이삿짐 명세서 PDF</h2><p>목록 및 이사업체 확정 후 받을 수 있어요.</p><Button onClick={() => setModal(false)}>확인</Button></div></div>}
  </main>;
}
