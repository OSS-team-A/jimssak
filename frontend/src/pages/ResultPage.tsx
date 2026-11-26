import { useState } from 'react';
import { BrandHeader } from '../components/common/BrandHeader';
import { Button } from '../components/common/Button';
import { navigate, routes } from '../app/routes';
import { useCurrentMove } from '../hooks/useCurrentMove';
import type { QuoteOption } from '../types/move';
import avatar from '../assets/chatAvatar.svg';

const labels: Record<QuoteOption['type'], string> = { general: '일반 이사', partial: '반포장 이사', full: '포장 이사' };
const cost = (amount: number) => `${Math.round(amount).toLocaleString('ko-KR')}원`;

export function ResultPage() {
  const { move, snapshot, loading, error, refresh } = useCurrentMove();
  const draftKey = move ? `jimssak-quote-choice-${move.id}` : '';
  const [selected, setSelected] = useState<QuoteOption['type'] | null>(() => {
    const stored = sessionStorage.getItem(draftKey);
    return stored === 'general' || stored === 'partial' || stored === 'full' ? stored : null;
  });
  const [modal, setModal] = useState(false);
  const quote = snapshot?.quote;
  const ready = quote?.status === 'ready' && Boolean(quote.options?.length);
  const choose = (type: QuoteOption['type']) => { setSelected(type); if (draftKey) sessionStorage.setItem(draftKey, type); };
  return <main className="app-page quote-page">
    <BrandHeader />
    <div className="quote-scroll">
      <div className="chat-speaker"><img src={avatar} alt="" /><strong>짐싹</strong></div>
      <p className="chat-prompt">이사 유형별 예상 비용을 비교해볼게요.</p>
      {loading ? <p className="inventory-feedback" role="status">예상 비용을 불러오는 중이에요.</p> : error ? <div className="inventory-feedback" role="alert">{error}<button type="button" onClick={() => void refresh()}>다시 시도하기</button></div> : !ready ? <div className="inventory-feedback" role="status">{move ? '예상 비용 산정 기능이 아직 연결되지 않았어요. 목록과 이사 정보는 유지돼요.' : '진행 중인 이사 기록이 없어요.'}<button type="button" onClick={() => navigate(move ? routes.chat : routes.home)}>{move ? '이삿짐 목록 확인하기' : '촬영 시작하기'}</button></div> : <div className="quote-panel"><h1>이사 유형별 예상 비용</h1>{quote.options?.map(option => <button key={option.type} type="button" className={`quote-option ${selected === option.type ? 'quote-option--selected' : ''}`} onClick={() => choose(option.type)} aria-pressed={selected === option.type}><strong>{labels[option.type]} <span>{cost(option.min_amount)}~{cost(option.max_amount)}</span></strong><small>{option.description}</small></button>)}<p>확인한 명세서를 내려받아 여러 업체에 전달하고 비교해보세요.</p><Button onClick={() => setModal(true)}>이삿짐 명세서 PDF 받기</Button><button className="quote-history-link" type="button" onClick={() => navigate(routes.history)}>나의 이사 기록에서 이사 완료하기↗︎</button></div>}
    </div>
    <div className="chat-bottom"><div className="chat-shortcuts"><button type="button" onClick={() => navigate(routes.inventory)}>☷ 이삿짐 목록 보러가기</button><button type="button" onClick={() => setModal(true)}>↓ 이삿짐 명세서 PDF 받기</button></div></div>
    {modal && <div className="chat-modal-backdrop" role="presentation" onClick={() => setModal(false)}><div className="chat-modal" role="dialog" aria-modal="true" aria-label="PDF 발급 안내" onClick={event => event.stopPropagation()}><h2>이삿짐 명세서 PDF</h2><p>{ready ? '명세서 발급 API가 아직 연결되지 않았어요. 이사 유형 선택 내용은 서버에 확정되지 않았어요.' : '목록과 이사 조건을 확인한 뒤 받을 수 있어요.'}</p><Button onClick={() => setModal(false)}>확인</Button></div></div>}
  </main>;
}
