import { useEffect, useState } from 'react';
import { navigate, routes } from '../app/routes';
import { useCurrentMove } from '../hooks/useCurrentMove';
import type { InventoryItem } from '../types/move';

export function InventoryPage() {
  const { move, snapshot, loading, error, refresh } = useCurrentMove();
  const items = snapshot?.data.inventory.items || [];
  const [expanded, setExpanded] = useState<string | null>(null);
  const [draft, setDraft] = useState<InventoryItem[] | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const [name, setName] = useState('');
  const key = move ? `jimssak-inventory-draft-${move.id}` : '';
  useEffect(() => {
    if (!key || !snapshot || !items.length) return;
    try {
      const saved = JSON.parse(sessionStorage.getItem(key) || 'null') as { version?: string; items?: InventoryItem[] } | null;
      setDraft(saved?.version === snapshot.data.inventory.version && Array.isArray(saved?.items) ? saved.items : items);
    } catch { setDraft(items); }
  }, [key, snapshot]);
  const rows = draft || items;
  const change = (next: InventoryItem[]) => {
    setDraft(next);
    if (key) sessionStorage.setItem(key, JSON.stringify({ version: snapshot?.data.inventory.version, items: next }));
  };
  const update = (id: string, fields: Partial<InventoryItem>) => change(rows.map(item => item.id === id ? { ...item, ...fields } : item));
  const add = () => {
    if (!name.trim()) return;
    const id = `local-${crypto.randomUUID()}`;
    change([...rows, { id, name: name.trim(), quantity: 1 }]);
    setName(''); setAdding(false); setExpanded(id);
  };
  const available = snapshot?.data.inventory.status === 'ready' && items.length > 0;
  return <main className="app-page inventory-page">
    <header className="inventory-header"><strong>이삿짐 목록</strong><button type="button" aria-label="닫기" onClick={() => navigate(routes.chat)}>×</button></header>
    <div className="inventory-body">
      <h1>이삿짐 목록</h1>
      <p className="inventory-subtitle">영상 분석을 바탕으로 작성한 목록이에요.<br />확인한 내용은 명세서에 반영돼요.</p>
      {loading ? <p className="inventory-feedback" role="status">목록을 불러오는 중이에요.</p> : error ? <div className="inventory-feedback" role="alert">{error}<button type="button" onClick={() => void refresh()}>다시 시도하기</button></div> : !available ? <div className="inventory-feedback" role="status">{move ? '아직 확인할 이삿짐 목록이 없어요. 영상 분석 결과를 기다리고 있어요.' : '진행 중인 이사 기록이 없어요.'}<button type="button" onClick={() => navigate(move ? routes.analysis : routes.home)}>{move ? '분석 상태 확인하기' : '촬영 시작하기'}</button></div> : <>
        <div className="inventory-summary">전체 {rows.filter(item => !item.excluded).length}종 · {rows.filter(item => !item.excluded).reduce((sum, item) => sum + item.quantity, 0)}개</div>
        <div className="inventory-columns"><span>번호</span><span>물품명</span><span>수량</span><span>예상 크기</span></div>
        {rows.map((item, index) => <div className="inventory-entry" key={item.id}>
          <button className="inventory-row" type="button" aria-expanded={expanded === item.id} onClick={() => setExpanded(expanded === item.id ? null : item.id)}><span>{index + 1}</span><strong>{item.name}{item.excluded ? ' · 제외' : ''}</strong><span>{item.quantity}개</span><span>{item.estimated_size || '확인 필요'}　{expanded === item.id ? '⌃' : '⌄'}</span></button>
          {expanded === item.id && <div className="inventory-details">
            {item.photo_url && <img src={item.photo_url} alt={`${item.name} 분석 사진`} />}
            <dl><div><dt>예상 크기</dt><dd>{item.estimated_size || '확인 필요'}</dd></div><div><dt>예상 무게</dt><dd>{item.estimated_weight || '확인 필요'}</dd></div><div><dt>분리 여부</dt><dd>{item.disassembly || '확인 필요'}</dd></div><div><dt>위치</dt><dd>{item.location || '확인 필요'}</dd></div><div><dt>특이사항</dt><dd>{item.note || '없음'}</dd></div></dl>
            {editing === item.id ? <div className="inventory-edit"><label>물품명<input value={item.name} onChange={event => update(item.id, { name: event.target.value })} /></label><label>수량<input type="number" min="1" inputMode="numeric" value={item.quantity} onChange={event => update(item.id, { quantity: Math.max(1, Number(event.target.value) || 1) })} /></label><label>예상 크기<input value={item.estimated_size || ''} onChange={event => update(item.id, { estimated_size: event.target.value })} /></label><label>특이사항<input value={item.note || ''} onChange={event => update(item.id, { note: event.target.value })} /></label><button type="button" onClick={() => setEditing(null)}>수정 닫기</button></div> : <button type="button" onClick={() => setEditing(item.id)}>수정하기</button>}
            <button type="button" onClick={() => update(item.id, { excluded: !item.excluded })}>{item.excluded ? '이사 목록에 다시 포함' : '이사 목록에서 제외'}</button>
          </div>}
        </div>)}
        {adding ? <form className="inventory-add" onSubmit={event => { event.preventDefault(); add(); }}><label>빠진 물품<input autoFocus value={name} onChange={event => setName(event.target.value)} placeholder="물품명을 입력해주세요" /></label><button type="submit" disabled={!name.trim()}>추가</button><button type="button" onClick={() => setAdding(false)}>취소</button></form> : <button className="inventory-add-button" type="button" onClick={() => setAdding(true)}>+ 빠진 물품 추가하기</button>}
        <div className="inventory-total"><strong>합계</strong><strong>{rows.filter(item => !item.excluded).reduce((sum, item) => sum + item.quantity, 0)}개</strong></div>
        <h2>확인해 주세요</h2><p className="inventory-notes">• 크기는 가로 × 깊이 × 높이 순서의 추정값이에요.<br />• 무게와 분리 여부는 실제 물품 확인이 필요해요.<br />• 누락된 물품이나 두고 갈 물품은 채팅으로 알려주세요.</p>
        <p className="inventory-unsaved" role="status">수정 내용은 이 브라우저에 임시 보관돼요. 서버 저장 기능은 아직 연결되지 않았어요.</p>
      </>}
    </div>
  </main>;
}
