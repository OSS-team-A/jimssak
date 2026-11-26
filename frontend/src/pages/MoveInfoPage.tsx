import { useEffect, useState } from 'react';
import { BrandHeader } from '../components/common/BrandHeader';
import { Button } from '../components/common/Button';
import { navigate, routes } from '../app/routes';
import { getMove, searchAddresses, updateMove } from '../api/moves';
import type { AddressCandidate, MoveSnapshot, PlaceInput } from '../types/move';
import calendarClockIcon from '../assets/move-calendar-clock.svg';
import searchIcon from '../assets/move-search.svg';
import searchClearIcon from '../assets/move-search-clear.svg';

type Side = 'origin' | 'destination';
type PlaceDraft = { address: string; candidate: AddressCandidate | null; detail: string; floor: number | null };
type Draft = { origin: PlaceDraft; destination: PlaceDraft; date: string; time: string };
const emptyPlace = (): PlaceDraft => ({ address: '', candidate: null, detail: '', floor: null });
const emptyDraft = (): Draft => ({ origin: emptyPlace(), destination: emptyPlace(), date: '', time: '' });
const times = ['07:00', '09:00', '10:00', '13:00', '14:00'];
const savedMove = () => {
  for (const key of ['jimssak-member-move', 'jimssak-guest-move']) {
    try {
      const value = sessionStorage.getItem(key);
      if (value) return JSON.parse(value) as { move_id: string; guest_token: string };
    } catch { /* A corrupt local draft cannot be used for server requests. */ }
  }
  return null;
};
const toPlace = (place: PlaceDraft): PlaceInput => ({
  road_address: place.address.trim(),
  detail_address: place.detail.trim() || null,
  floor: place.floor,
  ...(place.candidate ? {
    building_name: place.candidate.building_name, adm_cd: place.candidate.adm_cd,
    mountain: place.candidate.mountain, lot_main: place.candidate.lot_main, lot_sub: place.candidate.lot_sub,
  } : {}),
});
const fromSnapshot = (record: MoveSnapshot): Draft => ({
  origin: { address: record.data.origin.value?.road_address ?? '', candidate: null,
    detail: record.data.origin.value?.detail_address ?? '', floor: record.data.origin.value?.floor ?? null },
  destination: { address: record.data.destination.value?.road_address ?? '', candidate: null,
    detail: record.data.destination.value?.detail_address ?? '', floor: record.data.destination.value?.floor ?? null },
  date: record.data.schedule.value?.move_date ?? '',
  time: record.data.schedule.value?.start_time.slice(0, 5) ?? '',
});

export function MoveInfoPage() {
  const move = savedMove();
  const preview = import.meta.env.DEV && new URLSearchParams(window.location.search).has('preview');
  const draftKey = move ? `jimssak-draft-${move.move_id}` : '';
  const [hadDraft] = useState(() => Boolean(draftKey && sessionStorage.getItem(draftKey)));
  const [draft, setDraft] = useState<Draft>(() => {
    try { return JSON.parse(sessionStorage.getItem(draftKey) || 'null') || emptyDraft(); }
    catch { return emptyDraft(); }
  });
  const [customFloor, setCustomFloor] = useState<Record<Side, boolean>>({
    origin: draft.origin.floor !== null && ![-1, 1, 2, 3].includes(draft.origin.floor),
    destination: draft.destination.floor !== null && ![-1, 1, 2, 3].includes(draft.destination.floor),
  });
  const [sheet, setSheet] = useState<Side | 'schedule' | null>(null);
  const [keyword, setKeyword] = useState('');
  const [results, setResults] = useState<AddressCandidate[]>([]);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState('');
  const [confirm, setConfirm] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [month, setMonth] = useState(() => {
    const now = new Date(); return new Date(now.getFullYear(), now.getMonth(), 1);
  });
  const [pickedDate, setPickedDate] = useState(draft.date);
  const [pickedTime, setPickedTime] = useState(draft.time);
  const [tipOpen, setTipOpen] = useState(false);
  const [tipAnchor, setTipAnchor] = useState({ right: 24, bottom: 0 });

  useEffect(() => { if (draftKey) sessionStorage.setItem(draftKey, JSON.stringify(draft)); }, [draft, draftKey]);
  useEffect(() => {
    if (!move) return;
    void getMove(move.move_id, move.guest_token).then(record => {
      if (!hadDraft) setDraft(fromSnapshot(record));
    }).catch(() => setError('저장된 이사 정보를 불러오지 못했어요.'));
  }, [draftKey, hadDraft, move?.move_id, move?.guest_token]);
  useEffect(() => {
    if (!keyword.trim() || keyword.trim().length < 2 || sheet === 'schedule' || sheet === null) {
      setResults([]); setSearchError(''); return;
    }
    let active = true;
    const timer = window.setTimeout(() => {
      setSearching(true); setSearchError('');
      void searchAddresses(keyword.trim()).then(data => { if (active) setResults(data.results); })
        .catch(() => { if (active) { setResults([]); setSearchError('주소 검색을 사용할 수 없어요. 주소를 직접 입력하거나 다시 시도해주세요.'); } })
        .finally(() => { if (active) setSearching(false); });
    }, 350);
    return () => { active = false; window.clearTimeout(timer); };
  }, [keyword, sheet]);

  const updatePlace = (side: Side, change: Partial<PlaceDraft>) =>
    setDraft(current => ({ ...current, [side]: { ...current[side], ...change } }));
  const complete = draft.origin.address.trim() && draft.destination.address.trim()
    && draft.origin.floor !== null && draft.destination.floor !== null && draft.date && draft.time;
  const today = new Date().toLocaleDateString('en-CA', { timeZone: 'Asia/Seoul' });
  const monthDays = new Date(month.getFullYear(), month.getMonth() + 1, 0).getDate();
  const firstDay = new Date(month.getFullYear(), month.getMonth(), 1).getDay();

  async function save() {
    if (!complete) return;
    if (!move) { setError('미리보기에서는 저장할 수 없어요.'); return; }
    setSaving(true); setError('');
    try {
      await updateMove(move.move_id, move.guest_token, {
        origin: toPlace(draft.origin), destination: toPlace(draft.destination),
        schedule: { move_date: draft.date, start_time: draft.time },
      });
      sessionStorage.removeItem(draftKey);
      navigate(routes.analysis);
    } catch { setError('이사 정보를 저장하지 못했어요. 입력 내용을 확인하고 다시 시도해주세요.'); }
    finally { setSaving(false); }
  }

  if (!move && !preview) return <main className="app-page move-info-page"><BrandHeader />
    <h1>이사 정보를 찾을 수 없어요</h1><Button onClick={() => navigate(routes.capture)}>촬영 준비로 돌아가기</Button></main>;

  return <main className="app-page move-info-page">
    <BrandHeader />
    {!confirm ? <>
      <h1>이사 정보를 알려주세요</h1>
      <p className="move-intro">영상 분석 중에 입력해두면 바로 목록을 확인할 수 있어요.</p>
      {(['origin', 'destination'] as const).map(side => <section className="move-block" key={side}>
        <h2>{side === 'origin' ? '출발지' : '도착지'}</h2>
        <div className="move-field move-address-field">
          <button type="button" className="move-address-value" onClick={() => { setSheet(side); setKeyword(''); }}>
            {draft[side].address || '주소 검색'}
          </button>
          {draft[side].address
            ? <button type="button" className="move-address-icon" aria-label={`${side === 'origin' ? '출발지' : '도착지'} 주소 지우기`}
                onClick={() => updatePlace(side, { address: '', candidate: null, detail: '' })}><img src={searchClearIcon} alt="" /></button>
            : <span className="move-search-icon"><img src={searchIcon} alt="" /></span>}
        </div>
        {draft[side].address && <input className="move-field" aria-label={side === 'origin' ? '출발지 상세주소' : '도착지 상세주소'}
          placeholder="상세주소 (동·호)" value={draft[side].detail}
          onChange={e => updatePlace(side, { detail: e.target.value })} />}
        <label className="move-label">층수</label>
        <div className="floor-options">
          {([['지하', -1], ['1층', 1], ['2층', 2], ['3층', 3]] as const).map(([label, value]) =>
            <button type="button" key={label} className={draft[side].floor === value ? 'selected' : ''}
              onClick={() => { setCustomFloor(current => ({ ...current, [side]: false })); updatePlace(side, { floor: value }); }}>{label}</button>)}
          <button type="button" className={customFloor[side] ? 'selected' : ''}
            onClick={() => { setCustomFloor(current => ({ ...current, [side]: true })); updatePlace(side, { floor: null }); }}>직접 입력</button>
        </div>
        {customFloor[side] &&
          <input className="move-field floor-input" type="number" min="-5" max="80" placeholder="예: 5 (숫자만 입력)" aria-label="층수 직접 입력"
            value={draft[side].floor ?? ''} onChange={e => updatePlace(side, { floor: e.target.value === '' ? null : Number(e.target.value) })} />}
      </section>)}
      <section className="move-block schedule-block"><h2>이사 일정</h2>
        <button type="button" className="move-field" onClick={() => { setPickedDate(draft.date); setPickedTime(draft.time); setSheet('schedule'); }}>
          {draft.date && draft.time ? `${draft.date} · ${draft.time}` : '날짜·시간 선택'} <span className="move-calendar-icon"><img src={calendarClockIcon} alt="" /></span>
        </button>
      </section>
      {error && <p className="form-error" role="alert">{error}</p>}
      <div className="move-actions"><Button disabled={!complete} onClick={() => setConfirm(true)}>저장하기</Button></div>
    </> : <>
      <span className="capture-eyebrow">이사 정보 최종 확인</span>
      <h1>이사 정보를<br />마지막으로 확인할게요</h1>
      <p className="move-intro">입력한 주소를 통해 정보를 확인했어요.</p>
      {(['origin', 'destination'] as const).map(side => <div className="confirm-card" key={side}>
        <small>{side === 'origin' ? '출발지 주소' : '도착지 주소'}</small>
        <strong>{draft[side].address}</strong>
        <p>상세주소 {draft[side].detail || '직접 입력'}</p>
        <p>이사 층 {draft[side].floor}층 · 승강기 확인 필요</p>
      </div>)}
      <div className="confirm-card"><small>이사 날짜와 시간</small><strong>{draft.date} · {draft.time}</strong></div>
      {error && <p className="form-error" role="alert">{error}</p>}
      <div className="move-actions confirm-actions">
        <button type="button" aria-label="수정하기" onClick={() => setConfirm(false)}>‹</button>
        <Button disabled={saving} onClick={() => void save()}>{saving ? '저장 중...' : '저장하기'}</Button>
      </div>
    </>}
    {sheet && <div className="sheet-backdrop" onClick={() => setSheet(null)}>
      <div className={`move-sheet ${sheet === 'schedule' ? 'move-sheet--schedule' : 'move-sheet--address'}`} role="dialog" aria-modal="true" aria-label={sheet === 'schedule' ? '이사 날짜와 시간' : '주소 검색'} onClick={e => e.stopPropagation()}>
        <span className="sheet-handle" />
        {sheet === 'schedule' ? <>
          <h2>이사 날짜와 시간</h2>
          <div className="calendar-head">
            <button type="button" aria-label="이전 달" onClick={() => setMonth(new Date(month.getFullYear(), month.getMonth() - 1, 1))}>‹</button>
            <strong>{month.getFullYear()}년 {month.getMonth() + 1}월</strong>
            <button type="button" aria-label="다음 달" onClick={() => setMonth(new Date(month.getFullYear(), month.getMonth() + 1, 1))}>›</button>
          </div>
          <div className="calendar-grid">{['일', '월', '화', '수', '목', '금', '토'].map(day => <span key={day}>{day}</span>)}
            {Array.from({ length: firstDay }, (_, index) => <span key={`gap-${index}`} />)}
            {Array.from({ length: monthDays }, (_, index) => {
              const date = `${month.getFullYear()}-${String(month.getMonth() + 1).padStart(2, '0')}-${String(index + 1).padStart(2, '0')}`;
              return <button type="button" key={date} disabled={date < today} className={pickedDate === date ? 'selected' : ''}
                onClick={() => setPickedDate(date)}>{index + 1}</button>;
            })}
          </div>
          <div className="time-heading"><h3>이사 시작 시간</h3><button type="button" className="time-tip-button" aria-expanded={tipOpen} aria-controls="time-tip" onClick={event => {
            const bounds = event.currentTarget.getBoundingClientRect();
            setTipAnchor({ right: window.innerWidth - bounds.right, bottom: window.innerHeight - bounds.top + 17 });
            setTipOpen(open => !open);
          }}>tip</button></div>
          <div className="time-options">{times.map(time => <button type="button" key={time} className={pickedTime === time ? 'selected' : ''}
            onClick={() => setPickedTime(time)}>{Number(time.slice(0, 2)) < 12 ? '오전' : '오후'} {Number(time.slice(0, 2)) % 12 || 12}:00</button>)}</div>
          <div className="move-sheet-actions"><Button disabled={!pickedDate || !pickedTime} onClick={() => { setDraft(current => ({ ...current, date: pickedDate, time: pickedTime })); setSheet(null); }}>선택 완료</Button></div>
        </> : <>
          <h2>{sheet === 'origin' ? '출발지' : '도착지'} 주소 검색</h2>
          <div className="address-search"><input autoFocus placeholder="도로명, 건물명 검색" value={keyword} onChange={e => setKeyword(e.target.value)} />
            <button type="button" aria-label="검색어 지우기" onClick={() => setKeyword('')}><img src={searchClearIcon} alt="" /></button></div>
          <p className="move-label">검색 결과</p>
          {searching && <p>검색 중...</p>}
          {searchError && <p role="alert">{searchError}</p>}
          {!searching && keyword.length >= 2 && results.length === 0 && !searchError && <p>검색 결과가 없어요.</p>}
          <div className="address-results">{results.map(item => <button type="button" key={item.adm_cd + item.lot_main + item.road_address}
            onClick={() => { updatePlace(sheet, { address: item.road_address, candidate: item }); setSheet(null); }}>
            <strong>{item.road_address}</strong><small>{item.jibun_address}</small></button>)}</div>
          {results.length > 0 && <p className="address-selection-hint">주소를 선택하면 입력 화면으로 돌아가요</p>}
          <div className="move-sheet-actions"><Button disabled={!keyword.trim()} onClick={() => {
            updatePlace(sheet, { address: keyword.trim(), candidate: null }); setSheet(null);
          }}>입력한 주소 사용하기</Button></div>
        </>}
      </div>
      {sheet === 'schedule' && tipOpen && <div id="time-tip" className="time-tip-content" role="tooltip" style={tipAnchor} onClick={event => event.stopPropagation()}>
        오전이사: 하루 첫 일정이라 시작 시간이 비교적 안정적이지만 수요가 많아 오후 이사보다 비용이 높게 책정되는 편입니다.<br />오후이사: 오전 이사보다 보통 수만 원~20만 원 가량 저렴한 경우가 많지만, 앞 팀 작업이 늦어지면 시작 시간이 밀릴 수 있고, 마무리가 늦어질 수 있습니다.
      </div>}
    </div>}
  </main>;
}
