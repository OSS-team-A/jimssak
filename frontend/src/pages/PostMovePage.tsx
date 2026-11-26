import { useEffect, useRef, useState } from 'react';
import { BrandHeader } from '../components/common/BrandHeader';
import { Button } from '../components/common/Button';
import { navigate, routes } from '../app/routes';
import { useCurrentMove } from '../hooks/useCurrentMove';
import mascot from '../assets/capture-mascot.webp';

export function PostMovePage() {
  const { move, snapshot, loading, error, refresh } = useCurrentMove();
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState('');
  const [message, setMessage] = useState('');
  useEffect(() => {
    if (!file) { setPreview(''); return; }
    const url = URL.createObjectURL(file); setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  const select = (candidate?: File) => {
    if (!candidate) return;
    if (!/\.(mp4|mov)$/i.test(candidate.name) || !['video/mp4', 'video/quicktime', 'application/octet-stream', ''].includes(candidate.type)) {
      setMessage('MP4 또는 MOV 영상만 선택할 수 있어요.'); return;
    }
    setFile(candidate); setMessage('이사 후 영상 업로드 기능은 아직 연결되지 않았어요. 선택한 영상은 서버에 저장되지 않았어요.');
  };
  return <main className="app-page postmove-page">
    <BrandHeader />
    <div className="postmove-body">
      <span className="capture-eyebrow">이사 후 물품 확인</span>
      <h1>이사 후 물품 확인하기</h1>
      <p>이사 후 방을 촬영하면 이사 전에 찍어둔 영상과 비교해드려요. 상태가 달라 보이거나 확인되지 않은 물품을 살펴볼 수 있어요.</p>
      <img className="postmove-mascot" src={mascot} alt="" />
      {loading ? <p role="status">이사 기록 확인 중</p> : error ? <div className="inventory-feedback" role="alert">{error}<button type="button" onClick={() => void refresh()}>다시 시도하기</button></div> : !move ? <p role="status">진행 중인 이사 기록이 없어요.</p> : snapshot?.postmove?.status === 'comparing' ? <p role="status">이사 전후 영상을 비교하고 있어요.</p> : snapshot?.postmove?.status === 'failed' ? <p role="alert">{snapshot.postmove.error || '영상 비교에 실패했어요.'}</p> : null}
      {preview && <video className="postmove-preview" src={preview} controls playsInline aria-label="선택한 이사 후 영상 미리보기" />}
      {message && <p className="postmove-message" role="status">{message}</p>}
    </div>
    <div className="postmove-actions">
      {snapshot?.postmove?.status === 'ready' && <Button onClick={() => navigate(routes.comparison)}>비교 결과 확인하기</Button>}
      <Button onClick={() => move ? inputRef.current?.click() : navigate(routes.home)}>{move ? file ? '영상 다시 선택하기' : '이사 후 물품 확인하기' : '촬영 시작하기'}</Button>
      <Button variant="secondary" onClick={() => navigate(routes.history)}>나의 이사 기록</Button>
    </div>
    <input ref={inputRef} hidden type="file" accept=".mp4,.mov,video/mp4,video/quicktime" capture="environment" aria-label="이사 후 영상 선택" onChange={event => { select(event.target.files?.[0]); event.target.value = ''; }} />
  </main>;
}
