import { useEffect, useRef, useState, type ChangeEvent } from 'react';
import { createMove, uploadMoveVideo } from '../api/moves';
import { ApiError } from '../api/client';
import { navigate, routes } from '../app/routes';
import { BrandHeader } from '../components/common/BrandHeader';
import { Button } from '../components/common/Button';
import mascot from '../assets/capture-mascot.webp';
import analysisMascot from '../assets/analysis-mascot.webp';
import bulbIcon from '../assets/capture-bulb.svg';
import cameraIcon from '../assets/capture-camera.svg';
import arrowIcon from '../assets/capture-arrow.svg';
import permissionCameraIcon from '../assets/capture-permission-camera.svg';
import doubleArrowIcon from '../assets/capture-double-arrow.svg';

type Screen = 'prep' | 'camera' | 'status';
const acceptedVideo = (file: File) => /\.(mp4|mov)$/i.test(file.name)
  && ['video/mp4', 'video/quicktime', 'application/octet-stream', ''].includes(file.type);

export function CapturePage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const cancelledRef = useRef(false);
  const moveRef = useRef<{ move_id: string; guest_token: string } | null>((() => {
    try {
      const saved = sessionStorage.getItem('jimssak-member-move') || sessionStorage.getItem('jimssak-guest-move');
      return saved ? JSON.parse(saved) : null;
    } catch { return null; }
  })());
  const [screen, setScreen] = useState<Screen>('prep');
  const [permissionOpen, setPermissionOpen] = useState(false);
  const [recording, setRecording] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploaded, setUploaded] = useState(false);
  const [error, setError] = useState('');
  const mp4Recording = typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported('video/mp4');

  useEffect(() => () => {
    cancelledRef.current = true;
    if (recorderRef.current?.state === 'recording') recorderRef.current.stop();
    streamRef.current?.getTracks().forEach(track => track.stop());
  }, []);

  useEffect(() => {
    if (screen === 'camera' && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
      void videoRef.current.play().catch(() => {});
    }
  }, [screen]);

  async function openCamera() {
    setPermissionOpen(false);
    setError('');
    if (!mp4Recording) {
      fileRef.current?.click();
      return;
    }
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('카메라를 사용할 수 없어요.');
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: true });
      streamRef.current?.getTracks().forEach(track => track.stop());
      streamRef.current = stream;
      setScreen('camera');
    } catch {
      setError('카메라를 사용할 수 없어요. 브라우저의 카메라·마이크 권한을 확인해주세요.');
    }
  }

  async function upload(file: File) {
    setScreen('status');
    setUploading(true);
    setUploaded(false);
    setError('');
    streamRef.current?.getTracks().forEach(track => track.stop());
    streamRef.current = null;
    try {
      if (!acceptedVideo(file)) throw new Error('MP4 또는 MOV 영상만 업로드할 수 있어요.');
      let move = moveRef.current;
      if (!move) {
        const created = await createMove();
        move = { move_id: created.move_id, guest_token: created.guest_token ?? '' };
        moveRef.current = move;
        sessionStorage.setItem(move.guest_token ? 'jimssak-guest-move' : 'jimssak-member-move', JSON.stringify(move));
      }
      await uploadMoveVideo(move.move_id, move.guest_token, file);
      sessionStorage.setItem(move.guest_token ? 'jimssak-guest-move' : 'jimssak-member-move', JSON.stringify(move));
      setUploaded(true);
      navigate(routes.moveInfo);
    } catch (cause) {
      if (cause instanceof ApiError) {
        setError(cause.status === 413 ? '영상 용량이 너무 커요. 짧게 다시 촬영해주세요.'
          : cause.status === 415 ? '영상 형식을 확인하지 못했어요. MP4·MOV로 다시 촬영해주세요.'
          : '영상 업로드에 실패했어요. 연결 상태를 확인하고 다시 촬영해주세요.');
        if (cause.status === 404) moveRef.current = null;
      } else setError(cause instanceof Error ? cause.message : '영상 업로드에 실패했어요. 다시 촬영해주세요.');
    } finally {
      setUploading(false);
    }
  }

  function start() {
    if (!streamRef.current || !mp4Recording) return;
    try {
      chunksRef.current = [];
      const recorder = new MediaRecorder(streamRef.current, { mimeType: 'video/mp4' });
      recorderRef.current = recorder;
      recorder.ondataavailable = event => { if (event.data.size) chunksRef.current.push(event.data); };
      recorder.onstop = () => {
        setRecording(false);
        if (cancelledRef.current) return;
        if (!chunksRef.current.length) { setError('촬영된 영상이 없어요. 다시 시도해주세요.'); return; }
        void upload(new File(chunksRef.current, 'room-video.mp4', { type: 'video/mp4' }));
      };
      recorder.start();
      setRecording(true);
      setError('');
    } catch {
      setError('녹화를 시작할 수 없어요. 카메라 권한을 확인해주세요.');
    }
  }

  function chooseFile(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0];
    event.target.value = '';
    if (selected) void upload(selected);
  }

  return <main className="app-page capture-page">
    {screen === 'prep' && <div className="capture-prep">
      <BrandHeader />
      <span className="capture-eyebrow">촬영 준비</span>
      <h1>방을 찍기 전에<br />이것만 확인해주세요</h1>
      <div className="capture-mascot" role="img" aria-label="짐싹 캐릭터"><img src={mascot} alt="" /></div>
      <div className="capture-tips">
        <div className="capture-tip"><img className="capture-tip-icon" src={bulbIcon} alt="" /><div><strong>조명을 켜주세요</strong><p>밝을수록 물품을 확인하기 쉬워요.</p></div></div>
        <div className="capture-tip"><img className="capture-tip-icon" src={cameraIcon} alt="" /><div><strong>큰 가구부터, 빠짐없이</strong><p>가구 전체와 가려진 공간까지 담아주세요.</p></div></div>
        <div className="capture-tip"><img className="capture-tip-icon" src={arrowIcon} alt="" /><div><strong>수평을 맞추고 천천히</strong><p>한 방향으로 방을 둘러보듯 촬영해요.</p></div></div>
      </div>
      {error && <p role="alert">{error}</p>}
      <div className="capture-prep-actions"><Button onClick={() => setPermissionOpen(true)}>촬영 시작하기</Button></div>
      <input ref={fileRef} type="file" accept=".mp4,.mov,video/mp4,video/quicktime" capture="environment" onChange={chooseFile} hidden aria-label="카메라로 영상 촬영" />
      {permissionOpen && <div className="capture-permission-backdrop" onClick={() => setPermissionOpen(false)}>
        <div className="capture-permission" role="dialog" aria-modal="true" aria-label="카메라 사용 안내" onClick={event => event.stopPropagation()}>
          <button className="permission-close" type="button" aria-label="닫기" onClick={() => setPermissionOpen(false)}>×</button>
          <img src={permissionCameraIcon} width="40" height="40" alt="" />
          <h2>방을 촬영하려면<br />카메라 사용이 필요해요</h2>
          <p>이삿짐 목록을 만들기 위해 카메라를 사용해요.<br />다음 브라우저 안내에서 ‘허용’을 눌러주세요.</p>
          <Button onClick={() => void openCamera()}>카메라 사용 허용하기</Button>
        </div>
      </div>}
    </div>}
    {screen === 'camera' && <div className="capture-camera">
      <video ref={videoRef} autoPlay muted playsInline aria-label="카메라 미리보기" />
      <div className="capture-camera-guide" aria-hidden="true" />
      <p className="capture-camera-hint">{recording ? <>한 방향으로 천천히 이동해주세요 <img className="capture-double-arrow" src={doubleArrowIcon} alt="" /></> : '수평을 맞추고 천천히 움직여주세요'}</p>
      {error && <p role="alert" className="capture-camera-error">{error}</p>}
      <div className="capture-camera-controls"><button type="button" className={recording ? 'capture-shutter capture-shutter--recording' : 'capture-shutter'} onClick={() => recording ? recorderRef.current?.stop() : start()} aria-label={recording ? '촬영 종료' : '촬영 시작'} /></div>
    </div>}
    {screen === 'status' && <div className="capture-status">
      <BrandHeader />
      <h1>{uploading ? '영상을 저장하고 있어요' : uploaded ? '영상이 저장됐어요' : '영상을 저장하지 못했어요'}</h1>
      <p>{uploading ? '촬영한 영상을 업로드하고 있어요.' : uploaded ? '영상 분석은 아직 연결되지 않았어요.' : error}</p>
      <div className="capture-status-mascot"><img src={analysisMascot} alt="" /></div>
      <div className="capture-status-strip" role="status">{uploading ? '영상 업로드 중' : uploaded ? '영상 업로드 완료' : '영상 업로드 실패'}</div>
      {uploaded && <p className="capture-status-detail">이삿짐 목록과 명세서 생성은 아직 연결되지 않았어요.</p>}
      {!uploading && !uploaded && <div className="capture-status-actions"><Button variant="secondary" onClick={() => { setError(''); setScreen('prep'); }}>영상 다시 촬영하기</Button></div>}
    </div>}
  </main>;
}
