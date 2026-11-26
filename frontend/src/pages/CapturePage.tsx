import { useEffect, useRef, useState, type ChangeEvent } from 'react';
import { createMove, uploadMoveVideo } from '../api/moves';
import { ApiError } from '../api/client';
import { navigate, routes } from '../app/routes';
import { BrandHeader } from '../components/common/BrandHeader';
import { Button } from '../components/common/Button';
import mascot from '../assets/capture-mascot-prep.png';
import analysisMascot from '../assets/analysis-mascot.webp';
import bulbIcon from '../assets/capture-bulb.svg';
import cameraIcon from '../assets/capture-camera.svg';
import shoesIcon from '../assets/capture-shoes.png';
import permissionCameraIcon from '../assets/capture-permission-camera.svg';
import doubleArrowIcon from '../assets/capture-double-arrow.svg';
import levelEllipse from '../assets/capture-level-ellipse.svg';
import levelLine from '../assets/capture-level-line.svg';
import tiltedLevelLine from '../assets/capture-level-tilted.svg';

type Screen = 'prep' | 'camera' | 'review' | 'status';
const cameraPermissionKey = 'jimssak-camera-permission-granted';
const acceptedVideo = (file: File) => /\.(mp4|mov)$/i.test(file.name)
  && ['video/mp4', 'video/quicktime', 'application/octet-stream', ''].includes(file.type);

export function CapturePage() {
  const params = new URLSearchParams(window.location.search);
  const afterMove = params.get('mode') === 'after';
  const preview = import.meta.env.DEV && params.get('preview') === '1';
  const previewState = params.get('state');
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const reviewUrlRef = useRef<string | null>(null);
  const cancelledRef = useRef(false);
  const moveRef = useRef<{ move_id: string; guest_token: string } | null>((() => {
    try {
      const saved = sessionStorage.getItem('jimssak-member-move') || sessionStorage.getItem('jimssak-guest-move');
      return saved ? JSON.parse(saved) : null;
    } catch { return null; }
  })());
  const [screen, setScreen] = useState<Screen>(preview && previewState === 'review' ? 'review' : preview && (previewState === 'camera' || previewState === 'recording' || previewState === 'tilted') ? 'camera' : 'prep');
  const [permissionOpen, setPermissionOpen] = useState(false);
  const [recording, setRecording] = useState(preview && (previewState === 'recording' || previewState === 'tilted'));
  const [reviewFile, setReviewFile] = useState<File | null>(null);
  const [reviewUrl, setReviewUrl] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploaded, setUploaded] = useState(false);
  const [error, setError] = useState('');
  const mp4Recording = typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported('video/mp4');

  useEffect(() => () => {
    cancelledRef.current = true;
    if (recorderRef.current?.state === 'recording') recorderRef.current.stop();
    streamRef.current?.getTracks().forEach(track => track.stop());
    if (reviewUrlRef.current) URL.revokeObjectURL(reviewUrlRef.current);
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
    if (preview) { setScreen('camera'); return; }
    if (!mp4Recording) {
      fileRef.current?.click();
      return;
    }
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('카메라를 사용할 수 없어요.');
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: true });
      streamRef.current?.getTracks().forEach(track => track.stop());
      streamRef.current = stream;
      sessionStorage.setItem(cameraPermissionKey, 'true');
      setScreen('camera');
    } catch {
      setError('카메라를 사용할 수 없어요. 브라우저의 카메라·마이크 권한을 확인해주세요.');
    }
  }

  function showReview(file: File) {
    if (!acceptedVideo(file)) { setError('MP4 또는 MOV 영상만 업로드할 수 있어요.'); return; }
    if (reviewUrlRef.current) URL.revokeObjectURL(reviewUrlRef.current);
    const url = URL.createObjectURL(file);
    reviewUrlRef.current = url;
    setReviewFile(file);
    setReviewUrl(url);
    streamRef.current?.getTracks().forEach(track => track.stop());
    streamRef.current = null;
    setError('');
    setScreen('review');
  }

  async function upload(file: File) {
    if (afterMove) {
      if (preview) {
        window.history.pushState({}, '', `${routes.analysis}?mode=after&preview=1&state=waiting`);
        window.dispatchEvent(new PopStateEvent('popstate'));
      } else setError('이사 후 영상 업로드 API가 아직 연결되지 않았어요. 이사 전 영상은 그대로 보관돼요.');
      return;
    }
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
        showReview(new File(chunksRef.current, 'room-video.mp4', { type: 'video/mp4' }));
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
    if (selected) showReview(selected);
  }

  async function startCapture() {
    if (sessionStorage.getItem(cameraPermissionKey) === 'true') {
      await openCamera();
      return;
    }
    try {
      const [camera, microphone] = await Promise.all([
        navigator.permissions.query({ name: 'camera' as PermissionName }),
        navigator.permissions.query({ name: 'microphone' as PermissionName }),
      ]);
      if (camera.state === 'granted' && microphone.state === 'granted') {
        await openCamera();
        return;
      }
    } catch { /* Some browsers do not expose camera permissions through this API. */ }
    setPermissionOpen(true);
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
        <div className="capture-tip"><img className="capture-tip-icon" src={shoesIcon} alt="" /><div><strong>이동하며 방 전체를 촬영해 주세요</strong><p>천천히 이동하면서 방 구석구석 빠짐없이 담기도록<br/>충분히 촬영해 주세요.</p></div></div>
      </div>
      {error && <p role="alert">{error}</p>}
      <div className="capture-prep-actions"><Button onClick={() => void startCapture()}>촬영 시작하기</Button><Button variant="secondary" onClick={() => fileRef.current?.click()}>영상 업로드하기</Button></div>
      <input ref={fileRef} type="file" accept=".mp4,.mov,video/mp4,video/quicktime" onChange={chooseFile} hidden aria-label="MP4 또는 MOV 영상 선택" />
      {permissionOpen && <div className="capture-permission-backdrop" onClick={() => setPermissionOpen(false)}>
        <div className="capture-permission" role="dialog" aria-modal="true" aria-label="카메라 사용 안내" onClick={event => event.stopPropagation()}>
          <img src={permissionCameraIcon} width="40" height="40" alt="" />
          <h2>방을 촬영하려면<br />카메라 사용이 필요해요</h2>
          <p>이삿짐 목록을 만들기 위해 카메라를 사용해요.<br />다음 브라우저 안내에서 ‘허용’을 눌러주세요.</p>
          <Button onClick={() => void openCamera()}>카메라 사용 허용하기</Button>
        </div>
      </div>}
    </div>}
    {screen === 'camera' && <div className="capture-camera">
      {preview ? <img className="capture-camera-preview" src="/capture-preview-room.png" alt="촬영 화면 미리보기" /> : <video ref={videoRef} autoPlay muted playsInline aria-label="카메라 미리보기" />}
      <div className={`capture-camera-guide${preview && previewState === 'tilted' ? ' capture-camera-guide--tilted' : ''}`} aria-hidden="true"><img src={levelEllipse} alt="" /><img src={preview && previewState === 'tilted' ? tiltedLevelLine : levelLine} alt="" /></div>
      <p className="capture-camera-hint">{recording ? <>화면 안에 가구가 담겨야 정확도가 높아요 <img className="capture-double-arrow" src={doubleArrowIcon} alt="" /></> : '이동하며 방 전체를 충분히 촬영해주세요'}</p>
      {error && <p role="alert" className="capture-camera-error">{error}</p>}
      <div className="capture-camera-controls"><button type="button" className={recording ? 'capture-shutter capture-shutter--recording' : 'capture-shutter'} onClick={() => { if (preview) { if (recording) setScreen('review'); setRecording(!recording); } else if (recording) recorderRef.current?.stop(); else start(); }} aria-label={recording ? '촬영 종료' : '촬영 시작'} /></div>
    </div>}
    {screen === 'review' && <div className="capture-review">
      <BrandHeader />
      <div className="capture-review-content">
        <h1>촬영한 영상을 확인해주세요</h1>
        <p>가구와 가려진 공간이 빠짐없이 담겼는지 확인해주세요.</p>
        {reviewUrl ? <video src={reviewUrl} controls playsInline preload="metadata" aria-label="촬영한 영상" /> : preview ? <img className="capture-review-placeholder" src="/capture-preview-room.png" alt="촬영 영상 미리보기 예시" /> : <div className="capture-review-placeholder" role="img" aria-label="영상 미리보기" />}
        <div className="capture-review-note">촬영한 영상을 확인하고 영상을 다시 업로드할 수 있어요.</div>
        {error && <p role="alert" className="capture-review-error">{error}</p>}
      </div>
      <div className="capture-review-actions"><Button onClick={() => { if (reviewFile) void upload(reviewFile); else if (preview) { window.history.pushState({}, '', afterMove ? `${routes.analysis}?mode=after&preview=1&state=waiting` : `${routes.moveInfo}?preview=1`); window.dispatchEvent(new PopStateEvent('popstate')); } }}>이 영상으로 분석 시작하기</Button><Button variant="secondary" onClick={() => { setError(''); setReviewFile(null); setReviewUrl(null); if (reviewUrlRef.current) URL.revokeObjectURL(reviewUrlRef.current); reviewUrlRef.current = null; setScreen('prep'); }}>다시 촬영하기</Button></div>
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
