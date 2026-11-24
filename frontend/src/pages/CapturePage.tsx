import { useEffect, useRef, useState, type ChangeEvent } from 'react';
import { createMove, uploadMoveVideo } from '../api/moves';
import { ApiError } from '../api/client';
import { navigate, routes } from '../app/routes';
import { BrandHeader } from '../components/common/BrandHeader';
import { Button } from '../components/common/Button';

const acceptedVideo = (file: File) => /\.(mp4|mov)$/i.test(file.name)
  && ['video/mp4', 'video/quicktime', 'application/octet-stream', ''].includes(file.type);

export function CapturePage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const previewUrlRef = useRef<string | null>(null);
  const moveRef = useRef<{ move_id: string; guest_token: string } | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [recording, setRecording] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploaded, setUploaded] = useState(false);
  const [error, setError] = useState('');
  const [cameraReady, setCameraReady] = useState(false);
  const mp4Recording = typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported('video/mp4');

  useEffect(() => {
    let mounted = true;
    async function openCamera() {
      try {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error('HTTPS 연결과 브라우저 카메라 권한을 확인해주세요.');
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: true });
        if (!mounted) { stream.getTracks().forEach(track => track.stop()); return; }
        streamRef.current = stream;
        if (videoRef.current) videoRef.current.srcObject = stream;
        setCameraReady(true);
      } catch {
        if (mounted) setError('카메라와 마이크를 사용할 수 없어요. 권한을 허용하거나 MP4·MOV 파일을 선택해주세요.');
      }
    }
    void openCamera();
    return () => {
      mounted = false;
      if (recorderRef.current?.state === 'recording') recorderRef.current.stop();
      streamRef.current?.getTracks().forEach(track => track.stop());
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    };
  }, []);

  useEffect(() => {
    if (!previewUrl && videoRef.current && streamRef.current) videoRef.current.srcObject = streamRef.current;
  }, [previewUrl]);

  function showFile(nextFile: File) {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    const url = URL.createObjectURL(nextFile);
    previewUrlRef.current = url;
    setPreviewUrl(url);
    setFile(nextFile);
    setUploaded(false);
    setError('');
  }

  function start() {
    if (!streamRef.current || !mp4Recording) {
      setError('이 브라우저는 MP4 녹화를 지원하지 않아요. MP4·MOV 파일을 선택해주세요.');
      return;
    }
    try {
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
      setPreviewUrl(null);
      setFile(null);
      setUploaded(false);
      chunksRef.current = [];
      const recorder = new MediaRecorder(streamRef.current, { mimeType: 'video/mp4' });
      recorderRef.current = recorder;
      recorder.ondataavailable = event => { if (event.data.size) chunksRef.current.push(event.data); };
      recorder.onstop = () => {
        if (!chunksRef.current.length) {
          setError('촬영된 영상이 없어요. 다시 시도해주세요.');
          setRecording(false);
          return;
        }
        showFile(new File(chunksRef.current, 'room-video.mp4', { type: 'video/mp4' }));
        setRecording(false);
      };
      recorder.start();
      setRecording(true);
      setError('');
    } catch {
      setError('녹화를 시작할 수 없어요. MP4·MOV 파일을 선택해주세요.');
    }
  }

  function chooseFile(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0];
    if (!selected) return;
    if (!acceptedVideo(selected)) {
      setError('MP4 또는 MOV 영상만 올릴 수 있어요.');
      event.target.value = '';
      return;
    }
    showFile(selected);
  }

  async function upload() {
    if (!file || uploading) return;
    setUploading(true);
    setError('');
    try {
      let move = moveRef.current;
      if (!move) {
        const created = await createMove();
        if (!created.guest_token) throw new Error('비회원 촬영 정보를 만들지 못했어요.');
        move = { move_id: created.move_id, guest_token: created.guest_token };
        moveRef.current = move;
      }
      await uploadMoveVideo(move.move_id, move.guest_token, file);
      sessionStorage.setItem('jimssak-guest-move', JSON.stringify(move));
      setUploaded(true);
    } catch (cause) {
      if (cause instanceof ApiError) {
        setError(cause.status === 413 ? '영상 용량이 너무 커요. 더 짧은 영상을 선택해주세요.'
          : cause.status === 415 ? '서버에서 영상 형식을 확인하지 못했어요. MP4·MOV 파일을 다시 선택해주세요.'
          : cause.status === 404 ? '이사 정보가 만료되었어요. 다시 시도해주세요.'
          : '영상 업로드에 실패했어요. 연결 상태를 확인하고 다시 시도해주세요.');
        if (cause.status === 404) moveRef.current = null;
      } else setError('영상 업로드에 실패했어요. 연결 상태를 확인하고 다시 시도해주세요.');
    } finally {
      setUploading(false);
    }
  }

  return <main className="app-page capture-page">
    <BrandHeader />
    <h1>방을 천천히 한 바퀴 촬영해주세요</h1>
    <p>비회원도 영상을 올릴 수 있어요. 촬영한 영상을 확인한 뒤 업로드해주세요.</p>
    {previewUrl
      ? <video className="capture-preview" src={previewUrl} controls playsInline aria-label="촬영한 영상" />
      : <video ref={videoRef} className="capture-preview" autoPlay muted playsInline aria-label="카메라 미리보기" />}
    <p className="capture-notice">영상 분석과 이삿짐 목록·명세서 생성은 아직 연결되지 않았어요. 지금은 영상 업로드까지만 가능합니다.</p>
    {recording && <p className="capture-recording" role="status">● 촬영 중</p>}
    {uploaded && <p className="capture-success" role="status">영상 업로드가 완료됐어요. 분석 기능은 준비 중이에요.</p>}
    {error && <p role="alert">{error}</p>}
    <div className="capture-actions">
      {recording
        ? <Button onClick={() => recorderRef.current?.stop()}>촬영 마치기</Button>
        : <Button onClick={start} disabled={!cameraReady || !mp4Recording || uploading}>{file ? '다시 촬영하기' : '촬영 시작'}</Button>}
      {!recording && <label>또는 기기에서 MP4·MOV 영상 선택
        <input type="file" accept=".mp4,.mov,video/mp4,video/quicktime" onChange={chooseFile} disabled={uploading} />
      </label>}
      {file && !recording && <Button onClick={() => void upload()} disabled={uploading || uploaded}>{uploading ? '업로드 중…' : uploaded ? '업로드 완료' : '영상 업로드하기'}</Button>}
      <Button variant="secondary" onClick={() => navigate(routes.home)} disabled={uploading}>홈으로 돌아가기</Button>
    </div>
  </main>;
}
