import { useEffect, useRef, useState } from 'react';
import { navigate, routes } from '../app/routes';
import { BrandHeader } from '../components/common/BrandHeader';
import { Button } from '../components/common/Button';

export function CapturePage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const recordingUrlRef = useRef<string | null>(null);
  const [recordingUrl, setRecordingUrl] = useState<string | null>(null);
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    let mounted = true;
    async function openCamera() {
      try {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error('이 브라우저에서는 카메라를 사용할 수 없어요. HTTPS 연결과 브라우저 권한을 확인해주세요.');
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: true });
        if (!mounted) { stream.getTracks().forEach(track => track.stop()); return; }
        streamRef.current = stream;
        if (videoRef.current) videoRef.current.srcObject = stream;
      } catch (cause) {
        if (mounted) setError(cause instanceof Error && cause.message.startsWith('이 브라우저') ? cause.message : '카메라와 마이크 권한을 허용해야 촬영할 수 있어요.');
      }
    }
    void openCamera();
    return () => {
      mounted = false;
      if (recorderRef.current?.state === 'recording') recorderRef.current.stop();
      streamRef.current?.getTracks().forEach(track => track.stop());
      if (recordingUrlRef.current) URL.revokeObjectURL(recordingUrlRef.current);
    };
  }, []);

  function start() {
    if (!streamRef.current || !window.MediaRecorder) {
      setError('이 브라우저에서는 영상 녹화를 지원하지 않아요.');
      return;
    }
    try {
      if (recordingUrlRef.current) URL.revokeObjectURL(recordingUrlRef.current);
      recordingUrlRef.current = null;
      setRecordingUrl(null);
      chunksRef.current = [];
      const recorder = new MediaRecorder(streamRef.current);
      recorderRef.current = recorder;
      recorder.ondataavailable = event => { if (event.data.size) chunksRef.current.push(event.data); };
      recorder.onstop = () => {
        const url = URL.createObjectURL(new Blob(chunksRef.current, { type: recorder.mimeType || 'video/webm' }));
        recordingUrlRef.current = url;
        setRecordingUrl(url);
        setRecording(false);
      };
      recorder.start();
      setError('');
      setRecording(true);
    } catch {
      setError('녹화를 시작할 수 없어요. 브라우저 권한을 확인해주세요.');
    }
  }

  return <main className="app-page capture-page">
    <BrandHeader />
    <h1>방을 천천히 한 바퀴 촬영해주세요</h1>
    <p>비회원도 촬영할 수 있어요. 영상은 이 화면에서만 미리 볼 수 있으며 서버에 저장되지 않아요.</p>
    {recordingUrl
      ? <video className="capture-preview" src={recordingUrl} controls playsInline aria-label="촬영한 영상" />
      : <video ref={videoRef} className="capture-preview" autoPlay muted playsInline aria-label="카메라 미리보기" />}
    <p className="capture-notice">영상 분석과 이삿짐 목록 생성은 아직 연결되지 않았어요. 촬영 후 영상 확인까지만 가능합니다.</p>
    {error && <p role="alert">{error}</p>}
    <div className="capture-actions">
      {recording
        ? <Button onClick={() => recorderRef.current?.stop()}>촬영 마치기</Button>
        : <Button onClick={start} disabled={!streamRef.current}>{recordingUrl ? '다시 촬영하기' : '촬영 시작'}</Button>}
      <Button variant="secondary" onClick={() => navigate(routes.home)}>홈으로 돌아가기</Button>
    </div>
  </main>;
}
