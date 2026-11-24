import { useState } from 'react';
import { apiRequest } from '../api/client';

type Analysis = {
  mode: 'demo' | 'model';
  frame: { video_id: string; width_px: number; height_px: number };
  detections: Array<{
    track_id: string; label: string; confidence: number;
    dimensions: { width_cm: number; height_cm: number } | null;
    measurement_status: 'needs_calibration' | 'estimated';
  }>;
};

export function AnalysisDemoPage() {
  const [result, setResult] = useState<Analysis | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  async function analyze() {
    setBusy(true);
    setError('');
    setResult(null);
    try {
      setResult(await apiRequest<Analysis>('/api/v1/analysis/demo', {
        method: 'POST', signal: AbortSignal.timeout(10000),
      }));
    } catch (err) {
      setError(err instanceof Error ? err.message : '분석 요청에 실패했습니다.');
    } finally {
      setBusy(false);
    }
  }

  return <main className="demo-page">
    <span className="badge">JIMSSAK · 개발용 데모</span>
    <h1>이사 준비를 싹,<br/><span>짐싹.</span></h1>
    <p>방 영상에서 물품을 찾고, 이사 준비 목록으로 연결합니다.</p>
    <section>
      <h2>물품 분석 연결 확인</h2>
      <p>버튼을 누르면 FastAPI와 AI 데모 파이프라인을 호출합니다.
        현재 결과는 고정 샘플이며 실제 영상 인식은 아직 제공하지 않습니다.</p>
      <button onClick={analyze} disabled={busy}>{busy ? '불러오는 중…' : '데모 분석 실행'}</button>
      <p className="hint">크기 추정에는 카메라 보정과 미터 단위 깊이 정보가 필요합니다.</p>
      <div aria-live="polite" aria-busy={busy}>
        {error && <p role="alert">{error} 백엔드 실행과 환경변수를 확인해 주세요.</p>}
        {result && <>
          <h3>샘플 결과 · {result.detections.length}개</h3>
          <p>{result.frame.video_id} · {result.frame.width_px} × {result.frame.height_px} px</p>
          <ul>{result.detections.map(item => <li key={item.track_id}>
            <strong>{item.label}</strong> · 신뢰도 {Math.round(item.confidence * 100)}%
            <span className="detail">{item.dimensions
              ? `${item.dimensions.width_cm.toFixed(1)} × ${item.dimensions.height_cm.toFixed(1)} cm (추정)`
              : '크기 미확인 · 보정 정보 필요'}</span>
          </li>)}</ul>
        </>}
      </div>
    </section>
    <footer>영상 업로드 · 실제 탐지 및 추적 · 이사 전후 비교는 다음 구현 단계입니다.</footer>
  </main>;
}
