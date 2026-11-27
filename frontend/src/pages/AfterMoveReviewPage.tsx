import { navigate, routes } from '../app/routes';
import { BrandHeader } from '../components/common/BrandHeader';
import { Button } from '../components/common/Button';

/** The backend has after-media upload, but no after-move comparison result contract yet. */
export function AfterMoveReviewPage() {
  return <main className="app-page review-chat-page">
    <BrandHeader />
    <div className="review-gate">
      <p>이사 후 영상 비교 결과 API가 연결되면 물품 검토 결과를 확인할 수 있어요.</p>
      <Button variant="secondary" onClick={() => navigate(routes.history)}>나의 이사 기록으로 돌아가기</Button>
    </div>
  </main>;
}
