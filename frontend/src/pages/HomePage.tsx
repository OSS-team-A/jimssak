import { Button } from '../components/common/Button';

export function HomePage() {
  return <main className="app-page home-page">
    <header className="home-header"><strong>짐싹</strong></header>
    <div className="home-content">
      <h1>방 한 바퀴 촬영으로,<br />이사 준비를 시작해요</h1>
      <p>이삿짐 목록부터 예상 견적 비교까지,<br />싹싹이가 함께 정리해드려요.</p>
    </div>
    <div className="home-actions">
      <Button disabled>시작하기</Button>
      <Button variant="secondary" disabled>내 이사 기록 보기</Button>
      <small>로그인하면 이사 기록을 저장하고 이어볼 수 있어요.</small>
    </div>
  </main>;
}
