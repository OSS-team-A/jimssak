import { Button } from '../components/common/Button';
import { BrandHeader } from '../components/common/BrandHeader';
import { navigate, routes } from '../app/routes';
import mascot from '../assets/mascot.png';
import { hasSession } from '../api/auth';

export function HomePage() {
  const isMember = hasSession();
  return <main className="app-page home-page">
    <BrandHeader />
    <div className="home-content">
      <div className="mascot-window" role="img" aria-label="상자 옆에서 인사하는 짐싹 캐릭터"><img src={mascot} alt="" /></div>
      <h1>방 한 바퀴 촬영으로,<br />이사 준비를 시작해요</h1>
      <p>이삿짐 목록부터 예상 견적 비교까지,<br />싹싹이가 함께 정리해드려요.</p>
    </div>
    <div className="home-actions">
      <Button onClick={() => navigate(routes.capture)}>촬영하고 이삿짐 정리하기</Button>
      <Button variant="secondary" onClick={() => navigate(isMember ? routes.history : routes.login)}>{isMember ? '내 이사 기록 확인하기' : '로그인 / 회원가입'}</Button>
      {!isMember && <small>로그인하면 내 이사 기록을 확인할 수 있어요.</small>}
    </div>
  </main>;
}
