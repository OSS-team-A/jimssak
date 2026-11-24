import { Button } from '../components/common/Button';
import { BrandHeader } from '../components/common/BrandHeader';
import { navigate, routes } from '../app/routes';
import mascot from '../assets/mascot.png';

export function HomePage() {
  // 퍼블리싱 확인용 회원 상태. 실제 로그인 상태는 API 연동 시 교체한다.
  const memberPreview = new URLSearchParams(window.location.search).get('preview') === 'member';
  return <main className="app-page home-page">
    <BrandHeader />
    <div className="home-content">
      <div className="mascot-window" role="img" aria-label="상자 옆에서 인사하는 짐싹 캐릭터"><img src={mascot} alt="" /></div>
      <h1>방 한 바퀴 촬영으로,<br />이사 준비를 시작해요</h1>
      <p>이삿짐 목록부터 예상 견적 비교까지,<br />싹싹이가 함께 정리해드려요.</p>
    </div>
    <div className="home-actions">
      <Button onClick={() => navigate(routes.capture)}>촬영하고 이삿짐 정리하기</Button>
      <Button variant="secondary" onClick={() => navigate(memberPreview ? routes.history : routes.login)}>{memberPreview ? '내 이사 기록 확인하기' : '로그인 / 회원가입'}</Button>
      {!memberPreview && <small>로그인 없이 촬영할 수 있어요. 로그인하면 이사 기록을 저장하고 이어볼 수 있어요.</small>}
    </div>
  </main>;
}
