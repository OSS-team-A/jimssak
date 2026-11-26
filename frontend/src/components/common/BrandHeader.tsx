import logo from '../../assets/logo.png';
import { navigate, openGuide, routes } from '../../app/routes';

export function BrandHeader({ guideLink = true }: { guideLink?: boolean }) {
  return <header className="brand-header">
    <button className="brand-logo" type="button" aria-label="짐싹 홈" onClick={() => navigate(routes.home)}><img src={logo} alt="짐싹" /></button>
    {guideLink && <button className="header-guide" type="button" onClick={openGuide}>서비스 가이드</button>}
  </header>;
}
