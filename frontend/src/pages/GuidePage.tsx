import { useState } from 'react';
import { closeGuide } from '../app/routes';
import mascot from '../assets/mascot.png';
import guideBox from '../assets/guide-box.svg';
import guideHouse from '../assets/guide-house.svg';
import guideDocument from '../assets/guide-document.svg';
import guideShield from '../assets/guide-shield.svg';
import stepCamera from '../assets/step-camera.svg';
import stepPin from '../assets/step-pin.svg';
import stepBox from '../assets/step-box.svg';
import stepDocument from '../assets/step-document.svg';
import stepShield from '../assets/step-shield.svg';
import chevronDown from '../assets/chevron-down.svg';
import chevronUp from '../assets/chevron-up.svg';

type Tab = 'intro' | 'how' | 'faq';
const tabs: { id: Tab; label: string }[] = [
  { id: 'intro', label: '서비스 소개' }, { id: 'how', label: '이용 방법' }, { id: 'faq', label: '자주 묻는 질문' },
];
const features = [
  { icon: guideBox, title: '이삿짐 목록을 간편하게', description: '방을 촬영하면 AI가 가구와 가전을 찾아 종류·수량·예상 크기를 정리해요. 빠지거나 잘못 정리된 내용은 직접 수정할 수 있어요.' },
  { icon: guideHouse, title: '이사 유형별 예상 비용 비교', description: '확인한 이삿짐과 이사 조건을 바탕으로 일반·반포장·포장 이사의 예상 비용과 포함 작업을 살펴봐요. 실제 견적은 업체에 문의해 확인해주세요.' },
  { icon: guideDocument, title: '이삿짐 명세서를 간편하게', description: '확인한 물품과 이사 조건을 PDF 명세서로 받아 여러 업체에 전달할 수 있어요. 같은 이삿짐 목록을 매번 다시 작성하지 않아도 돼요.' },
  { icon: guideShield, title: '이사 후에도 한 번 더 확인', description: '이사 전후 영상을 비교해 상태 변화가 의심되거나 확인이 필요한 물품을 함께 살펴봐요.' },
];
const steps = [
  { icon: stepCamera, title: '방을 촬영해주세요', description: '조명을 켜고, 이사할 물품이 잘 보이도록 천천히 촬영해주세요. 다른 공간도 추가로 찍을 수 있어요.' },
  { icon: stepPin, title: '이사 정보를 알려주세요', description: '영상이 분석되는 동안 출발지와 도착지, 층수, 이사 날짜와 시간을 입력해주세요.' },
  { icon: stepBox, title: '이삿짐 목록을 확인해주세요', description: '물품과 수량을 확인하고, 수정할 내용은 목록이나 채팅으로 알려주세요. 두고 갈 물품도 제외할 수 있어요.' },
  { icon: stepDocument, title: '예상 비용을 비교하고 명세서를 받아요', description: '이사 유형별 예상 비용과 포함 작업을 살펴본 뒤, 이삿짐 명세서 PDF를 받아 업체에 견적을 문의해보세요.' },
  { icon: stepShield, title: '이사 후 상태를 확인해요', description: '저장한 이사 기록에서 다시 촬영하고, 이사 전과 달라진 부분을 확인해요.' },
];
const faqs = [
  { question: '회원가입 없이도 이용할 수 있나요?', answer: '바로 촬영하고 목록을 확인할 수 있어요. 이사 기록을 저장하고 나중에 이어보려면 로그인해주세요.' },
  { question: 'AI가 정리한 목록이 실제와 다르면 어떻게 하나요?', answer: '물품을 추가하거나 수량·크기를 수정할 수 있어요. 채팅으로 수정할 내용을 알려줘도 돼요.' },
  { question: '물품 크기와 견적은 정확한가요?', answer: '영상으로 분석한 크기와 비용은 추정값이에요. 실제 크기를 알고 있다면 수정해주시고, 최종 비용은 업체와 확인해주세요.' },
  { question: '이사 후 영상에 물품이 보이지 않으면 분실인가요?', answer: '가려져 있거나 촬영 범위에서 빠졌을 수 있어요. 미확인 물품을 추가로 촬영하거나 직접 확인해주세요.' },
  { question: '짐싹에서 실제 업체 견적도 받을 수 있나요?', answer: '짐싹은 이사 유형별 예상 비용을 보여드려요. 실제 업체 견적은 이삿짐 명세서 PDF를 전달해 직접 문의해주세요.' },
];

export function GuidePage() {
  const [tab, setTab] = useState<Tab>('intro');
  const [openFaq, setOpenFaq] = useState<number | null>(0);

  return <main className="app-page guide-page">
    <header className="sub-header"><strong>서비스 가이드</strong><button type="button" aria-label="닫기" onClick={closeGuide}>×</button></header>
    <nav className="guide-tabs" aria-label="서비스 가이드">
      {tabs.map(item => <button key={item.id} type="button" className={tab === item.id ? 'active' : ''} aria-current={tab === item.id ? 'page' : undefined} onClick={() => setTab(item.id)}>{item.label}</button>)}
    </nav>
    {tab === 'intro' && <section className="guide-body intro-body">
      <span className="eyebrow">이사 준비, 짐싹과 함께</span>
      <h1>방 한 바퀴 촬영으로,<br />이사 준비를 시작해요</h1>
      <p>어떤 짐이 있는지 정리하고 견적을 비교하는 일까지,<br />짐싹이 함께 도와드려요.</p>
      <div className="guide-mascot"><div className="mascot-window" role="img" aria-label="상자 옆에 선 짐싹 캐릭터"><img src={mascot} alt="" /></div></div>
      <div className="feature-list">{features.map(feature => <article className="feature" key={feature.title}>
        <img src={feature.icon} alt="" /><h2>{feature.title}</h2><p>{feature.description}</p>
      </article>)}</div>
    </section>}
    {tab === 'how' && <section className="guide-body how-body"><span className="eyebrow">이용 방법</span><h1>짐싹, 이렇게 이용해요</h1>
      <div className="step-list">{steps.map((step, index) => <article className="step" key={step.title}>
        <img src={step.icon} alt="" /><div><span className="eyebrow">STEP {String(index + 1).padStart(2, '0')}</span><h2>{step.title}</h2><p>{step.description}</p></div>
      </article>)}</div>
    </section>}
    {tab === 'faq' && <section className="guide-body faq-body"><h1>자주 묻는 질문</h1>
      <div className="faq-list">{faqs.map((faq, index) => <button type="button" className="faq-item" aria-expanded={openFaq === index} key={faq.question} onClick={() => setOpenFaq(openFaq === index ? null : index)}>
        <span className="faq-question">{faq.question}<img src={openFaq === index ? chevronUp : chevronDown} alt="" /></span>
        {openFaq === index && <span className="faq-answer">{faq.answer}</span>}
      </button>)}</div>
    </section>}
  </main>;
}
