import { useState, type FormEvent } from 'react';
import { navigate, routes } from '../app/routes';
import { BrandHeader } from '../components/common/BrandHeader';
import { Button } from '../components/common/Button';
import { login, saveSession, signup } from '../api/auth';
import { ApiError } from '../api/client';
import { claimMove } from '../api/moves';

export function AuthPage({ mode }: { mode: 'login' | 'signup' }) {
  const isSignup = mode === 'signup';
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmation, setConfirmation] = useState('');
  const [message, setMessage] = useState('');
  const [submitting, setSubmitting] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (isSignup && password !== confirmation) {
      setMessage('비밀번호가 일치하지 않아요. 다시 확인해주세요.');
      return;
    }
    setSubmitting(true);
    setMessage('');
    try {
      if (isSignup) await signup(email, password);
      saveSession(await login(email, password));
      const guest = sessionStorage.getItem('jimssak-guest-move');
      if (guest) {
        const pending = JSON.parse(guest) as { move_id?: string; guest_token?: string };
        if (pending.move_id && pending.guest_token) {
          await claimMove(pending.move_id, pending.guest_token);
          sessionStorage.setItem('jimssak-member-move', JSON.stringify({ move_id: pending.move_id, guest_token: '' }));
          sessionStorage.removeItem('jimssak-guest-move');
        }
      }
      navigate(routes.history);
    } catch (cause) {
      setMessage(cause instanceof ApiError
        ? cause.status === 409 ? '이미 가입한 이메일이에요.' : cause.status === 401 ? '이메일 또는 비밀번호를 확인해주세요.' : cause.message
        : '서버에 연결하지 못했어요. 연결 상태를 확인해주세요.');
    } finally { setSubmitting(false); }
  }

  return <main className="app-page auth-page">
    <BrandHeader guideLink={isSignup} />
    <form className="auth-form" onSubmit={submit}>
      <div className="auth-intro">
        <h1>{isSignup ? <>짐싹과 함께<br />이사 준비를 시작해요</> : '다시 만나 반가워요'}</h1>
        <p>{isSignup ? '사용할 이메일과 비밀번호를 입력해주세요.' : '로그인하고 이사 기록을 이어보세요.'}</p>
      </div>
      <div className="auth-fields">
        {isSignup && <label>이름<input autoComplete="name" placeholder="이름을 입력해주세요" value={name} onChange={e => setName(e.target.value)} required /></label>}
        <label>이메일<input type="email" autoComplete="email" inputMode="email" placeholder="example@email.com" value={email} onChange={e => setEmail(e.target.value)} required /></label>
        <label>비밀번호<input type="password" autoComplete={isSignup ? 'new-password' : 'current-password'} placeholder="비밀번호를 입력해주세요" value={password} onChange={e => setPassword(e.target.value)} required /></label>
        {isSignup && <label>비밀번호 확인<input type="password" autoComplete="new-password" placeholder="비밀번호를 한 번 더 입력해주세요" value={confirmation} onChange={e => setConfirmation(e.target.value)} required /></label>}
      </div>
      {message && <p className="auth-message" role="status">{message}</p>}
      <div className="auth-actions"><Button type="submit" disabled={submitting}>{submitting ? '처리 중...' : isSignup ? '가입하기' : '로그인'}</Button>
        {!isSignup && <p>처음이신가요? <button type="button" onClick={() => navigate(routes.signup)}>회원가입</button></p>}
        {isSignup && <p><button type="button" onClick={() => navigate(routes.login)}>이미 계정이 있나요? 로그인</button></p>}
      </div>
    </form>
  </main>;
}
