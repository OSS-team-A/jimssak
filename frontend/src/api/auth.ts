import { apiRequest } from './client';

export type Session = {
  access_token: string;
  refresh_token: string;
  token_type: 'bearer';
  expires_in: number;
  refresh_expires_in: number;
};

const sessionKey = 'jimssak-session';
export const hasSession = () => Boolean(sessionStorage.getItem(sessionKey));
export const accessToken = () => {
  try { return (JSON.parse(sessionStorage.getItem(sessionKey) || 'null') as Session | null)?.access_token; }
  catch { return undefined; }
};
export const saveSession = (session: Session) => sessionStorage.setItem(sessionKey, JSON.stringify(session));
export const clearSession = () => sessionStorage.removeItem(sessionKey);
export const signup = (email: string, password: string) =>
  apiRequest<{ id: number; email: string }>('/api/v1/auth/signup', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
export const login = (email: string, password: string) =>
  apiRequest<Session>('/api/v1/auth/login', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
