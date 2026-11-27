import { accessToken, clearSession, refreshToken, saveSession, type Session } from './auth';

const baseUrl = (import.meta.env.VITE_API_BASE_URL || 'https://api.kyooo.shop').replace(/\/$/, '');

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

type ApiInit = RequestInit & { skipAuthRefresh?: boolean; responseType?: 'blob' };
let refreshing: Promise<Session> | null = null;

export async function apiRequest<T>(path: string, init?: ApiInit): Promise<T> {
  const token = accessToken();
  const headers = new Headers(init?.headers);
  if (token && !headers.has('X-Guest-Token') && !headers.has('Authorization')) headers.set('Authorization', `Bearer ${token}`);
  const response = await fetch(`${baseUrl}${path}`, { ...init, headers });
  if (response.status === 401 && token && (!headers.has('X-Guest-Token') || headers.has('Authorization')) && !init?.skipAuthRefresh && refreshToken()) {
    refreshing ??= fetch(`${baseUrl}/api/v1/auth/refresh`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: refreshToken() }),
    }).then(async result => {
      if (!result.ok) throw new ApiError(result.status, '로그인이 만료됐어요. 다시 로그인해주세요.');
      const session = await result.json() as Session;
      saveSession(session);
      return session;
    }).catch(error => { clearSession(); throw error; }).finally(() => { refreshing = null; });
    const renewed = await refreshing;
    headers.set('Authorization', `Bearer ${renewed.access_token}`);
    return apiRequest<T>(path, { ...init, headers, skipAuthRefresh: true });
  }
  if (!response.ok) {
    if (response.status === 401 && headers.has('Authorization')) clearSession();
    let detail = `API 응답 오류 (${response.status})`;
    try { const data = await response.json() as { detail?: unknown }; if (typeof data.detail === 'string') detail = data.detail; }
    catch { /* Non-JSON errors keep the HTTP status. */ }
    throw new ApiError(response.status, detail);
  }
  if (response.status === 204) return undefined as T;
  if (init?.responseType === 'blob') return response.blob() as Promise<T>;
  return response.json() as Promise<T>;
}
