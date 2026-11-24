const baseUrl = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

export async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const token = accessToken();
  const headers = new Headers(init?.headers);
  if (token && !headers.has('X-Guest-Token') && !headers.has('Authorization')) headers.set('Authorization', `Bearer ${token}`);
  const response = await fetch(`${baseUrl}${path}`, { ...init, headers });
  if (!response.ok) throw new ApiError(response.status, `API 응답 오류 (${response.status})`);
  return response.json() as Promise<T>;
}
import { accessToken } from './auth';
