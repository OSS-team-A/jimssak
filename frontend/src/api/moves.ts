import { apiRequest } from './client';
import type { MoveCreated, MoveSnapshot } from '../types/move';

export const createMove = () => apiRequest<MoveCreated>('/api/v1/moves', { method: 'POST' });
export const getMove = (moveId: string, guestToken?: string) =>
  apiRequest<MoveSnapshot>(`/api/v1/moves/${encodeURIComponent(moveId)}`, {
    headers: guestToken ? { 'X-Guest-Token': guestToken } : undefined,
  });
