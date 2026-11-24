import { apiRequest } from './client';
import type { MoveCreated, MoveSnapshot } from '../types/move';

export const createMove = () => apiRequest<MoveCreated>('/api/v1/moves', { method: 'POST' });
export const getMove = (moveId: string, guestToken?: string) =>
  apiRequest<MoveSnapshot>(`/api/v1/moves/${encodeURIComponent(moveId)}`, {
    headers: guestToken ? { 'X-Guest-Token': guestToken } : undefined,
  });

export type UploadedVideo = {
  media_id: string;
  filename: string;
  content_type: 'video/mp4' | 'video/quicktime';
  size_bytes: number;
  replaced: boolean;
};

export const uploadMoveVideo = (moveId: string, guestToken: string, file: File) => {
  const form = new FormData();
  form.append('file', file);
  return apiRequest<UploadedVideo>(`/api/v1/moves/${encodeURIComponent(moveId)}/media`, {
    method: 'POST',
    headers: { 'X-Guest-Token': guestToken },
    body: form,
  });
};
