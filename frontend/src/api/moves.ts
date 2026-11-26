import { apiRequest } from './client';
import type { MoveCreated, MoveSnapshot, MoveUpdate, AddressCandidate, AnalysisResult, MoveList } from '../types/move';

export const createMove = () => apiRequest<MoveCreated>('/api/v1/moves', { method: 'POST' });
export const getMove = (moveId: string, guestToken?: string) =>
  apiRequest<MoveSnapshot>(`/api/v1/moves/${encodeURIComponent(moveId)}`, {
    headers: guestToken ? { 'X-Guest-Token': guestToken } : undefined,
  });
export const getAnalysis = (moveId: string, guestToken?: string) =>
  apiRequest<AnalysisResult>(`/api/v1/moves/${encodeURIComponent(moveId)}/analysis`, {
    headers: guestToken ? { 'X-Guest-Token': guestToken } : undefined,
  });
export const listMoves = (limit = 100, offset = 0) =>
  apiRequest<MoveList>(`/api/v1/moves?limit=${limit}&offset=${offset}`);
export const updateMove = (moveId: string, guestToken: string, update: MoveUpdate) =>
  apiRequest<MoveSnapshot>(`/api/v1/moves/${encodeURIComponent(moveId)}`, {
    method: 'PATCH', headers: { 'Content-Type': 'application/json', ...(guestToken ? { 'X-Guest-Token': guestToken } : {}) },
    body: JSON.stringify(update),
  });
export const searchAddresses = (keyword: string) =>
  apiRequest<{ results: AddressCandidate[]; total: number }>(`/api/v1/addresses?keyword=${encodeURIComponent(keyword)}`);

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
    headers: guestToken ? { 'X-Guest-Token': guestToken } : undefined,
    body: form,
  });
};

export const uploadAfterMoveVideo = (moveId: string, guestToken: string, file: File) => {
  const form = new FormData();
  form.append('file', file);
  return apiRequest<UploadedVideo>(`/api/v1/moves/${encodeURIComponent(moveId)}/after-media`, {
    method: 'POST',
    headers: guestToken ? { 'X-Guest-Token': guestToken } : undefined,
    body: form,
  });
};
