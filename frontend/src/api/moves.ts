import { apiRequest } from './client';
import { accessToken } from './auth';
import type { MoveCreated, MoveSnapshot, MoveUpdate, AddressCandidate, AnalysisResult, MoveList, MoveSummary, Inventory, InventoryItem, InventoryItemCreate, InventoryItemPatch } from '../types/move';

export const createMove = () => apiRequest<MoveCreated>('/api/v1/moves', { method: 'POST' });
export const getMove = (moveId: string, guestToken?: string) =>
  apiRequest<MoveSnapshot>(`/api/v1/moves/${encodeURIComponent(moveId)}`, {
    headers: guestToken ? { 'X-Guest-Token': guestToken } : undefined,
  });
export const getAnalysis = (moveId: string, guestToken?: string) =>
  apiRequest<AnalysisResult>(`/api/v1/moves/${encodeURIComponent(moveId)}/analysis`, {
    headers: guestToken ? { 'X-Guest-Token': guestToken } : undefined,
  });
export const getAnalysisPhoto = (path: string, guestToken?: string) => {
  if (!path.startsWith('/api/v1/moves/') || !path.includes('/analysis/photos/')) {
    throw new Error('지원하지 않는 분석 사진 주소예요.');
  }
  return apiRequest<Blob>(path, {
    headers: guestToken ? { 'X-Guest-Token': guestToken } : undefined,
    responseType: 'blob',
  });
};
export const listMoves = (limit = 100, offset = 0) =>
  apiRequest<MoveList>(`/api/v1/moves?limit=${limit}&offset=${offset}`);
export async function listAllMoves() {
  const all: MoveSummary[] = [];
  while (true) {
    const page = await listMoves(100, all.length);
    all.push(...page.moves);
    if (all.length >= page.total || page.moves.length === 0) return all;
  }
}
export const claimMove = (moveId: string, guestToken: string) =>
  apiRequest<MoveSnapshot>(`/api/v1/moves/${encodeURIComponent(moveId)}/claim`, {
    method: 'POST', headers: { 'X-Guest-Token': guestToken, Authorization: `Bearer ${accessToken()}` },
  });
export const getInventory = (moveId: string, guestToken?: string) =>
  apiRequest<Inventory>(`/api/v1/moves/${encodeURIComponent(moveId)}/inventory`, {
    headers: guestToken ? { 'X-Guest-Token': guestToken } : undefined,
  });
export const addInventoryItem = (moveId: string, guestToken: string, input: InventoryItemCreate) =>
  apiRequest<InventoryItem>(`/api/v1/moves/${encodeURIComponent(moveId)}/inventory/items`, {
    method: 'POST', headers: { 'Content-Type': 'application/json', ...(guestToken ? { 'X-Guest-Token': guestToken } : {}) }, body: JSON.stringify(input),
  });
export const patchInventoryItem = (moveId: string, guestToken: string, itemId: string, input: InventoryItemPatch) =>
  apiRequest<InventoryItem>(`/api/v1/moves/${encodeURIComponent(moveId)}/inventory/items/${encodeURIComponent(itemId)}`, {
    method: 'PATCH', headers: { 'Content-Type': 'application/json', ...(guestToken ? { 'X-Guest-Token': guestToken } : {}) }, body: JSON.stringify(input),
  });
export const deleteInventoryItem = (moveId: string, guestToken: string, itemId: string) =>
  apiRequest<void>(`/api/v1/moves/${encodeURIComponent(moveId)}/inventory/items/${encodeURIComponent(itemId)}`, {
    method: 'DELETE', headers: guestToken ? { 'X-Guest-Token': guestToken } : undefined,
  });
export const confirmInventory = (moveId: string, guestToken: string) =>
  apiRequest<void>(`/api/v1/moves/${encodeURIComponent(moveId)}/inventory/confirm`, {
    method: 'POST', headers: guestToken ? { 'X-Guest-Token': guestToken } : undefined,
  });
export const unconfirmInventory = (moveId: string, guestToken: string) =>
  apiRequest<void>(`/api/v1/moves/${encodeURIComponent(moveId)}/inventory/confirm`, {
    method: 'DELETE', headers: guestToken ? { 'X-Guest-Token': guestToken } : undefined,
  });
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
