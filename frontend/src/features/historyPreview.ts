import type { MoveSummary } from '../types/move';

export const historyPreviewMove = (stage: MoveSummary['stage']): MoveSummary => ({
  move_id: '__history_preview__',
  created_at: '2026-11-20T00:00:00Z',
  move_date: '2026-11-20',
  start_time: '09:00',
  origin_address: '서울특별시 마포구',
  destination_address: '서울특별시 성동구',
  stage,
});
