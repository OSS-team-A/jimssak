import { useEffect, useState } from 'react';
import { getAnalysis } from '../api/moves';
import type { AnalysisItem, AnalysisResult } from '../types/move';

type MoveSession = { move_id: string; guest_token: string };
export type ReviewDraft = {
  analysisId: string;
  step: number;
  quantities: Record<string, number>;
  added: { id: string; label: string; quantity: number }[];
  notes: { step: number; text: string }[];
};

function activeMove(): MoveSession | null {
  try {
    const saved = sessionStorage.getItem('jimssak-member-move') || sessionStorage.getItem('jimssak-guest-move');
    return saved ? JSON.parse(saved) as MoveSession : null;
  } catch { return null; }
}

const emptyDraft = (analysisId: string): ReviewDraft => ({ analysisId, step: 0, quantities: {}, added: [], notes: [] });
const storageKey = (moveId: string) => `jimssak-review-${moveId}`;

function loadDraft(moveId: string, analysisId: string): ReviewDraft {
  try {
    const saved = sessionStorage.getItem(storageKey(moveId));
    const draft = saved ? JSON.parse(saved) as ReviewDraft : null;
    if (draft?.analysisId === analysisId) return draft;
  } catch { /* Damaged browser data is discarded. */ }
  return emptyDraft(analysisId);
}

export function useReview() {
  const [move] = useState(activeMove);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [draft, setDraft] = useState<ReviewDraft | null>(null);
  const [loading, setLoading] = useState(!!move);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!move) return;
    let active = true;
    getAnalysis(move.move_id, move.guest_token).then(result => {
      if (!active) return;
      setAnalysis(result);
      if (result.status === 'completed' && result.analysis_id) setDraft(loadDraft(move.move_id, result.analysis_id));
      setError('');
    }).catch(() => { if (active) setError('분석 결과를 불러오지 못했어요. 잠시 후 다시 시도해주세요.'); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [move]);

  function updateDraft(change: (current: ReviewDraft) => ReviewDraft) {
    if (!move || !draft) return;
    const next = change(draft);
    sessionStorage.setItem(storageKey(move.move_id), JSON.stringify(next));
    setDraft(next);
  }

  const items = analysis?.items.map(item => ({ ...item, quantity: draft?.quantities[item.item_id] ?? item.quantity })) || [];
  const added = draft?.added || [];
  const total = items.reduce((sum, item) => sum + item.quantity, 0) + added.reduce((sum, item) => sum + item.quantity, 0);
  return { move, analysis, draft, loading, error, items, added, total, updateDraft };
}

export function dimensions(item: AnalysisItem): string {
  const { width, depth, height } = item.dimensions_cm;
  if (width == null || depth == null || height == null) return '확인 필요';
  return `${width} × ${depth} × ${height}`;
}
