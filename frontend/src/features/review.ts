import { useEffect, useState } from 'react';
import { confirmInventory, deleteInventoryItem, getAnalysis, getInventory, patchInventoryItem, unconfirmInventory } from '../api/moves';
import { navigate, type AppRoute } from '../app/routes';
import { ApiError } from '../api/client';
import type { AnalysisItem, AnalysisResult, Inventory } from '../types/move';
import type { ChatTurn } from '../api/chat';

type MoveSession = { move_id: string; guest_token: string };
export type ReviewDraft = {
  analysisId: string;
  step: number;
  quantities: Record<string, number>;
  added: { id: string; label: string; quantity: number }[];
  notes: { step: number; text: string }[];
  answers: { step: number; text: string }[];
  selectedMoveType: 'standard' | 'semi' | 'full' | null;
  chatMessages: (ChatTurn & { step: number; inventory?: Inventory })[];
  pendingProposal: { id: string; summary: string[]; step: number } | null;
  choices: string[];
};

function activeMove(): MoveSession | null {
  try {
    const saved = sessionStorage.getItem('jimssak-member-move') || sessionStorage.getItem('jimssak-guest-move');
    return saved ? JSON.parse(saved) as MoveSession : null;
  } catch { return null; }
}

const emptyDraft = (analysisId: string): ReviewDraft => ({ analysisId, step: 0, quantities: {}, added: [], notes: [], answers: [], selectedMoveType: null, chatMessages: [], pendingProposal: null, choices: [] });
const storageKey = (moveId: string) => `jimssak-review-${moveId}`;
export function navigateReview(path: AppRoute) {
  navigate(path);
}

function loadDraft(moveId: string, analysisId: string): ReviewDraft {
  try {
    const saved = sessionStorage.getItem(storageKey(moveId));
    const draft = saved ? JSON.parse(saved) as ReviewDraft : null;
    if (draft?.analysisId === analysisId) return { ...draft, answers: draft.answers ?? draft.notes.map(note => ({ step: note.step, text: note.text })), selectedMoveType: draft.selectedMoveType ?? null, chatMessages: draft.chatMessages ?? [], pendingProposal: draft.pendingProposal ?? null, choices: draft.choices ?? [] };
  } catch { /* Damaged browser data is discarded. */ }
  return emptyDraft(analysisId);
}

export function useReview() {
  const [move] = useState<MoveSession | null>(activeMove);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [inventory, setInventory] = useState<Inventory | null>(null);
  const [draft, setDraft] = useState<ReviewDraft | null>(null);
  const [loading, setLoading] = useState(!!move);
  const [error, setError] = useState('');
  const [retryKey, setRetryKey] = useState(0);

  useEffect(() => {
    if (!move) return;
    let active = true;
    setAnalysis(null);
    setInventory(null);
    setDraft(null);
    Promise.all([getAnalysis(move.move_id, move.guest_token), getInventory(move.move_id, move.guest_token)]).then(([result, currentInventory]) => {
      if (!active) return;
      setAnalysis(result);
      setInventory(currentInventory);
      if (result.status === 'completed' && result.analysis_id) setDraft(loadDraft(move.move_id, result.analysis_id));
      setError('');
    }).catch(cause => { if (active) setError(cause instanceof ApiError && cause.status === 401 ? '로그인이 만료됐어요. 다시 로그인해주세요.' : '분석 결과를 불러오지 못했어요. 잠시 후 다시 시도해주세요.'); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [move, retryKey]);

  async function changeQuantity(itemId: string, quantity: number) {
    if (!move) return;
    try {
      if (inventory?.status === 'confirmed') await unconfirmInventory(move.move_id, move.guest_token);
      await patchInventoryItem(move.move_id, move.guest_token, itemId, { quantity });
      if (inventory?.status === 'confirmed') await confirmInventory(move.move_id, move.guest_token);
      setInventory(await getInventory(move.move_id, move.guest_token));
      setError('');
    } catch { setError('수량을 저장하지 못했어요. 다시 시도해주세요.'); }
  }

  async function removeItem(itemId: string) {
    if (!move) return;
    try {
      if (inventory?.status === 'confirmed') await unconfirmInventory(move.move_id, move.guest_token);
      await deleteInventoryItem(move.move_id, move.guest_token, itemId);
      if (inventory?.status === 'confirmed') await confirmInventory(move.move_id, move.guest_token);
      setInventory(await getInventory(move.move_id, move.guest_token));
      setError('');
    } catch { setError('물품을 삭제하지 못했어요. 다시 시도해주세요.'); }
  }

  async function confirmList() {
    if (!move) return false;
    try {
      await confirmInventory(move.move_id, move.guest_token);
      setInventory(await getInventory(move.move_id, move.guest_token));
      setError('');
      return true;
    } catch { setError('목록을 확정하지 못했어요. 다시 시도해주세요.'); return false; }
  }

  async function prepareChat() {
    if (!move) throw new Error('이사 정보를 찾을 수 없어요.');
    if (inventory?.status === 'confirmed') {
      await unconfirmInventory(move.move_id, move.guest_token);
      setInventory(await getInventory(move.move_id, move.guest_token));
    }
  }

  function setChatInventory(next: Inventory) {
    setInventory(next);
    setError('');
  }

  async function refreshInventory() {
    if (!move) return;
    setInventory(await getInventory(move.move_id, move.guest_token));
  }

  function updateDraft(change: (current: ReviewDraft) => ReviewDraft) {
    if (!move || !draft) return;
    const next = change(draft);
    sessionStorage.setItem(storageKey(move.move_id), JSON.stringify(next));
    setDraft(next);
  }

  const items = inventory?.items ?? [];
  const added: ReviewDraft['added'] = [];
  const total = inventory?.total_quantity ?? 0;
  return { move, analysis, inventory, draft, loading, error, items, added, total, updateDraft, changeQuantity, removeItem, confirmList, prepareChat, setChatInventory, refreshInventory, retry: () => { setLoading(true); setRetryKey(key => key + 1); } };
}

export function dimensions(item: AnalysisItem): string {
  const { width, depth, height } = item.dimensions_cm;
  if (width == null || depth == null || height == null) return '확인 필요';
  return `${width} × ${depth} × ${height}`;
}
