import { useCallback, useEffect, useState } from 'react';
import { getMove } from '../api/moves';
import type { MoveSnapshot } from '../types/move';

export function currentMove() {
  const raw = sessionStorage.getItem('jimssak-member-move') || sessionStorage.getItem('jimssak-guest-move');
  try {
    const parsed = raw ? JSON.parse(raw) as { move_id?: string; guest_token?: string } : null;
    return parsed?.move_id ? { id: parsed.move_id, token: parsed.guest_token || '' } : null;
  } catch { return null; }
}

export function useCurrentMove() {
  const move = currentMove();
  const [snapshot, setSnapshot] = useState<MoveSnapshot | null>(null);
  const [loading, setLoading] = useState(Boolean(move));
  const [error, setError] = useState('');
  const refresh = useCallback(async () => {
    if (!move) return;
    setLoading(true); setError('');
    try { setSnapshot(await getMove(move.id, move.token)); }
    catch { setError('상태를 불러오지 못했어요. 다시 시도해주세요.'); }
    finally { setLoading(false); }
  }, [move?.id, move?.token]);
  useEffect(() => { void refresh(); }, [refresh]);
  return { move, snapshot, loading, error, refresh };
}
