import { uploadMoveVideo } from '../api/moves';

type MoveSession = { move_id: string; guest_token: string };

let pending: Promise<void> | null = null;
let lastFile: File | null = null;
let lastMove: MoveSession | null = null;
let failure: unknown = null;

export function startInitialVideoUpload(move: MoveSession, file: File) {
  lastMove = move;
  lastFile = file;
  failure = null;
  pending = uploadMoveVideo(move.move_id, move.guest_token, file).then(() => {
    failure = null;
  }).catch(cause => {
    failure = cause;
    throw cause;
  });
  // The form stays usable while the request runs; the final step awaits this promise.
  void pending.catch(() => {});
}

export async function waitForInitialVideoUpload() {
  if (pending) await pending;
  else if (failure) throw failure;
}

export function canRetryInitialVideoUpload() {
  return Boolean(lastMove && lastFile && failure);
}

export function retryInitialVideoUpload() {
  if (!lastMove || !lastFile) throw new Error('다시 업로드할 영상을 찾을 수 없어요.');
  startInitialVideoUpload(lastMove, lastFile);
  return waitForInitialVideoUpload();
}
