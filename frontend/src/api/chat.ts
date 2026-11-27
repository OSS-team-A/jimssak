import { apiRequest } from './client';
import type { Inventory } from '../types/move';

export type ChatTurn = { role: 'user' | 'assistant'; content: string };
export type ChatReply = {
  message: string;
  proposal_id: string | null;
  summary: string[];
  choices: string[];
  advance: boolean;
};
export type ChatApplied = { inventory: Inventory; message: string; choices: string[]; advance: boolean };

const chatHeaders = (guestToken: string) => ({
  'Content-Type': 'application/json',
  ...(guestToken ? { 'X-Guest-Token': guestToken } : {}),
});
const chatPath = (moveId: string) => `/api/v1/moves/${encodeURIComponent(moveId)}/chat`;

export const sendMoveChat = (moveId: string, guestToken: string, message: string, history: ChatTurn[]) =>
  apiRequest<ChatReply>(chatPath(moveId), {
    method: 'POST',
    headers: chatHeaders(guestToken),
    body: JSON.stringify({ message, history: history.slice(-8) }),
  });

export const applyMoveChat = (moveId: string, guestToken: string, proposalId: string) =>
  apiRequest<ChatApplied>(`${chatPath(moveId)}/apply`, {
    method: 'POST',
    headers: chatHeaders(guestToken),
    body: JSON.stringify({ proposal_id: proposalId }),
  });
