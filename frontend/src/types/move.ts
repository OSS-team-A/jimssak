export type MoveStep = 'room_video' | 'origin' | 'destination' | 'schedule';
export type MoveCreated = {
  move_id: string;
  owner: 'member' | 'guest';
  guest_token: string | null;
  expires_at: string | null;
};
export type MoveSnapshot = {
  move_id: string;
  owner: 'member' | 'guest';
  stage: 'collecting';
  data: {
    room_video: { status: 'empty' | 'uploaded'; video: { media_id: string; filename: string } | null };
    analysis: { status: 'not_started' | 'empty' };
    inventory: { status: 'not_started' | 'empty' };
    origin: { status: 'not_started' | 'empty' };
    destination: { status: 'not_started' | 'empty' };
    schedule: { status: 'not_started' | 'empty' };
  };
  missing: MoveStep[];
  next_step: MoveStep | null;
};
