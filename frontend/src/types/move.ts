export type MoveStep = 'room_video' | 'origin' | 'destination' | 'schedule';
export type MoveCreated = {
  move_id: string; owner: 'member' | 'guest'; guest_token: string | null; expires_at: string | null;
};
export type AddressCandidate = {
  road_address: string; jibun_address: string; building_name: string | null;
  adm_cd: string; mountain: boolean; lot_main: number; lot_sub: number; zip_code: string | null;
};
export type PlaceInput = {
  road_address: string; detail_address?: string | null; floor?: number | null;
  elevator?: 'available' | 'unavailable' | 'unknown';
  building_name?: string | null; adm_cd?: string; mountain?: boolean; lot_main?: number; lot_sub?: number;
};
export type MoveUpdate = {
  origin?: PlaceInput; destination?: PlaceInput;
  schedule?: { move_date: string; start_time: string };
};
// The analysis and inventory fields below describe the contract the UI needs.
// The current server only returns not_started/empty; no result is synthesized.
export type InventoryItem = {
  id: string; name: string; quantity: number;
  estimated_size?: string | null; estimated_weight?: string | null;
  disassembly?: string | null; location?: string | null;
  photo_url?: string | null; excluded?: boolean; note?: string | null;
};
export type MoveSnapshot = {
  move_id: string; owner: 'member' | 'guest'; stage: 'collecting';
  data: {
    room_video: { status: 'empty' | 'uploaded'; video: { media_id: string; filename: string } | null };
    analysis: { status: 'not_started' | 'empty' | 'queued' | 'running' | 'completed' | 'failed'; error?: string | null };
    inventory: { status: 'not_started' | 'empty' | 'ready'; items?: InventoryItem[]; version?: string };
    origin: { status: 'empty' | 'partial' | 'filled'; value: (PlaceInput & { building: { status: string }; elevator: string }) | null };
    destination: { status: 'empty' | 'partial' | 'filled'; value: (PlaceInput & { building: { status: string }; elevator: string }) | null };
    route: { status: 'not_ready' | 'found' | 'not_found' | 'failed'; distance_m: number | null };
    schedule: { status: 'empty' | 'filled'; value: { move_date: string; start_time: string } | null };
  };
  missing: MoveStep[]; next_step: MoveStep | null;
};
