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
export type MoveSnapshot = {
  move_id: string; owner: 'member' | 'guest'; stage: 'collecting';
  data: {
    room_video: { status: 'empty' | 'uploaded'; video: { media_id: string; filename: string } | null };
    analysis: { status: 'not_started' | 'queued' | 'running' | 'completed' | 'failed'; progress?: number | null; item_count?: number | null; demo?: boolean; error?: { code?: string; message?: string } | null };
    inventory: { status: 'not_started' | 'empty' | 'draft' | 'confirmed'; item_count?: number; total_quantity?: number };
    origin: { status: 'empty' | 'partial' | 'filled'; value: (PlaceInput & { building: { status: string }; elevator: string }) | null };
    destination: { status: 'empty' | 'partial' | 'filled'; value: (PlaceInput & { building: { status: string }; elevator: string }) | null };
    route: { status: 'not_ready' | 'found' | 'not_found' | 'failed'; distance_m: number | null };
    schedule: { status: 'empty' | 'filled'; value: { move_date: string; start_time: string } | null };
  };
  missing: MoveStep[]; next_step: MoveStep | null;
};

export type MoveSummary = {
  move_id: string;
  created_at: string;
  move_date: string | null;
  start_time: string | null;
  origin_address: string | null;
  destination_address: string | null;
  stage: 'preparing' | 'after_move' | 'comparing' | 'done';
};

export type MoveList = { moves: MoveSummary[]; total: number; limit: number; offset: number };

export type AnalysisItem = {
  item_id: string; label_ko: string; quantity: number;
  dimensions_cm: { width: number | null; depth: number | null; height: number | null };
  disassembly_required: boolean | null; needs_review: boolean; review_reasons: string[];
  snapshots: { time_ms: number; image_url: string | null }[];
};

export type AnalysisResult = {
  analysis_id: string | null;
  status: 'not_started' | 'queued' | 'running' | 'completed' | 'failed';
  demo?: boolean;
  items: AnalysisItem[];
};

export type InventoryItem = AnalysisItem & {
  source: 'demo' | 'ai' | 'user';
  notes: string | null;
};
export type Inventory = { status: 'empty' | 'draft' | 'confirmed'; item_count: number; total_quantity: number; items: InventoryItem[] };
export type InventoryItemCreate = { label_ko: string; quantity?: number; dimensions_cm?: AnalysisItem['dimensions_cm']; notes?: string | null; disassembly_required?: boolean | null };
export type InventoryItemPatch = Partial<InventoryItemCreate>;
