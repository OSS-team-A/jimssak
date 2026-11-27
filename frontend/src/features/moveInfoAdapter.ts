import type { AddressCandidate, MoveSnapshot, PlaceInput } from '../types/move';

export type PlaceDraft = { address: string; candidate: AddressCandidate | null; detail: string; floor: number | null };
export type MoveInfoDraft = { origin: PlaceDraft; destination: PlaceDraft; date: string; time: string };

export const emptyPlace = (): PlaceDraft => ({ address: '', candidate: null, detail: '', floor: null });
export const emptyMoveInfoDraft = (): MoveInfoDraft => ({ origin: emptyPlace(), destination: emptyPlace(), date: '', time: '' });

export const toPlaceInput = (place: PlaceDraft): PlaceInput => ({
  road_address: place.address.trim(),
  detail_address: place.detail.trim() || null,
  floor: place.floor,
  ...(place.candidate ? {
    building_name: place.candidate.building_name,
    adm_cd: place.candidate.adm_cd,
    mountain: place.candidate.mountain,
    lot_main: place.candidate.lot_main,
    lot_sub: place.candidate.lot_sub,
  } : {}),
});

export const fromMoveSnapshot = (record: MoveSnapshot): MoveInfoDraft => ({
  origin: { address: record.data.origin.value?.road_address ?? '', candidate: null,
    detail: record.data.origin.value?.detail_address ?? '', floor: record.data.origin.value?.floor ?? null },
  destination: { address: record.data.destination.value?.road_address ?? '', candidate: null,
    detail: record.data.destination.value?.detail_address ?? '', floor: record.data.destination.value?.floor ?? null },
  date: record.data.schedule.value?.move_date ?? '',
  time: record.data.schedule.value?.start_time.slice(0, 5) ?? '',
});

export const elevatorLabel = (value: MoveSnapshot['data']['origin']['value']) =>
  value?.elevator === 'available' ? '있음' : value?.elevator === 'unavailable' ? '없음' : '확인 필요';

export const buildingNote = (value: MoveSnapshot['data']['origin']['value']) =>
  value?.building?.status === 'found'
    ? '건물 정보에서 찾았어요. 실제 조건과 다르면 수정해주세요.'
    : '건물 정보를 확인하지 못했어요. 실제 조건을 확인해주세요.';
