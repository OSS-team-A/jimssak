import chairPhoto from '../assets/review-preview-chair.png';
import type { AnalysisResult } from '../types/move';

// Figma sample data is available only through the development preview route.
export const reviewPreview: AnalysisResult = {
  analysis_id: 'figma-sc07-preview',
  status: 'completed',
  items: [
    { item_id: 'preview-bed', label_ko: '침대', quantity: 1, dimensions_cm: { width: 150, depth: 200, height: 50 }, disassembly_required: null, needs_review: true, review_reasons: ['분리 여부 확인'], snapshots: [] },
    { item_id: 'preview-desk', label_ko: '책상', quantity: 1, dimensions_cm: { width: 120, depth: 60, height: 75 }, disassembly_required: null, needs_review: true, review_reasons: ['크기 확인'], snapshots: [] },
    { item_id: 'preview-chair', label_ko: '의자', quantity: 1, dimensions_cm: { width: 60, depth: 60, height: 110 }, disassembly_required: false, needs_review: false, review_reasons: [], snapshots: [{ time_ms: 0, image_url: chairPhoto }] },
    { item_id: 'preview-wardrobe', label_ko: '옷장', quantity: 1, dimensions_cm: { width: 120, depth: 60, height: 200 }, disassembly_required: null, needs_review: true, review_reasons: ['분리 여부 확인'], snapshots: [] },
    { item_id: 'preview-monitor', label_ko: '모니터', quantity: 1, dimensions_cm: { width: 62, depth: 20, height: 45 }, disassembly_required: false, needs_review: false, review_reasons: [], snapshots: [] },
  ],
};
