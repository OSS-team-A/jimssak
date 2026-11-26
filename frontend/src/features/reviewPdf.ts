import { getMove } from '../api/moves';
import type { ReviewDraft } from './review';
import type { AnalysisItem, MoveSnapshot } from '../types/move';

type PdfInput = {
  moveId: string;
  guestToken: string;
  preview: boolean;
  draft: ReviewDraft;
  items: AnalysisItem[];
  added: { label: string; quantity: number }[];
};

const W = 595;
const H = 842;
const ink = '#1a2630';
const line = '#d8dee4';
const pale = '#f0f7f5';
const mint = '#8ef0df';

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, radius: number, fill: string) {
  ctx.fillStyle = fill;
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, radius);
  ctx.fill();
}

function text(ctx: CanvasRenderingContext2D, value: string, x: number, y: number, size = 9, weight = 400, color = ink) {
  ctx.fillStyle = color;
  ctx.font = `${weight} ${size}px Pretendard, sans-serif`;
  ctx.fillText(value, x, y);
}

function cell(ctx: CanvasRenderingContext2D, value: string, x: number, y: number, width: number, height: number, label = false) {
  ctx.fillStyle = label ? pale : '#fff';
  ctx.fillRect(x, y, width, height);
  ctx.strokeStyle = line;
  ctx.lineWidth = .8;
  ctx.strokeRect(x + .4, y + .4, width - .8, height - .8);
  ctx.save();
  ctx.beginPath();
  ctx.rect(x + 6, y, width - 12, height);
  ctx.clip();
  text(ctx, value, x + 7, y + height / 2 + 3, 8, label ? 600 : 400);
  ctx.restore();
}

function sectionTitle(ctx: CanvasRenderingContext2D, number: number, title: string, y: number) {
  roundRect(ctx, 40, y, 18, 18, 9, '#40d9c2');
  text(ctx, String(number), 46, y + 12, 9, 700);
  text(ctx, title, 65, y + 13, 11, 700);
}

function place(record: MoveSnapshot | null, side: 'origin' | 'destination') {
  const value = record?.data[side].value;
  if (!value) return '입력되지 않음';
  const floor = value.floor == null ? '' : value.floor === -1 ? ' · 지하' : ` · ${value.floor}층`;
  return `${value.road_address}${value.detail_address ? ` ${value.detail_address}` : ''}${floor}`;
}

function building(record: MoveSnapshot | null, side: 'origin' | 'destination', preview: boolean) {
  if (preview) return `${side === 'origin' ? 3 : 5}층 · 엘리베이터 확인 필요`;
  const value = record?.data[side].value;
  if (!value) return '확인 필요';
  const floor = value.floor == null ? '층수 확인 필요' : value.floor === -1 ? '지하' : `${value.floor}층`;
  const elevator = value.elevator === 'available' ? '엘리베이터 있음' : value.elevator === 'unavailable' ? '엘리베이터 없음' : '엘리베이터 확인 필요';
  return `${floor} · ${elevator}`;
}

function encodePdf(images: Uint8Array[]) {
  const encoder = new TextEncoder();
  const chunks: Uint8Array[] = [];
  let offset = 0;
  const offsets = [0];
  const add = (data: string | Uint8Array) => {
    const bytes = typeof data === 'string' ? encoder.encode(data) : data;
    chunks.push(bytes);
    offset += bytes.length;
  };
  const object = (id: number, content: string | Uint8Array, tail = '') => {
    offsets[id] = offset;
    add(`${id} 0 obj\n`);
    add(content);
    add(`${tail}\nendobj\n`);
  };
  add('%PDF-1.4\n');
  const pageIds = images.map((_, index) => 3 + index * 3);
  object(1, '<< /Type /Catalog /Pages 2 0 R >>');
  object(2, `<< /Type /Pages /Kids [${pageIds.map(id => `${id} 0 R`).join(' ')}] /Count ${images.length} >>`);
  images.forEach((jpeg, index) => {
    const page = pageIds[index];
    object(page, `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${W} ${H}] /Resources << /XObject << /Im${index} ${page + 1} 0 R >> >> /Contents ${page + 2} 0 R >>`);
    offsets[page + 1] = offset;
    add(`${page + 1} 0 obj\n<< /Type /XObject /Subtype /Image /Width ${W * 2} /Height ${H * 2} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length ${jpeg.length} >>\nstream\n`);
    add(jpeg);
    add('\nendstream\nendobj\n');
    const draw = `q ${W} 0 0 ${H} 0 0 cm /Im${index} Do Q`;
    object(page + 2, `<< /Length ${encoder.encode(draw).length} >>\nstream\n${draw}\nendstream`);
  });
  const xref = offset;
  add(`xref\n0 ${offsets.length}\n0000000000 65535 f \n`);
  for (let id = 1; id < offsets.length; id++) add(`${String(offsets[id]).padStart(10, '0')} 00000 n \n`);
  add(`trailer\n<< /Size ${offsets.length} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF`);
  return new Blob(chunks as BlobPart[], { type: 'application/pdf' });
}

export async function downloadReviewPdf({ moveId, guestToken, preview, draft, items, added }: PdfInput) {
  if (!draft.selectedMoveType) throw new Error('이사 유형을 선택해주세요.');
  const record = preview ? null : await getMove(moveId, guestToken);
  await document.fonts.ready;
  const entries = [
    ...items.filter(item => item.quantity > 0).map(item => ({ label: item.label_ko, quantity: item.quantity, size: Object.values(item.dimensions_cm).every(value => value != null) ? `${item.dimensions_cm.width} × ${item.dimensions_cm.depth} × ${item.dimensions_cm.height}` : '확인 필요', disassembly: item.disassembly_required == null ? '확인 필요' : item.disassembly_required ? '분리 필요' : '해당 없음', note: item.review_reasons.join(', ') || '해당 없음' })),
    ...added.filter(item => item.quantity > 0).map(item => ({ label: item.label, quantity: item.quantity, size: '확인 필요', disassembly: '확인 필요', note: '직접 추가' })),
  ];
  const moveType = { standard: '일반 이사', semi: '반포장 이사', full: '포장 이사' }[draft.selectedMoveType];
  const schedule = record?.data.schedule.value;
  const date = schedule ? `${schedule.move_date} · ${schedule.start_time.slice(0, 5)}` : preview ? '2026. 12. 20. 오전 9시' : '입력되지 않음';
  const pages: Uint8Array[] = [];
  const pageCount = Math.max(1, Math.ceil(entries.length / 9));
  for (let page = 0; page < pageCount; page++) {
    const canvas = document.createElement('canvas');
    canvas.width = W * 2;
    canvas.height = H * 2;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('PDF 문서를 만들 수 없어요.');
    ctx.scale(2, 2);
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, W, H);
    roundRect(ctx, 40, 32, 22, 22, 7, '#40d9c2');
    text(ctx, '짐', 45, 47, 11, 700);
    text(ctx, '짐싹', 68, 47, 12, 700);
    roundRect(ctx, 421, 35, 50, 18, 9, mint);
    text(ctx, '확인 완료', 428, 47, 8, 600);
    text(ctx, `문서번호  JS-${moveId.slice(0, 8).toUpperCase()}`, 478, 47, 7);
    ctx.textAlign = 'center';
    text(ctx, '이삿짐 명세서', W / 2, 88, 20, 700);
    text(ctx, '촬영 영상과 사용자가 확인한 내용을 기준으로 작성했어요.', W / 2, 105, 8);
    ctx.textAlign = 'left';
    sectionTitle(ctx, 1, '이사 기본 정보', 125);
    const origin = place(record, 'origin');
    const destination = place(record, 'destination');
    const fields: [string, string, string, string][] = [
      ['고객명', '미입력', '연락처', '미입력'],
      ['출발지', preview ? '서울특별시 강남구 테헤란로 00 · 3층' : origin, '', ''],
      ['도착지', preview ? '서울특별시 마포구 월드컵북로 00 · 5층' : destination, '', ''],
      ['이사 예정일', date, '이사 유형', moveType],
    ];
    fields.forEach(([a, b, c, d], index) => {
      const y = 151 + index * 26;
      cell(ctx, a, 40, y, 70, 26, true);
      if (c) {
        cell(ctx, b, 110, y, 188, 26);
        cell(ctx, c, 298, y, 70, 26, true);
        cell(ctx, d, 368, y, 187, 26);
      } else cell(ctx, b, 110, y, 445, 26);
    });
    sectionTitle(ctx, 2, '물품 내역', 272);
    const widths = [28, 48, 68, 34, 95, 67, 70, 105];
    const headers = ['No.', '공간', '물품', '수량', '예상 크기(cm)', '예상 무게', '분리 여부', '특이사항'];
    let x = 40;
    headers.forEach((header, index) => { cell(ctx, header, x, 299, widths[index], 26, true); ctx.fillStyle = mint; ctx.fillRect(x + 1, 300, widths[index] - 2, 24); text(ctx, header, x + 5, 316, 7.5, 600); x += widths[index]; });
    const pageEntries = entries.slice(page * 9, (page + 1) * 9);
    pageEntries.forEach((entry, index) => {
      const y = 325 + index * 26;
      let left = 40;
      [String(page * 9 + index + 1), '미분류', entry.label, `${entry.quantity}개`, entry.size, '확인 필요', entry.disassembly, entry.note].forEach((value, column) => {
        cell(ctx, value, left, y, widths[column], 26);
        left += widths[column];
      });
    });
    const total = entries.reduce((sum, entry) => sum + entry.quantity, 0);
    const totalY = 337 + pageEntries.length * 26;
    roundRect(ctx, 40, totalY, 515, 27, 0, pale);
    text(ctx, `총 ${entries.length}종 · ${total}개`, 50, totalY + 17, 9, 700);
    text(ctx, '예상 총 무게  확인 필요', 441, totalY + 17, 8.5, 500);
    const conditionsY = totalY + 38;
    sectionTitle(ctx, 3, '운반 및 현장 조건', conditionsY);
    const buildingY = conditionsY + 26;
    cell(ctx, '출발지', 40, buildingY, 70, 26, true);
    cell(ctx, building(record, 'origin', preview), 110, buildingY, 188, 26);
    cell(ctx, '도착지', 298, buildingY, 70, 26, true);
    cell(ctx, building(record, 'destination', preview), 368, buildingY, 187, 26);
    cell(ctx, '주차', 40, buildingY + 26, 70, 26, true);
    cell(ctx, '업체 확인 필요', 110, buildingY + 26, 188, 26);
    cell(ctx, '사다리차', 298, buildingY + 26, 70, 26, true);
    cell(ctx, '필요 여부 업체 확인', 368, buildingY + 26, 187, 26);
    const note = draft.notes.map(entry => entry.text).join(' · ') || '업체와 실제 작업 조건을 확인해주세요.';
    const memoY = buildingY + 52;
    cell(ctx, '확인 메모', 40, memoY, 70, 48, true);
    cell(ctx, '', 110, memoY, 445, 48);
    ctx.font = '400 8px Pretendard, sans-serif';
    const lines = [''];
    for (const char of note) {
      const current = lines.length - 1;
      if (ctx.measureText(lines[current] + char).width > 425) lines.push('');
      lines[lines.length - 1] += char;
    }
    lines.slice(0, 3).forEach((lineText, index) => text(ctx, lineText, 117, memoY + 14 + index * 13, 8));
    text(ctx, '크기와 무게는 촬영 영상으로 분석한 추정값이에요. 실제 작업 조건과 최종 비용은 이사업체와 확인해주세요.', 42, memoY + 62, 7.5);
    text(ctx, '짐싹 · 촬영으로 정리하는 이사 준비', 40, 815, 7);
    text(ctx, `${page + 1} / ${pageCount}`, 530, 815, 7);
    const binary = atob(canvas.toDataURL('image/jpeg', .92).split(',')[1]);
    pages.push(Uint8Array.from(binary, char => char.charCodeAt(0)));
  }
  const url = URL.createObjectURL(encodePdf(pages));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `짐싹_이삿짐_명세서_${preview ? '미리보기' : moveId.slice(0, 8)}.pdf`;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
