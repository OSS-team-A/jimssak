/** Formatting only documented MoveSummary fields keeps API data outside the page markup. */
export function shortPlace(address: string | null) {
  if (!address) return '주소 확인 전';
  const [province, district] = address.split(' ');
  return `${province.replace(/특별시|광역시|특별자치시|특별자치도|도$/, '')} ${district || ''}`.trim();
}

export function moveDate(date: string | null) {
  if (!date) return '이사 날짜 확인 전';
  const [year, month, day] = date.split('-').map(Number);
  return `${year}년 ${month}월 ${day}일 이사`;
}
