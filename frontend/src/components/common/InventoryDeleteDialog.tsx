function objectParticle(name: string) {
  const last = name.charCodeAt(name.length - 1);
  return last >= 0xac00 && last <= 0xd7a3 && (last - 0xac00) % 28 === 0 ? '를' : '을';
}

export function InventoryDeleteDialog({ label, onCancel, onConfirm }: {
  label: string;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  return <div className="review-delete-backdrop" onClick={onCancel}>
    <div className="review-delete-dialog" role="dialog" aria-modal="true" aria-labelledby="review-delete-title" onClick={event => event.stopPropagation()}>
      <h2 id="review-delete-title">{label}{objectParticle(label)} 삭제할까요?</h2>
      <p>{label}{objectParticle(label)} 목록에서 삭제할까요?</p>
      <div><button type="button" onClick={onCancel}>아니요</button><button type="button" onClick={onConfirm}>예</button></div>
    </div>
  </div>;
}
