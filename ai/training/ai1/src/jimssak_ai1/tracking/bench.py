"""추적·재식별 벤치마크 (합성 영상): 검출기 × 추적기(BoT-SORT/ByteTrack) × 재등장 연결 방식.

재등장 연결 방식
- none      : 추적기 출력 그대로 (B0/B1의 기본)
- hist_rule : 기존 규칙 — 같은 클래스·시간 겹침 없음·짧은 공백·끊긴 위치 박스 IoU·색 히스토그램
  유사도 모두 충족 시 병합
- dinov2    : 같은 클래스·시간 겹침 없음 + DINOv2 임베딩 코사인 유사도 ≥ 임계값 (위치 조건 없음 →
  루프 복귀 재등장 처리)
- trained   : dinov2와 같되 ABO triplet 학습 헤드 임베딩 사용
임계값은 검증용 합성 영상(테스트와 다른 이미지)에서 물품 F1이 최대가 되도록 고른다.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..appearance import hist_similarity, masked_hsv_hist
from ..common import log
from ..frameeval import box_iou_matrix
from .mot_metrics import compute_mot, item_level
from .synth_video import generate, load_objects

FPS = 30.0


def _tracker(kind: str, device: str):
    import ultralytics.cfg as ucfg
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import YAML, IterableSimpleNamespace

    y = Path(ucfg.__file__).parent / "trackers" / f"{kind}.yaml"
    cfg = IterableSimpleNamespace(**YAML.load(str(y)))
    cfg.device = device
    return TRACKER_MAP[cfg.tracker_type](args=cfg)


def detect_sequence(model, cls_map, frames, device, imgsz=640):
    out = []
    for fr in frames:
        r = model.predict(fr, conf=0.1, imgsz=imgsz, device=device, verbose=False)[0]
        out.append(r.boxes.cpu().numpy())
    return out


def track_sequence(kind, dets, frames, cls_map, device):
    trk = _tracker(kind, device)
    tracks = defaultdict(lambda: {"frames": [], "boxes": [], "conf": [], "cls": Counter()})
    per_frame = []
    for t, (d, fr) in enumerate(zip(dets, frames)):
        res = trk.update(d, fr)
        cur = []
        for row in res:
            tid = int(row[4])
            box = row[:4].astype(float)
            conf = float(row[5])
            cid = int(row[6])
            code = cls_map.get(cid)
            if code is None:
                continue
            T = tracks[tid]
            T["frames"].append(t)
            T["boxes"].append(box)
            T["conf"].append(conf)
            T["cls"][code] += conf
            cur.append((tid, box))
        per_frame.append(cur)
    for T in tracks.values():
        T["class"] = T["cls"].most_common(1)[0][0]
    return dict(tracks), per_frame


def track_crops(tracks, frames, k=8):
    crops = {}
    for tid, T in tracks.items():
        n = len(T["frames"])
        sel = np.linspace(0, n - 1, min(k, n)).astype(int)
        cs = []
        for i in sel:
            x1, y1, x2, y2 = [int(v) for v in T["boxes"][i]]
            fr = frames[T["frames"][i]]
            c = fr[max(0, y1) : max(y1 + 2, y2), max(0, x1) : max(x1 + 2, x2)]
            if c.size:
                cs.append(c)
        crops[tid] = cs
    return crops


def _overlap(a, b):
    return bool(set(a["frames"]) & set(b["frames"]))


def group_tracks(tracks, pair_ok):
    """pair_ok(a,b) -> score|None. 점수 높은 순으로 병합, 그룹 내 시간 겹침 금지."""
    ids = sorted(tracks)
    parent = {i: i for i in ids}
    members = {i: [i] for i in ids}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    cands = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            A, B = tracks[a], tracks[b]
            if A["class"] != B["class"] or _overlap(A, B):
                continue
            s = pair_ok(a, b)
            if s is not None:
                cands.append((s, a, b))
    for s, a, b in sorted(cands, reverse=True):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if any(_overlap(tracks[x], tracks[y]) for x in members[ra] for y in members[rb]):
            continue
        parent[rb] = ra
        members[ra] += members.pop(rb)
    return {t: find(t) for t in ids}


def track_embeddings(embedder, crops) -> dict:
    E = {}
    for t, cs in crops.items():
        if cs:
            z = embedder.embed(cs).mean(0)
            E[t] = z / max(np.linalg.norm(z), 1e-12)
    return E


def relink(method, tracks, crops, embedders, thr, hist_cfg):
    if method == "none":
        return {t: t for t in tracks}
    if method == "hist_rule":
        hists = {
            t: [h for h in (masked_hsv_hist(c, None) for c in cs) if h is not None]
            for t, cs in crops.items()
        }

        def ok(a, b):
            A, B = (
                (tracks[a], tracks[b])
                if tracks[a]["frames"][0] < tracks[b]["frames"][0]
                else (tracks[b], tracks[a])
            )
            ha, hb = (hists[a], hists[b]) if A is tracks[a] else (hists[b], hists[a])
            gap = (B["frames"][0] - A["frames"][-1]) / FPS
            iou = float(box_iou_matrix(np.array([A["boxes"][-1]]), np.array([B["boxes"][0]]))[0, 0])
            if not ha or not hb:
                return None
            sim = float(np.mean([max(hist_similarity(x, y) for y in hb) for x in ha]))
            if (
                gap <= hist_cfg["auto_max_gap_sec"]
                and iou >= hist_cfg["auto_min_box_iou"]
                and sim >= hist_cfg["auto_min_appearance"]
            ):
                return sim
            return None

        return group_tracks(tracks, ok)
    E = (
        embedders[method]
        if isinstance(embedders[method], dict)
        else track_embeddings(embedders[method], crops)
    )

    def ok2(a, b):
        if a not in E or b not in E:
            return None
        s = float(E[a] @ E[b])
        return s if s >= thr else None

    return group_tracks(tracks, ok2)


def evaluate_grouping(tracks, per_frame, gt_frames, gmap, confirm_frames, confirm_conf):
    groups = defaultdict(list)
    for t, g in gmap.items():
        groups[g].append(t)
    counted, item_class = set(), {}
    for g, ts in groups.items():
        nf = sum(len(tracks[t]["frames"]) for t in ts)
        mc = np.mean([c for t in ts for c in tracks[t]["conf"]])
        item_class[g] = Counter(
            {tracks[t]["class"]: len(tracks[t]["frames"]) for t in ts}
        ).most_common(1)[0][0]
        if nf >= confirm_frames and mc >= confirm_conf:
            counted.add(g)
    frames = [
        {"gt": [(i, b) for i, _, b in gf], "tr": [(gmap[t], b) for t, b in pf]}
        for gf, pf in zip(gt_frames, per_frame)
    ]
    gt_class = {i: c for gf in gt_frames for i, c, _ in gf}
    nvis = Counter(i for gf in gt_frames for i, _, _ in gf)
    listable = {i for i, n in nvis.items() if n >= confirm_frames}
    mot = compute_mot(frames)
    il = item_level(frames, {t: gmap[t] for t in tracks}, gt_class, item_class, listable, counted)
    return mot, il


def run_bench(
    *,
    seq_images: list[Path],
    lbl_dir: Path,
    names: list[str],
    detectors: dict,
    trackers: list[str],
    methods: list[str],
    embedders: dict,
    thresholds: dict,
    hist_cfg: dict,
    n_frames: int,
    out_size,
    device: str,
    tax,
    confirm_frames: int = 8,
    confirm_conf: float = 0.35,
    seed: int = 0,
    collect_scores_for: list[str] | None = None,
    thr_grid=None,
) -> list[dict]:
    """반환: 행 목록 (seq, detector, tracker, method, 지표...). thr_grid가 있으면 embedding 방식의
    임계값별 결과도 행으로.
    """
    from ultralytics import YOLO

    rows = []
    models = {k: YOLO(v) for k, v in detectors.items()}
    for si, ip in enumerate(seq_images):
        img, objs = load_objects(ip, lbl_dir / (ip.stem + ".txt"), names)
        frames, gt_frames = [], []
        for fr, g in generate(img, objs, n_frames, tuple(out_size), seed + si):
            frames.append(fr)
            gt_frames.append(g)
        for dl, model in models.items():
            cls_map, _ = tax.model_class_to_code({int(k): v for k, v in model.names.items()})
            dets = detect_sequence(model, cls_map, frames, device)
            for kind in trackers:
                tracks, per_frame = track_sequence(kind, dets, frames, cls_map, device)
                crops = track_crops(tracks, frames)
                cached = {
                    m: track_embeddings(embedders[m], crops) for m in methods if m in embedders
                }
                for m in methods:
                    grid = (
                        thr_grid
                        if (thr_grid is not None and m in embedders)
                        else [thresholds.get(m)]
                    )
                    for thr in grid:
                        if m in embedders and thr is None:
                            continue  # 검증에서 임계값을 정하지 못한 경우 실행하지 않음
                        gmap = relink(m, tracks, crops, cached, thr, hist_cfg)
                        mot, il = evaluate_grouping(
                            tracks, per_frame, gt_frames, gmap, confirm_frames, confirm_conf
                        )
                        rows.append(
                            {
                                "sequence": ip.stem,
                                "detector": dl,
                                "tracker": kind,
                                "relink": m,
                                "threshold": thr,
                                **{f"mot_{k}": v for k, v in mot.items()},
                                **{
                                    f"item_{k}": v for k, v in il.items() if k != "count_abs_errors"
                                },
                                "count_abs_errors": il["count_abs_errors"],
                            }
                        )
        log(f"    합성 영상 {si + 1}/{len(seq_images)} 완료 ({ip.stem}, 객체 {len(objs)}개)")
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    keys = defaultdict(list)
    for r in rows:
        keys[(r["detector"], r["tracker"], r["relink"], r["threshold"])].append(r)
    out = []
    for (d, t, m, thr), rs in sorted(keys.items(), key=lambda x: str(x[0])):
        agg = {"detector": d, "tracker": t, "relink": m, "threshold": thr, "sequences": len(rs)}
        for k in [
            "mot_HOTA",
            "mot_DetA",
            "mot_AssA",
            "mot_IDF1",
            "mot_MOTA",
            "item_precision",
            "item_recall",
            "item_f1",
            "item_duplicate_rate",
            "item_wrong_merge_rate",
            "item_reappearance_link_accuracy",
        ]:
            v = [r[k] for r in rs if isinstance(r.get(k), (int, float))]
            agg[k] = round(float(np.mean(v)), 4) if v else None
        for k in [
            "mot_IDSW",
            "item_duplicates",
            "item_wrong_merges",
            "item_TP",
            "item_FP",
            "item_FN",
            "item_items_gt",
            "item_reappearing_gt",
            "item_reappearance_link_correct",
        ]:
            agg[k] = int(sum(r.get(k) or 0 for r in rs))
        errs = [e for r in rs for e in r["count_abs_errors"].values()]
        agg["count_MAE"] = round(float(np.mean(errs)), 4) if errs else None
        out.append(agg)
    return out
