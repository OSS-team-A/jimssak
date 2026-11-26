"""영상 1편 추론: YOLO 분할 + BoT-SORT 추적 → 프레임별 JSONL, 결과 영상, 마스크 샘플, 물품 후보
목록.

Ultralytics의 model.track()은 추적기가 확정하지 않은 검출을 결과에서 지워 버린다.
여기서는 '모든 검출 + nullable track_id'를 기록하기 위해 model.predict()로 검출한 뒤,
Ultralytics에 포함된 BoT-SORT 추적기를 model.track()과 같은 방식(같은 설정, 같은 입력)으로 직접
호출한다.
  - 검출 conf 하한 0.1 (model.track 기본값과 동일; configs/pipeline_default.yaml)
  - tracker.update(result.boxes.cpu().numpy(), 원본 프레임)
track_id는 이 영상 실행 안에서만 의미가 있는 임시 추적 번호다. 같은 실물이 다시 나타나면 다른 번호가
될 수 있다.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .appearance import masked_hsv_hist
from .common import (
    code_fingerprint,
    environment_info,
    git_info,
    imwrite,
    log,
    now_iso,
    sha256_file,
    sha256_text,
    write_csv,
    write_json,
)
from .taxonomy import Taxonomy
from .video_io import cleanup_temp, open_video

ITEM_FIELDS = [
    "video_id",
    "run_id",
    "pred_item_id",
    "status",
    "category",
    "category_ko",
    "model_class",
    "box_fit",
    "n_frames",
    "mean_conf",
    "max_conf",
    "first_frame",
    "last_frame",
    "first_time_sec",
    "last_time_sec",
    "rep_frame_idx",
    "rep_time_sec",
    "rep_images",
    "track_ids",
    "track_n_frames",
    "note",
]
TRACK_FIELDS = [
    "track_id",
    "pred_item_id",
    "status",
    "status_reason",
    "listed",
    "category",
    "model_class",
    "class_votes",
    "n_frames",
    "mean_conf",
    "max_conf",
    "first_frame",
    "last_frame",
    "first_time_sec",
    "last_time_sec",
    "first_box",
    "last_box",
]
AI1_BOX_FIT = "unknown"  # AI1은 박스 적합을 판정하지 않는다


def _color(tid: int | None) -> tuple[int, int, int]:
    if tid is None:
        return (160, 160, 160)
    rng = np.random.default_rng(int(tid) * 7919 + 17)
    c = rng.integers(60, 255, size=3)
    return int(c[0]), int(c[1]), int(c[2])


def _mask_polygons(mask: np.ndarray, offset=(0, 0), eps: float = 1.0) -> list[list[list[int]]]:
    """이진 마스크 → 외곽선 다각형 목록 (원본 좌표, ε=1px 단순화, 구멍은 표현하지 않음).

    mask가 박스 영역만 잘라낸 것이면 offset=(x0, y0)을 더해 원본 좌표로 되돌린다.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polys = []
    ox, oy = offset
    for c in contours:
        if cv2.contourArea(c) < 4:
            continue
        a = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(a) >= 3:
            polys.append([[int(x) + ox, int(y) + oy] for x, y in a])
    return polys


def _box_region(box, w: int, h: int, pad: int = 2) -> tuple[int, int, int, int]:
    """마스크 연산을 박스 주변으로 제한하기 위한 정수 영역.

    Ultralytics 마스크는 박스로 잘려 있다.
    """
    x1 = max(0, int(np.floor(box[0])) - pad)
    y1 = max(0, int(np.floor(box[1])) - pad)
    x2 = min(w, int(np.ceil(box[2])) + pad)
    y2 = min(h, int(np.ceil(box[3])) + pad)
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class RepCandidate:
    conf: float
    time_sec: float
    frame_idx: int
    crop: np.ndarray
    full: np.ndarray


@dataclass
class TrackState:
    track_id: int
    class_conf: Counter = field(default_factory=Counter)
    confs: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    times: list = field(default_factory=list)
    first_box: list | None = None
    last_box: list | None = None
    first_crop: np.ndarray | None = None
    last_crop: np.ndarray | None = None
    reps: list = field(default_factory=list)
    hists: list = field(default_factory=list)
    hist_stride: int = 1
    hist_counter: int = 0


def _crop(frame: np.ndarray, box, pad: float = 0.1, max_side: int = 512) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - pad * bw))
    y1 = int(max(0, y1 - pad * bh))
    x2 = int(min(w, x2 + pad * bw))
    y2 = int(min(h, y2 + pad * bh))
    c = frame[y1 : max(y1 + 1, y2), x1 : max(x1 + 1, x2)]
    s = max_side / max(c.shape[:2])
    if s < 1:
        c = cv2.resize(
            c,
            (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))),
            interpolation=cv2.INTER_AREA,
        )
    return c.copy()


def _thumb_with_box(frame, box, mask, color, max_side=720):
    """전체 프레임 축소본 위에 해당 물품의 마스크·박스를 표시한 대표 사진."""
    s = min(1.0, max_side / max(frame.shape[:2]))
    size = (max(1, int(frame.shape[1] * s)), max(1, int(frame.shape[0] * s)))
    img = cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if s < 1 else frame.copy()
    if mask is not None:
        ms = (
            cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
            if s < 1
            else mask
        )
        img[ms] = (img[ms] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
    x1, y1, x2, y2 = [int(v * s) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    return img


def _offer_rep(ts: TrackState, cand_fn, conf: float, t: float, fidx: int, k: int, gap: float):
    """시간 간격 gap 이상을 유지하며 confidence 상위 k개 대표 후보를 온라인으로 유지."""
    conflicts = [r for r in ts.reps if abs(r.time_sec - t) < gap]
    if conflicts and conf <= max(r.conf for r in conflicts):
        return
    if not conflicts and len(ts.reps) >= k and conf <= min(r.conf for r in ts.reps):
        return
    crop, full = cand_fn()
    ts.reps = [r for r in ts.reps if r not in conflicts]
    ts.reps.append(RepCandidate(conf, t, fidx, crop, full))
    ts.reps.sort(key=lambda r: -r.conf)
    ts.reps = ts.reps[:k]


def make_contact_sheet(
    run_dir: Path,
    item_rows: list[dict],
    out_name: str = "items_contact_sheet.jpg",
    cell: int = 240,
    cols: int = 6,
) -> Path | None:
    """물품 후보별 대표 crop을 한 장에 모은 검수용 이미지."""
    if not item_rows:
        return None
    from .common import imread

    tiles = []
    for r in item_rows:
        pid = r["pred_item_id"]
        tid = r.get("track_ids", "").split(";")[0]
        crop_path = (
            run_dir / "items" / (f"T{int(tid):05d}" if tid.isdigit() else pid) / "rep1_crop.jpg"
        )
        tile = np.full((cell + 44, cell, 3), 255, np.uint8)
        if crop_path.is_file():
            c = imread(crop_path)
            s = cell / max(c.shape[:2])
            c = cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))))
            tile[44 : 44 + c.shape[0], : c.shape[1]] = c
        st = "OK" if r["status"] == "confirmed" else "REVIEW"
        cv2.putText(tile, f"{pid} {st}", (3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(
            tile,
            f"{r['model_class']} {float(r['first_time_sec']):.1f}-{float(r['last_time_sec']):.1f}s",
            (3, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 160),
            1,
        )
        tiles.append(tile)
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]
    out = run_dir / out_name
    imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


def run_inference(
    *,
    video: Path,
    video_id: str,
    run_dir: Path,
    run_label: str,
    weights: Path,
    device: str,
    cfg: dict,
    cfg_path: Path,
    taxonomy: Taxonomy,
    tracker_cfg_path: Path | None,
    max_frames: int | None,
    save_video: bool,
    argv: list[str],
) -> dict:
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import YAML, IterableSimpleNamespace

    det_cfg, list_cfg, ev_cfg = cfg["detector"], cfg["listing"], cfg["evidence"]
    stride = int(cfg.get("video", {}).get("vid_stride", 1))
    timing = defaultdict(float)
    notes = [
        "track_id는 이 실행 안에서만 유효한 임시 추적 번호이며, 같은 실물의 재등장 시 다른 번호가 "
        "될 수 있음",
        "좌표·마스크는 회전 메타데이터를 적용한 표시 방향의 원본 해상도 픽셀 좌표",
        "box_fit은 AI1이 판정하지 않음(unknown). 이 목록은 '개별 운반 후보 목록'이며 치수·박스 "
        "적합 판정이 아님",
        "JSONL의 mask_polygons는 외곽선을 ε=1px로 단순화한 것(구멍 미표현). 정확한 마스크는 "
        "mask_samples/*.png",
    ]
    if device == "cpu":
        notes.append("CPU 실행: 처리 시간은 서버 GPU 시간이 아님")

    # ---------- 영상 열기 ----------
    t0 = time.perf_counter()
    cap, vinfo, tmp_video = open_video(video)
    timing["video_open_s"] = time.perf_counter() - t0
    log(
        f"영상: {video} | {vinfo.decoded_width}x{vinfo.decoded_height} (회전 적용 후) | "
        f"FPS {vinfo.fps:.3f} | "
        f"보고 프레임 수 {vinfo.frame_count_reported} | 열기 방식 {vinfo.opened_via}"
    )
    video_sha = sha256_file(video)

    # ---------- 모델 로딩 (초기 로딩 시간) ----------
    t0 = time.perf_counter()
    model = YOLO(str(weights))
    if model.task != "segment":
        cleanup_temp(tmp_video)
        raise SystemExit(f"오류: 분할(segment) 모델이 아닙니다: {weights} (task={model.task})")
    names = {int(k): v for k, v in model.names.items()}
    dummy = np.zeros((vinfo.decoded_height, vinfo.decoded_width, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=det_cfg["imgsz"], device=device, verbose=False)  # 장치 이동·워밍업
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    timing["model_load_and_warmup_s"] = time.perf_counter() - t0
    log(f"모델 로딩+워밍업 {timing['model_load_and_warmup_s']:.2f}s | {weights} | 장치 {device}")

    cls_to_code, map_mode = taxonomy.model_class_to_code(names)
    support = taxonomy.support_table(names)
    if list_cfg.get("include_categories", "all_mapped") == "all_mapped":
        listed_codes = {c for c in cls_to_code.values() if c != "other"}
    else:
        listed_codes = set(list_cfg["include_categories"])

    # ---------- 추적기 ----------
    import ultralytics.cfg as ucfg

    default_tracker = Path(ucfg.__file__).parent / "trackers" / "botsort.yaml"
    tpath = Path(tracker_cfg_path) if tracker_cfg_path else default_tracker
    tracker_text = tpath.read_text(encoding="utf-8")
    shutil.copyfile(tpath, run_dir / "tracker_config.yaml")
    tcfg = IterableSimpleNamespace(**YAML.load(str(tpath)))
    tcfg.device = device
    tracker = TRACKER_MAP[tcfg.tracker_type](args=tcfg)

    # ---------- 출력 준비 ----------
    (run_dir / "mask_samples").mkdir(exist_ok=True)
    (run_dir / "items").mkdir(exist_ok=True)
    (run_dir / "tracks").mkdir(exist_ok=True)
    jsonl_f = open(run_dir / "frames.jsonl", "w", encoding="utf-8")
    writer = None
    out_video_path = None
    vscale = float(ev_cfg.get("annotated_video_scale", 0.5))
    out_fps = (vinfo.fps or 30.0) / stride
    if save_video:
        ow, oh = (
            int(vinfo.decoded_width * vscale) // 2 * 2,
            int(vinfo.decoded_height * vscale) // 2 * 2,
        )
        out_video_path = run_dir / "annotated.mp4"
        writer = cv2.VideoWriter(
            str(out_video_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (ow, oh)
        )
        if not writer.isOpened():
            out_video_path = run_dir / "annotated.avi"
            writer = cv2.VideoWriter(
                str(out_video_path), cv2.VideoWriter_fourcc(*"MJPG"), out_fps, (ow, oh)
            )
        if not writer.isOpened():
            notes.append("결과 영상 저장 실패: VideoWriter를 열 수 없음")
            writer = None
            out_video_path = None

    tracks: dict[int, TrackState] = {}
    mask_every = float(ev_cfg.get("mask_sample_every_sec", 2.0))
    next_mask_t = 0.0
    mask_index = []
    n_read = n_proc = n_det_total = n_det_tracked = 0
    speed_acc = Counter()
    rep_k = int(ev_cfg.get("rep_images_per_item", 3))
    rep_gap = float(ev_cfg.get("rep_min_gap_sec", 0.5))
    hist_cap = int(ev_cfg.get("appearance_samples_per_track", 24))
    t_loop = time.perf_counter()
    t_offset = 0.0
    ow = oh = 0
    if writer is not None:
        ow, oh = (
            int(vinfo.decoded_width * vscale) // 2 * 2,
            int(vinfo.decoded_height * vscale) // 2 * 2,
        )

    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            timing["decode_s"] += time.perf_counter() - t0
            if not ok:
                break
            fidx = n_read
            n_read += 1
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if fidx % stride != 0:
                continue
            if max_frames is not None and n_proc >= max_frames:
                break
            t_sec = (
                (pos_ms / 1000.0)
                if pos_ms and pos_ms > 0
                else (fidx / vinfo.fps if vinfo.fps else 0.0)
            )
            # POS_MSEC는 '다음' 프레임 위치를 줄 수 있는 백엔드가 있어 첫 프레임 0 기준으로 보정
            if fidx == 0:
                t_offset = t_sec
            t_sec = max(0.0, t_sec - t_offset)
            n_proc += 1

            # --- 검출·분할 ---
            t0 = time.perf_counter()
            r = model.predict(
                frame,
                conf=det_cfg["conf"],
                iou=det_cfg["iou"],
                imgsz=det_cfg["imgsz"],
                max_det=det_cfg.get("max_det", 300),
                retina_masks=det_cfg.get("retina_masks", True),
                device=device,
                verbose=False,
            )[0]
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            timing["predict_s"] += time.perf_counter() - t0
            for k, v in (r.speed or {}).items():
                speed_acc[k] += float(v or 0.0)

            # --- 추적 (model.track과 같은 입력) ---
            t0 = time.perf_counter()
            det_np = r.boxes.cpu().numpy()
            trk = tracker.update(det_np, frame)
            timing["track_s"] += time.perf_counter() - t0
            idx_to_tid = {}
            if len(trk):
                for row in trk:
                    idx_to_tid[int(row[-1])] = int(row[4])

            # --- 기록·근거 수집·결과 영상 ---
            t0 = time.perf_counter()
            n = len(r.boxes)
            xyxy = r.boxes.xyxy.cpu().numpy() if n else np.zeros((0, 4))
            confs = r.boxes.conf.cpu().numpy() if n else np.zeros((0,))
            clss = r.boxes.cls.cpu().numpy().astype(int) if n else np.zeros((0,), int)
            masks = None
            if n and r.masks is not None:
                masks = r.masks.data.cpu().numpy() > 0.5
                if masks.shape[1:] != frame.shape[:2]:
                    # retina_masks=False 설정 등으로 해상도가 다르면 원본 크기로 보정
                    masks = np.stack(
                        [
                            cv2.resize(
                                m.astype(np.uint8),
                                (frame.shape[1], frame.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )
                            > 0
                            for m in masks
                        ]
                    )
            dets = []
            vis = frame.copy() if writer is not None else None
            fh, fw = frame.shape[:2]
            for i in range(n):
                tid = idx_to_tid.get(i)
                box = [round(float(v), 1) for v in xyxy[i]]
                cid = int(clss[i])
                m = masks[i] if masks is not None else None
                rx1, ry1, rx2, ry2 = _box_region(box, fw, fh)
                mc = m[ry1:ry2, rx1:rx2] if m is not None else None
                area = (
                    int(mc.sum()) if mc is not None else None
                )  # 마스크는 박스 밖이 0으로 잘려 있음
                code = cls_to_code.get(cid)
                dets.append(
                    {
                        "det_idx": i,
                        "class_id": cid,
                        "class_name": names[cid],
                        "category": code,
                        "conf": round(float(confs[i]), 4),
                        "box_xyxy": box,
                        "track_id": tid,
                        "mask_area_px": area,
                        "mask_polygons": _mask_polygons(mc, (rx1, ry1)) if mc is not None else None,
                    }
                )
                n_det_total += 1
                if tid is not None:
                    n_det_tracked += 1
                    ts = tracks.get(tid)
                    if ts is None:
                        ts = tracks[tid] = TrackState(track_id=tid)
                    ts.class_conf[cid] += float(confs[i])
                    ts.confs.append(float(confs[i]))
                    ts.frames.append(fidx)
                    ts.times.append(t_sec)
                    if ts.first_box is None:
                        ts.first_box = box
                        ts.first_crop = _crop(frame, box)
                    ts.last_box = box
                    ts.last_crop = _crop(frame, box, max_side=256)
                    color = _color(tid)
                    _offer_rep(
                        ts,
                        lambda: (_crop(frame, box), _thumb_with_box(frame, box, m, color)),
                        float(confs[i]),
                        t_sec,
                        fidx,
                        rep_k,
                        rep_gap,
                    )
                    ts.hist_counter += 1
                    if ts.hist_counter % ts.hist_stride == 0 and mc is not None:
                        hist = masked_hsv_hist(frame[ry1:ry2, rx1:rx2], mc)
                        if hist is not None:
                            ts.hists.append(hist)
                            if len(ts.hists) > hist_cap:
                                ts.hists = ts.hists[::2]
                                ts.hist_stride *= 2
                if vis is not None:
                    color = _color(tid)
                    if mc is not None:
                        sub = vis[ry1:ry2, rx1:rx2]
                        sub[mc] = (sub[mc] * 0.55 + np.array(color) * 0.45).astype(np.uint8)
                    x1, y1, x2, y2 = [int(v) for v in box]
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
                    label = (
                        f"{'#' + str(tid) if tid is not None else '-'} {names[cid]} {confs[i]:.2f}"
                    )
                    cv2.putText(
                        vis,
                        label,
                        (x1 + 4, max(30, y1 + 34)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.1,
                        (0, 0, 0),
                        6,
                    )
                    cv2.putText(
                        vis,
                        label,
                        (x1 + 4, max(30, y1 + 34)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.1,
                        color,
                        2,
                    )
            rec = {"frame_idx": fidx, "time_sec": round(t_sec, 4), "n_det": n, "detections": dets}
            jsonl_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if masks is not None and t_sec + 1e-6 >= next_mask_t:
                inst = np.zeros(frame.shape[:2], dtype=np.uint16)
                for i in range(n):
                    inst[masks[i] & (inst == 0)] = i + 1
                stem = f"f{fidx:06d}"
                imwrite(run_dir / "mask_samples" / f"{stem}_instances.png", inst)
                ov = frame.copy()
                for i in range(n):
                    c = _color(idx_to_tid.get(i))
                    ov[masks[i]] = (ov[masks[i]] * 0.5 + np.array(c) * 0.5).astype(np.uint8)
                imwrite(
                    run_dir / "mask_samples" / f"{stem}_overlay.jpg",
                    cv2.resize(ov, (ov.shape[1] // 2, ov.shape[0] // 2)),
                    [cv2.IMWRITE_JPEG_QUALITY, 85],
                )
                mask_index.append(
                    {
                        "frame_idx": fidx,
                        "time_sec": round(t_sec, 4),
                        "instances_png": f"mask_samples/{stem}_instances.png",
                        "overlay_jpg": f"mask_samples/{stem}_overlay.jpg",
                        "pixel_value_to_det": {
                            str(i + 1): {
                                "det_idx": i,
                                "class_name": names[int(clss[i])],
                                "track_id": idx_to_tid.get(i),
                            }
                            for i in range(n)
                        },
                    }
                )
                next_mask_t = t_sec + mask_every

            if writer is not None:
                cv2.putText(
                    vis,
                    f"{run_label} f{fidx} {t_sec:.2f}s",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.6,
                    (0, 0, 0),
                    8,
                )
                cv2.putText(
                    vis,
                    f"{run_label} f{fidx} {t_sec:.2f}s",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.6,
                    (255, 255, 255),
                    3,
                )
                writer.write(cv2.resize(vis, (ow, oh), interpolation=cv2.INTER_AREA))
            timing["record_and_save_s"] += time.perf_counter() - t0

            if n_proc % 50 == 0:
                el = time.perf_counter() - t_loop
                log(
                    f"  처리 {n_proc}프레임 (원본 {fidx + 1}) | {t_sec:.1f}s 지점 | "
                    f"경과 {el:.1f}s | 활성 트랙 누적 {len(tracks)}"
                )
    finally:
        cap.release()
        cleanup_temp(tmp_video)
        jsonl_f.close()
        if writer is not None:
            writer.release()
    timing["frame_loop_total_s"] = time.perf_counter() - t_loop

    # ---------- 트랙 → 물품 후보 ----------
    t0 = time.perf_counter()
    track_rows, item_rows, feat = [], [], {}
    for tid in sorted(tracks):
        ts = tracks[tid]
        cid = ts.class_conf.most_common(1)[0][0]
        code = cls_to_code.get(cid)
        nf = len(ts.frames)
        mc = float(np.mean(ts.confs))
        pid = f"T{tid:05d}"
        if code is None or code not in listed_codes:
            status, reason, listed = "not_listed", "목록 대상 품목이 아님(미매핑 클래스)", False
        elif nf >= list_cfg["confirm_min_frames"] and mc >= list_cfg["confirm_min_mean_conf"]:
            status, reason, listed = "confirmed", "", True
        elif nf >= list_cfg["review_min_frames"] and mc >= list_cfg["review_min_mean_conf"]:
            status, reason, listed = (
                "needs_review",
                f"짧거나 낮은 신뢰 트랙(n={nf}, mean_conf={mc:.2f})",
                True,
            )
        else:
            status, reason, listed = "dropped", f"확인 기준 미달(n={nf}, mean_conf={mc:.2f})", False
        votes = {names[c]: round(v, 3) for c, v in ts.class_conf.most_common()}
        track_rows.append(
            {
                "track_id": tid,
                "pred_item_id": pid if listed else "",
                "status": status,
                "status_reason": reason,
                "listed": listed,
                "category": code or "",
                "model_class": names[cid],
                "class_votes": json.dumps(votes, ensure_ascii=False),
                "n_frames": nf,
                "mean_conf": round(mc, 4),
                "max_conf": round(max(ts.confs), 4),
                "first_frame": ts.frames[0],
                "last_frame": ts.frames[-1],
                "first_time_sec": round(ts.times[0], 3),
                "last_time_sec": round(ts.times[-1], 3),
                "first_box": json.dumps(ts.first_box),
                "last_box": json.dumps(ts.last_box),
            }
        )
        if not listed:
            continue
        idir = run_dir / "items" / pid
        rep_paths = []
        reps = sorted(ts.reps, key=lambda r: r.time_sec)
        for j, rp in enumerate(reps, start=1):
            imwrite(idir / f"rep{j}_crop.jpg", rp.crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            imwrite(
                idir / f"rep{j}_frame{rp.frame_idx:06d}_{rp.time_sec:.2f}s.jpg",
                rp.full,
                [cv2.IMWRITE_JPEG_QUALITY, 85],
            )
            rep_paths.append(f"items/{pid}/rep{j}_frame{rp.frame_idx:06d}_{rp.time_sec:.2f}s.jpg")
        best = max(ts.reps, key=lambda r: r.conf)
        imwrite(
            run_dir / "tracks" / f"{pid}_first.jpg", ts.first_crop, [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        imwrite(
            run_dir / "tracks" / f"{pid}_last.jpg", ts.last_crop, [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        item_rows.append(
            {
                "video_id": video_id,
                "run_id": run_dir.name,
                "pred_item_id": pid,
                "status": status,
                "category": code,
                "category_ko": taxonomy.ko(code),
                "model_class": names[cid],
                "box_fit": AI1_BOX_FIT,
                "n_frames": nf,
                "mean_conf": round(mc, 4),
                "max_conf": round(max(ts.confs), 4),
                "first_frame": ts.frames[0],
                "last_frame": ts.frames[-1],
                "first_time_sec": round(ts.times[0], 3),
                "last_time_sec": round(ts.times[-1], 3),
                "rep_frame_idx": best.frame_idx,
                "rep_time_sec": round(best.time_sec, 3),
                "rep_images": ";".join(rep_paths),
                "track_ids": str(tid),
                "track_n_frames": str(nf),
                "note": reason,
            }
        )
        feat[pid] = {
            "hists": np.stack(ts.hists) if ts.hists else np.zeros((0, 1), np.float32),
            "frames": np.asarray(ts.frames, dtype=np.int32),
            "times": np.asarray(ts.times, dtype=np.float32),
        }
    write_csv(run_dir / "tracks.csv", track_rows, TRACK_FIELDS)
    write_csv(run_dir / "items.csv", item_rows, ITEM_FIELDS)
    write_csv(
        run_dir / "category_support.csv",
        support,
        ["category", "ko", "supported", "model_class", "mapping_mode", "note"],
    )
    np.savez_compressed(
        run_dir / "track_features.npz",
        **{f"{pid}__{k}": v for pid, d in feat.items() for k, v in d.items()},
    )
    write_json(run_dir / "mask_samples" / "index.json", mask_index)
    make_contact_sheet(run_dir, item_rows)
    timing["listing_s"] = time.perf_counter() - t0

    # ---------- manifest ----------
    duration = None
    try:
        duration = float((vinfo.ffprobe or {}).get("format", {}).get("duration"))
    except (TypeError, ValueError):
        pass
    counts = Counter(r["status"] for r in track_rows)
    item_counts = Counter((r["status"], r["category"]) for r in item_rows)
    manifest = {
        "run_id": run_dir.name,
        "run_label": run_label,
        "created_at": now_iso(),
        "command": " ".join([Path(sys.executable).name] + argv),
        "video": {
            "video_id": video_id,
            "path": str(video),
            "filename": video.name,
            "sha256": video_sha,
            "size_bytes": video.stat().st_size,
            "fps_reported": vinfo.fps,
            "frame_count_reported": vinfo.frame_count_reported,
            "frames_decoded": n_read,
            "frames_processed": n_proc,
            "vid_stride": stride,
            "duration_sec_ffprobe": duration,
            "decoded_width": vinfo.decoded_width,
            "decoded_height": vinfo.decoded_height,
            "orientation_meta_deg": vinfo.orientation_meta_deg,
            "orientation_auto_applied": vinfo.orientation_auto,
            "opened_via": vinfo.opened_via,
            "ffprobe": vinfo.ffprobe,
            "notes": vinfo.notes,
            "max_frames_limit": max_frames,
        },
        "model": {
            "weights_path": str(weights),
            "weights_sha256": sha256_file(weights),
            "task": model.task,
            "num_classes": len(names),
            "class_mapping_mode": map_mode,
            "class_to_category": {names[k]: v for k, v in cls_to_code.items()},
            "ultralytics_version": ultralytics.__version__,
        },
        "detector": det_cfg,
        "tracker": {
            "config_path": str(tpath),
            "config_sha256": sha256_text(tracker_text),
            "config": YAML.load(str(tpath)),
            "invocation": "predict(conf=detector.conf) → tracker.update(boxes.cpu().numpy(), "
            "frame) (model.track과 동일 입력)",
        },
        "listing": list_cfg,
        "listed_categories": sorted(listed_codes),
        "evidence": ev_cfg,
        "pipeline_config_path": str(cfg_path),
        "pipeline_config_sha256": sha256_file(cfg_path),
        "device": device,
        "environment": environment_info(),
        "code_sha256": code_fingerprint(),
        **git_info(),
        "timing": {k: round(v, 4) for k, v in timing.items()},
        "ultralytics_speed_ms_sum": {k: round(v, 2) for k, v in speed_acc.items()},
        "counts": {
            "detections_total": n_det_total,
            "detections_with_track_id": n_det_tracked,
            "tracks_total": len(tracks),
            "tracks_by_status": dict(counts),
            "items_by_status_category": {
                f"{s}|{c}": v for (s, c), v in sorted(item_counts.items())
            },
        },
        "unsupported_categories": [r["category"] for r in support if r["supported"] != "yes"],
        "outputs": {
            "frames_jsonl": "frames.jsonl",
            "items_csv": "items.csv",
            "tracks_csv": "tracks.csv",
            "annotated_video": out_video_path.name if out_video_path else None,
            "mask_samples": "mask_samples/",
            "item_images": "items/",
            "track_features": "track_features.npz",
            "category_support": "category_support.csv",
        },
        "notes": notes,
    }
    if n_proc and timing["frame_loop_total_s"] > 0:
        manifest["timing"]["processed_fps"] = round(n_proc / timing["frame_loop_total_s"], 3)
    write_json(run_dir / "run_manifest.json", manifest)
    return manifest
