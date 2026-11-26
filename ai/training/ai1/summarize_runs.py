#!/usr/bin/env python
"""처리 시간·비용 요약: run_manifest.json들을 모아 버전(label)별 편당 시간 중앙값과 범위를 낸다.

예) python summarize_runs.py --runs runs/B0/* runs/B1/* --out evaluations/timing_summary.csv
    python summarize_runs.py --runs runs/B0/* --gpu-hourly-cost 1500 --cost-basis "가정: 영상 처리
    시간만 초 단위 과금"

비용은 단가와 '실제로 과금된다고 가정한 시간'을 명시할 때만 환산한다(초기 로딩 포함 여부도 표시).
연구실 이용료·상시 서버 유휴 비용은 섞지 않는다. 단가를 주지 않으면 '미측정'.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from jimssak_ai1.common import fail, log, read_json, write_csv, write_json  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="실행별 처리 시간 요약")
    ap.add_argument("--runs", nargs="+", type=Path, required=True, help="infer.py 결과 폴더들")
    ap.add_argument("--out", type=Path, required=True, help="요약 CSV 경로 (같은 이름 JSON도 저장)")
    ap.add_argument(
        "--gpu-hourly-cost",
        type=float,
        default=None,
        help="GPU 시간당 단가 (통화 단위는 --currency)",
    )
    ap.add_argument("--currency", default="KRW")
    ap.add_argument("--cost-basis", default=None, help="과금 가정 설명 (필수: 단가를 줄 때)")
    ap.add_argument(
        "--include-load-in-cost", action="store_true", help="초기 로딩 시간도 과금 시간에 포함"
    )
    args = ap.parse_args(argv)
    if args.gpu_hourly_cost is not None and not args.cost_basis:
        fail("--gpu-hourly-cost를 줄 때는 --cost-basis로 과금 가정을 명시하세요")

    rows = []
    for rd in args.runs:
        mp = rd / "run_manifest.json"
        if not mp.is_file():
            continue
        m = read_json(mp)
        t, v, env = m["timing"], m["video"], m["environment"]
        dev_name = (
            (env.get("cuda_devices") or ["CPU"])[0] if m["device"].startswith("cuda") else "CPU"
        )
        process_s = t["frame_loop_total_s"] + t.get("listing_s", 0)
        billed = process_s + (t["model_load_and_warmup_s"] if args.include_load_in_cost else 0)
        rows.append(
            {
                "label": m["run_label"],
                "run_id": m["run_id"],
                "video_id": v["video_id"],
                "device": m["device"],
                "device_name": dev_name,
                "video_duration_sec": v.get("duration_sec_ffprobe"),
                "frames_processed": v["frames_processed"],
                "vid_stride": v["vid_stride"],
                "model_load_s": t["model_load_and_warmup_s"],
                "decode_s": t["decode_s"],
                "predict_s": t["predict_s"],
                "track_s": t["track_s"],
                "record_save_s": t["record_and_save_s"],
                "listing_s": t.get("listing_s"),
                "video_process_total_s": round(process_s, 3),
                "processed_fps": t.get("processed_fps"),
                "cost": round(billed / 3600 * args.gpu_hourly_cost, 2)
                if args.gpu_hourly_cost is not None
                else "미측정",
                "cost_basis": args.cost_basis or "",
                "max_frames_limit": v.get("max_frames_limit"),
            }
        )
    if not rows:
        fail("run_manifest.json이 있는 실행 폴더가 없습니다")
    by = defaultdict(list)
    for r in rows:
        if r["max_frames_limit"] in (None, "", "None"):
            by[(r["label"], r["device_name"])].append(r["video_process_total_s"])
    summary = [
        {
            "label": k[0],
            "device_name": k[1],
            "n_videos": len(v),
            "per_video_median_s": round(statistics.median(v), 2),
            "per_video_min_s": round(min(v), 2),
            "per_video_max_s": round(max(v), 2),
        }
        for k, v in sorted(by.items())
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, rows, list(rows[0].keys()))
    write_json(
        args.out.with_suffix(".json"),
        {
            "per_run": rows,
            "per_label": summary,
            "note": "스모크(--max-frames) 실행은 편당 통계에서 제외. CPU 시간은 서버 GPU 시간이 "
            "아님.",
        },
    )
    for s in summary:
        log(
            f"{s['label']} [{s['device_name']}] 영상 {s['n_videos']}편 | 편당 처리 "
            f"중앙값 {s['per_video_median_s']}s "
            f"(범위 {s['per_video_min_s']}–{s['per_video_max_s']}s)"
        )
    log(f"저장: {args.out}")


if __name__ == "__main__":
    main()
