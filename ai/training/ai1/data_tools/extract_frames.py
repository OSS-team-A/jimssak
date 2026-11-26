#!/usr/bin/env python
"""라벨링용 프레임 추출: splits.csv의 세션 영상에서 일정 간격으로 프레임을 뽑는다 (회전 적용, 원본
해상도).

예) python -u data_tools/extract_frames.py --splits /workspace/jimssak_data/splits.csv \
        --out /workspace/jimssak_data/frames_v1 --every-sec 1.0 --max-per-video 60

출력: <out>/<split>/<session_id>_f000123.jpg, <out>/frames_manifest.csv
흐림이 심한 프레임은 라플라시안 분산으로 걸러낼 수 있다(--min-sharpness). 원본 영상·프레임은 Git에
올리지 않는다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jimssak_ai1.common import fail, imwrite, log, read_csv, write_csv  # noqa: E402
from jimssak_ai1.video_io import cleanup_temp, open_video  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="라벨링용 프레임 추출 (세션별, 분할 폴더 유지)")
    ap.add_argument("--splits", required=True, type=Path, help="make_split.py 결과")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--every-sec", type=float, default=1.0)
    ap.add_argument("--max-per-video", type=int, default=60)
    ap.add_argument(
        "--min-sharpness", type=float, default=0.0, help="라플라시안 분산 하한 (0=끄기)"
    )
    ap.add_argument("--splits-to-extract", default="train,val,test")
    args = ap.parse_args(argv)

    want = {s.strip() for s in args.splits_to_extract.split(",")}
    rows = [r for r in read_csv(args.splits) if r["split"] in want]
    if not rows:
        fail("추출할 세션이 없습니다")
    manifest = []
    for r in rows:
        cap, info, tmp = open_video(Path(r["video_path"]))
        next_t, idx, kept = 0.0, 0, 0
        try:
            while kept < args.max_per_video:
                ok, fr = cap.read()
                if not ok:
                    break
                t = (
                    cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    if cap.get(cv2.CAP_PROP_POS_MSEC) > 0
                    else idx / (info.fps or 30)
                )
                if t + 1e-6 >= next_t:
                    sharp = float(
                        cv2.Laplacian(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                    )
                    if sharp >= args.min_sharpness:
                        name = f"{r['session_id']}_f{idx:06d}.jpg"
                        imwrite(args.out / r["split"] / name, fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        manifest.append(
                            {
                                "file": f"{r['split']}/{name}",
                                "split": r["split"],
                                "session_id": r["session_id"],
                                "room_id": r["room_id"],
                                "group_id": r.get("group_id", ""),
                                "frame_idx": idx,
                                "time_sec": round(t, 3),
                                "width": fr.shape[1],
                                "height": fr.shape[0],
                                "sharpness": round(sharp, 1),
                            }
                        )
                        kept += 1
                        next_t = t + args.every_sec
                idx += 1
        finally:
            cap.release()
            cleanup_temp(tmp)
        log(f"  {r['session_id']} ({r['split']}): {kept}장")
    write_csv(
        args.out / "frames_manifest.csv",
        manifest,
        [
            "file",
            "split",
            "session_id",
            "room_id",
            "group_id",
            "frame_idx",
            "time_sec",
            "width",
            "height",
            "sharpness",
        ],
    )
    log(f"완료: {len(manifest)}장 → {args.out}")


if __name__ == "__main__":
    main()
