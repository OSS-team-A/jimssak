#!/usr/bin/env python
"""발표용 종합 비교표: evaluate_list.py 결과들(+ 처리 시간 요약)을 한 표로 모은다.

예) python compare_runs.py --eval B0=evaluations/<B0 평가> B1=... B2=... ChatGPT=... \
        --timing evaluations/timing_summary.json --video-rooms annotations/video_rooms.csv --out
        evaluations/comparison

표 열 순서(고정): 평가 영상 수 / 방 수 / 정답 물품 수 → 누락(FN) → 확인된 중복 → 잘못된 병합 → 수량
MAE → 목록 F1
                → 편당 처리 시간 → 비용(실측/추정/미측정)
원자료가 없는 칸은 '미측정', 근거가 없어 계산할 수 없는 칸은 '판정 불가'로 둔다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from jimssak_ai1.common import fail, log, read_csv, read_json, write_csv  # noqa: E402

COLS = [
    "버전",
    "비교 범위",
    "평가 영상 수",
    "평가 방 수",
    "정답 물품 수(대상)",
    "누락(FN)",
    "확인된 중복",
    "잘못된 병합",
    "수량 MAE",
    "목록 P",
    "목록 R",
    "목록 F1",
    "확인 요청",
    "편당 처리 시간(중앙값, 범위)",
    "비용",
]


def main(argv=None):
    ap = argparse.ArgumentParser(description="발표용 종합 비교표")
    ap.add_argument("--eval", nargs="+", required=True, help="LABEL=evaluate_list 결과 폴더")
    ap.add_argument("--mode", default="list", choices=["list", "box"])
    ap.add_argument("--timing", type=Path, default=None, help="summarize_runs.py JSON")
    ap.add_argument(
        "--video-rooms", type=Path, default=None, help="video_id,room_id CSV (방 수 계산)"
    )
    ap.add_argument("--out", type=Path, required=True, help="출력 경로 접두어 (.csv, .md)")
    args = ap.parse_args(argv)

    rooms = (
        {r["video_id"]: r["room_id"] for r in read_csv(args.video_rooms)}
        if args.video_rooms
        else None
    )
    timing = {}
    if args.timing:
        for s in read_json(args.timing)["per_label"]:
            timing[s["label"]] = s
        cost_rows = read_json(args.timing)["per_run"]
    rows = []
    for spec in args.eval:
        if "=" not in spec:
            fail(f"--eval 형식은 LABEL=폴더: {spec}")
        label, d = spec.split("=", 1)
        sm = read_json(Path(d) / "summary.json")
        m = sm["modes"].get(args.mode, {"status": "미실행"})
        vids = sm["inputs"]["videos"]
        row = {
            "버전": label,
            "비교 범위": {"list": "물품 목록만(사람이 정한 대상)", "box": "박스 필터 포함"}[
                args.mode
            ],
            "평가 영상 수": len(vids),
            "평가 방 수": len({rooms.get(v, "?") for v in vids}) if rooms else "미기재",
        }
        if m.get("status") != "완료":
            row.update({k: m.get("status", "미실행") for k in COLS[4:13]})
        else:
            j = m.get("f1_judgeable_all_videos", True)
            row.update(
                {
                    "정답 물품 수(대상)": m["n_gt_target"],
                    "누락(FN)": m["fn"] if j else "판정 불가",
                    "확인된 중복": m["fp_duplicate"] if j else "판정 불가",
                    "잘못된 병합": m["wrong_merge"] if j else "판정 불가",
                    "수량 MAE": m["count_mae"],
                    "목록 P": m["precision"] if j else "판정 불가",
                    "목록 R": m["recall"] if j else "판정 불가",
                    "목록 F1": m["f1"] if j else "판정 불가",
                    "확인 요청": m["needs_review_total"],
                }
            )
        tlabel = (
            "B0" if label == "B2" and "B2" not in timing else label
        )  # B2는 B0/B1 추론 결과에 후처리만 추가
        t = timing.get(tlabel)
        row["편당 처리 시간(중앙값, 범위)"] = (
            f"{t['per_video_median_s']}s ({t['per_video_min_s']}–{t['per_video_max_s']}s, "
            f"{t['device_name']}, n={t['n_videos']})"
            if t
            else "미측정"
        )
        row["비용"] = "미측정"
        if args.timing and t:
            costs = [r for r in cost_rows if r["label"] == tlabel and r["cost"] != "미측정"]
            if costs:
                row["비용"] = f"추정 {costs[0]['cost_basis']}"
        if label.lower().startswith("chatgpt"):
            row["편당 처리 시간(중앙값, 범위)"] = "채팅창 직접 사용 (chatgpt_runs.csv 기록 참고)"
            row["비용"] = "API 편당 비용 미측정"
        rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out.with_suffix(".csv"), rows, COLS)
    md = ["| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
    md += ["| " + " | ".join(str(r.get(c, "")) for c in COLS) + " |" for r in rows]
    n_v = {r["평가 영상 수"] for r in rows}
    md.append("")
    md.append(
        "- B2 처리 시간은 같은 추론 결과에 재등장 연결 후처리만 더한 것(후처리 시간은 "
        "relink_manifest 참고)."
    )
    if max(n_v) <= 2:
        md.append(
            "- 독립 평가 영상·방 수가 적어 '파일럿에서 관찰한 차이'로만 해석한다. 통계적 유의성은 "
            "주장하지 않는다."
        )
    args.out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"저장: {args.out.with_suffix('.csv')} , {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
