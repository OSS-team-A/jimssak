#!/usr/bin/env python
"""영상 1편에 대해 YOLO 분할 + BoT-SORT 추론을 실행하고 실행별 결과 폴더를 만든다.

예) B0 기준선 (사전 학습 YOLO11n-seg + 기본 BoT-SORT)
    python -u infer.py --video /workspace/jimssak_data/videos/언니방영상.mp4 --video-id
    unnie_room_01 \
        --label B0 --weights weights/yolo11n-seg.pt --device cuda:0

결과: <output-root>/<label>/<run-name>/  (기존 폴더가 있으면 덮어쓰지 않고 실패)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from jimssak_ai1.common import (  # noqa: E402
    PROJECT_ROOT,
    fail,
    load_yaml,
    log,
    make_run_dir,
    now_stamp,
    resolve_device,
    safe_id,
    set_offline_friendly_ultralytics_env,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="짐싹 AI1 영상 추론: 프레임별 클래스·confidence·박스·인스턴스 마스크·임시 "
        "track_id 기록, "
        "결과 영상·마스크 샘플·개별 운반 후보 목록 저장.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", required=True, type=Path, help="입력 영상 경로 (한글 파일명 가능)")
    p.add_argument(
        "--video-id",
        default=None,
        help="정답 CSV의 video_id와 같게 맞출 영상 식별자. 생략하면 파일명에서 ASCII 식별자를 "
        "만든다",
    )
    p.add_argument(
        "--label", default="B0", help="비교용 버전 이름 (B0/B1 등). 결과 폴더 1단계 이름"
    )
    p.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_ROOT / "weights" / "yolo11n-seg.pt",
        help="YOLO 분할 가중치(.pt). 기본 경로에 없고 파일명이 yolo11n-seg.pt면 Ultralytics 공식 "
        "배포본을 내려받는다",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pipeline_default.yaml",
        help="검출·추적·목록 집계 설정 YAML",
    )
    p.add_argument("--taxonomy", type=Path, default=PROJECT_ROOT / "configs" / "taxonomy.yaml")
    p.add_argument(
        "--tracker-config",
        type=Path,
        default=None,
        help="BoT-SORT 설정 YAML. 생략 시 설치된 Ultralytics 기본 botsort.yaml",
    )
    p.add_argument(
        "--device",
        default="cuda:0",
        help="cuda:0 또는 cpu. GPU가 없는데 cuda를 주면 실패한다(자동 CPU 전환 없음)",
    )
    p.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs", help="결과 상위 폴더")
    p.add_argument("--run-name", default=None, help="결과 폴더 이름. 기본: <시각>_<video-id>")
    p.add_argument(
        "--overwrite", action="store_true", help="같은 이름의 결과 폴더가 있어도 덮어쓴다"
    )
    p.add_argument(
        "--max-frames", type=int, default=None, help="스모크 테스트용: 처리할 최대 프레임 수"
    )
    p.add_argument(
        "--vid-stride", type=int, default=None, help="설정 파일의 video.vid_stride 덮어쓰기"
    )
    p.add_argument(
        "--no-video", action="store_true", help="결과 영상(annotated.mp4)을 저장하지 않는다"
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_offline_friendly_ultralytics_env()
    from jimssak_ai1.pipeline import run_inference
    from jimssak_ai1.taxonomy import load_taxonomy

    video = args.video.expanduser().resolve()
    if not video.is_file():
        fail(f"영상 파일이 없습니다: {video}")
    cfg = load_yaml(args.config)
    if args.vid_stride is not None:
        if args.vid_stride < 1:
            fail("--vid-stride는 1 이상이어야 합니다")
        cfg.setdefault("video", {})["vid_stride"] = args.vid_stride
    taxonomy = load_taxonomy(args.taxonomy)
    device = resolve_device(args.device)

    weights = args.weights.expanduser()
    if not weights.is_file():
        if weights.name.startswith(("yolo11", "yolov8", "yolo26")) and weights.suffix == ".pt":
            from ultralytics.utils.downloads import attempt_download_asset

            weights.parent.mkdir(parents=True, exist_ok=True)
            log(f"가중치가 없어 Ultralytics 공식 배포본을 내려받습니다: {weights.name} → {weights}")
            weights = Path(attempt_download_asset(str(weights)))
        if not weights.is_file():
            fail(f"가중치 파일이 없습니다: {weights}")
    weights = weights.resolve()

    video_id = args.video_id or safe_id(video.stem)
    run_name = args.run_name or f"{now_stamp()}_{safe_id(video_id)}"
    run_dir = make_run_dir(
        args.output_root / safe_id(args.label), run_name, overwrite=args.overwrite
    )
    log(f"결과 폴더: {run_dir.resolve()}")

    manifest = run_inference(
        video=video,
        video_id=video_id,
        run_dir=run_dir,
        run_label=args.label,
        weights=weights,
        device=device,
        cfg=cfg,
        cfg_path=args.config.resolve(),
        taxonomy=taxonomy,
        tracker_cfg_path=args.tracker_config,
        max_frames=args.max_frames,
        save_video=not args.no_video,
        argv=sys.argv,
    )
    t = manifest["timing"]
    c = manifest["counts"]
    log("완료")
    log(
        f"  처리 프레임 {manifest['video']['frames_processed']} / "
        f"디코딩 {manifest['video']['frames_decoded']} | 장치 {device}"
    )
    log(
        f"  초기 로딩 {t['model_load_and_warmup_s']:.2f}s | 프레임 "
        f"루프 {t['frame_loop_total_s']:.2f}s "
        f"(디코딩 {t['decode_s']:.2f} / 검출 {t['predict_s']:.2f} / 추적 {t['track_s']:.2f} / "
        f"기록·저장 {t['record_and_save_s']:.2f})"
    )
    log(
        f"  검출 {c['detections_total']}건 (track_id 있음 {c['detections_with_track_id']}) | "
        f"트랙 {c['tracks_total']} | {c['tracks_by_status']}"
    )
    log(f"  미지원 품목(현재 모델): {', '.join(manifest['unsupported_categories'])}")
    log(f"  후보 목록: {run_dir / 'items.csv'}")
    log(f"  프레임 JSONL: {run_dir / 'frames.jsonl'}")
    if manifest["outputs"]["annotated_video"]:
        log(f"  결과 영상: {run_dir / manifest['outputs']['annotated_video']}")
    log(f"  manifest: {run_dir / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
