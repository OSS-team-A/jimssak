#!/usr/bin/env python
"""컨테이너(환경 창)에서 실행: PyTorch·CUDA·GPU·OpenCV 영상 디코딩·YOLO 실제 실행을 확인한다.

    python scripts/check_env.py --require-gpu --video /workspace/jimssak_data/videos/언니방영상.mp4
    --yolo

장치 목록만 보지 않고 실제 CUDA 텐서 연산(행렬곱·conv)과 CPU 결과 비교를 수행한다.
GPU 확인이 실패하면 0이 아닌 코드로 종료한다(CPU 성공으로 바꾸지 않음).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="환경 확인: torch/CUDA/GPU 실제 연산, OpenCV 영상 디코딩, YOLO 1프레임 실행"
    )
    ap.add_argument(
        "--require-gpu", action="store_true", help="CUDA GPU가 없거나 연산이 실패하면 실패 종료"
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--video", type=Path, default=None, help="디코딩·회전 확인할 영상")
    ap.add_argument(
        "--yolo", action="store_true", help="weights/yolo11n-seg.pt로 1프레임 실제 추론"
    )
    ap.add_argument("--weights", type=Path, default=ROOT / "weights" / "yolo11n-seg.pt")
    ap.add_argument("--out", type=Path, default=None, help="결과 JSON 저장 경로")
    args = ap.parse_args(argv)

    from jimssak_ai1.common import environment_info, set_offline_friendly_ultralytics_env

    set_offline_friendly_ultralytics_env()
    report = {"ok": True, "checks": {}}

    def check(name, fn):
        t0 = time.perf_counter()
        try:
            res = fn()
            report["checks"][name] = {
                "ok": True,
                "result": res,
                "sec": round(time.perf_counter() - t0, 3),
            }
            print(f"[OK]   {name}: {res}", flush=True)
        except Exception as e:
            report["checks"][name] = {
                "ok": False,
                "error": repr(e),
                "trace": traceback.format_exc()[-1500:],
            }
            report["ok"] = False
            print(f"[FAIL] {name}: {e!r}", flush=True)

    report["environment"] = environment_info()
    env = report["environment"]
    print(
        f"python {env['python']} | torch {env.get('torch_version')} | "
        f"torch.version.cuda {env.get('torch_cuda_build')} "
        f"| cuda_available {env.get('cuda_available')} | "
        f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )
    print(f"packages: {env['packages']}", flush=True)

    import torch

    gpu_ok = torch.cuda.is_available()
    if gpu_ok:

        def gpu_tensor():
            dev = torch.device(args.device)
            idx = dev.index or 0
            name = torch.cuda.get_device_name(idx)
            cap = torch.cuda.get_device_capability(idx)
            a = torch.randn(1024, 1024, device=dev)
            b = torch.randn(1024, 1024, device=dev)
            c = a @ b
            torch.cuda.synchronize()
            err = (c.cpu() - a.cpu() @ b.cpu()).abs().max().item()
            x = torch.randn(2, 3, 64, 64, device=dev)
            conv = torch.nn.Conv2d(3, 8, 3).to(dev)
            y = conv(x)
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info(idx)
            if not err < 1e-1:
                raise RuntimeError(f"GPU 행렬곱 결과가 CPU와 다름 (max abs err {err})")
            return {
                "device": str(dev),
                "name": name,
                "capability": f"sm_{cap[0]}{cap[1]}",
                "matmul_max_abs_err": err,
                "conv_out": list(y.shape),
                "cudnn": torch.backends.cudnn.version(),
                "mem_free_GiB": round(free / 2**30, 2),
                "mem_total_GiB": round(total / 2**30, 2),
                "arch_list": torch.cuda.get_arch_list(),
            }

        check("cuda_tensor_ops", gpu_tensor)

        def tv_nms():
            import torchvision
            from torchvision.ops import nms

            boxes = torch.tensor(
                [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]],
                dtype=torch.float32,
                device=args.device,
            )
            keep = nms(boxes, torch.tensor([0.9, 0.8, 0.7], device=args.device), 0.5)
            return {"torchvision": torchvision.__version__, "nms_keep": keep.cpu().tolist()}

        check("torchvision_cuda_nms", tv_nms)
    else:
        msg = "CUDA GPU가 보이지 않음 (torch.cuda.is_available() == False)"
        report["checks"]["cuda_tensor_ops"] = {"ok": False, "error": msg}
        print(f"[{'FAIL' if args.require_gpu else 'WARN'}] cuda_tensor_ops: {msg}", flush=True)
        if args.require_gpu:
            report["ok"] = False

    def cv2_info():
        import cv2

        bi = cv2.getBuildInformation()
        ff = [ln.strip() for ln in bi.splitlines() if "FFMPEG" in ln]
        return {"cv2": cv2.__version__, "ffmpeg": ff[:2]}

    check("opencv", cv2_info)

    if args.video:

        def video_check():
            from jimssak_ai1.video_io import cleanup_temp, open_video

            cap, info, tmp = open_video(args.video)
            n = 0
            while n < 30:
                ok, _ = cap.read()
                if not ok:
                    break
                n += 1
            cap.release()
            cleanup_temp(tmp)
            if n == 0:
                raise RuntimeError("프레임을 하나도 디코딩하지 못함")
            return {
                "decoded_wh": [info.decoded_width, info.decoded_height],
                "fps": info.fps,
                "frames_reported": info.frame_count_reported,
                "orientation_meta": info.orientation_meta_deg,
                "first_frames_decoded": n,
                "notes": info.notes,
            }

        check("video_decode", video_check)

    if args.yolo:

        def yolo_check():
            import numpy as np
            from ultralytics import YOLO

            dev = args.device if gpu_ok else "cpu"
            if not args.weights.is_file():
                from ultralytics.utils.downloads import attempt_download_asset

                args.weights.parent.mkdir(parents=True, exist_ok=True)
                attempt_download_asset(str(args.weights))
            m = YOLO(str(args.weights))
            img = np.full((640, 640, 3), 114, np.uint8)
            if args.video:
                from jimssak_ai1.video_io import cleanup_temp, open_video

                cap, _, tmp = open_video(args.video)
                ok, fr = cap.read()
                cap.release()
                cleanup_temp(tmp)
                if ok:
                    img = fr
            t0 = time.perf_counter()
            r = m.predict(img, device=dev, verbose=False, retina_masks=True)[0]
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            return {
                "device": dev,
                "n_det": len(r.boxes),
                "classes": [m.names[int(c)] for c in r.boxes.cls[:10]],
                "first_call_sec": round(time.perf_counter() - t0, 3),
                "speed_ms": r.speed,
            }

        check("yolo_predict", yolo_check)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(f"결과 저장: {args.out}")
    print("전체 결과:", "OK" if report["ok"] else "FAIL", flush=True)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
