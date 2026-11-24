# ARKit 촬영과 크기 추정

## iPhone에서 촬영

기존 iOS 앱 또는 새 iOS 앱의 Xcode 프로젝트에 `capture/ios`를 로컬 Swift Package로
추가하고 `JimssakCapture`를 대상에 연결한다. ARKit 월드 트래킹을 지원하는 iOS 16+
실기기를 사용한다. 앱의 Info.plist에는 `NSCameraUsageDescription`을 설정한다.

카메라 권한을 받은 뒤 `ARVideoRecorder` 인스턴스를 유지한다.

```swift
import AVFoundation
import JimssakCapture

// View controller 등에서 유지하는 프로퍼티
let recorder = ARVideoRecorder()

// 녹화 시작 이벤트
if await AVCaptureDevice.requestAccess(for: .video) {
    let folder = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try recorder.start(in: folder, pixelRotationCCW: 0)
}

// 녹화 종료 이벤트
recorder.stop { result in
    // 성공하면 result에 capture.mov와 camera.json이 있는 폴더가 들어 있다.
    // 앱에서 두 파일을 함께 공유하거나 측정 시스템으로 전달한다.
}
```

녹화 중 기기 화면 방향을 유지한다. `pixelRotationCCW`는 원본 센서 영상을 모델과
주석에 맞게 똑바로 세우는 반시계 90° 회전 횟수다. 0을 쓰는 경우 원본 센서 이미지가
똑바로 보이는 방향으로 촬영한다. 가구의 바닥 접점과 가로·깊이의 양 끝이 보이도록
가구 주위를 걸으며 찍는다. 여러 영상을 찍을 때 각 녹화를 독립된 폴더에 둔다.

- `capture.mov`는 카메라 원본 픽셀을 회전 태그 없이 저장한다.
- `camera.json`에는 영상에 기록된 프레임만 순서대로 포함된다. 프레임마다
  0부터 시작하는 `frame_index`, 최초 프레임 기준 타임스탬프(µs),
  카메라→월드 자세·위치 행렬(미터), 카메라 내부 행렬(픽셀)을 저장한다.
  두 행렬은 JSON에서 행 우선이다.
- 카메라 자세 추적이 불충분하거나 인코더가 준비되지 않은 프레임은 영상과 자세
  파일에서 함께 건너뛴다. 세션이 중단되거나 ARKit이 위치를 다시 잡아야 하는 경우
  녹화는 오류로 종료한다. 편집을 거치지 않은 영상과 같은 폴더의 JSON을 넘긴다.

카메라 자세와 영상은 같은 ARFrame에서 취한다.
[Apple ARFrame](https://developer.apple.com/documentation/arkit/arframe)
[Apple ARCamera](https://developer.apple.com/documentation/arkit/arcamera)

## 치수 계산

`observations.json` 예시. 좌표는 회전을 적용한 뒤 아래 이미지 크기로 맞춘 픽셀이다.
프레임 번호는 저장된 MOV의 디코딩 순서 기준이다. 같은 가구에는 같은 `track_id`를 쓴다.

```json
{
  "annotation_size_px": [960, 540],
  "views": [
    {
      "frame_index": 28,
      "track_id": "table-1",
      "label": "table",
      "mask_polygon_px": [[90, 112], [251, 86], [389, 157], [355, 341], [112, 323]]
    }
  ]
}
```

마스크를 여러 프레임에서 넣는다. 수동 `bbox`도 받을 수 있으며 기존 방식대로 프레임별
근사 치수만 산출한다. 3D 시점 결합에는 가구 전체 마스크 세 장 이상과 충분한 카메라
이동이 필요하다.

```bash
python -m ai.measurement.arkit_capture /path/to/camera.json /path/to/observations.json \
  --output outputs/arkit-job.json
python -m ai.measurement.moving_video outputs/arkit-job.json \
  --model /path/to/metric-depth-checkpoint --device auto --output outputs/arkit-results.json
```

첫 명령에 `--compare`를 붙이면 동일한 영상·마스크·내부 행렬·중력 방향으로 RGB 자세
추정과 기록된 ARKit 자세를 비교하는 작업을 함께 생성한다. 가져오기 과정에서 프레임
개수와 원본 해상도를 검사하고, ARKit 좌표를 크기 추정 모듈의 좌표계로 바꾼다. 회전과
해상도 변경을 각 프레임의 내부 행렬에도 적용한다.

## 비교 검증

실제 [ARKitScenes](https://machinelearning.apple.com/research/arkitscenes) 촬영에서
원본 RGB 프레임과 기록된 카메라 위치를 위 형식으로 재구성해, Mac GPU의 동일한
Depth Anything V2 체크포인트로 측정했다. 가구 치수는 ARKitScenes 3D 주석을 채점에만
사용했다. 이번 검증은 새 iPhone 녹화에서 얻은 파일의 실기기 검증과 구분된다.

| 가구 | 기준 가로·깊이·높이(cm) | RGB 자세 절대오차(cm) | 기록된 ARKit 자세 절대오차(cm) |
|---|---|---|---|
| 나무 테이블 `48458732` | 74.0 / 49.5 / 42.6 | 42.3 / 16.2 / 4.6 | 10.4 / 2.7 / 8.7 |
| 원목 수납장 `47332886` | 72.1 / 35.9 / 89.9 | 10.8 / 6.1 / 5.0 | 5.4 / 16.0 / 20.1 |

테이블은 두 방식 모두 마스크 3장을 결합했다. 수납장은 두 방식에 같은 5장을 넣었고,
RGB 자세 추정은 위치가 이어진 3장, ARKit 자세는 5장을 결합했다. 가로 오차는 두
가구에서 감소했지만, 수납장의 깊이와 두 가구의 높이 오차는 증가했다.

수납장의 동일한 마지막 3장만 골라 다시 비교해도 같은 경향이었다. RGB 방식의 오차는
10.8 / 6.1 / 5.0cm였고, ARKit 위치를 사용한 오차는 8.3 / 12.6 / 23.2cm였다.
