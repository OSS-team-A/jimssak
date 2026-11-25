#if os(iOS)
import ARKit
import AVFoundation
import CoreMedia
import CoreVideo
import Dispatch
import Foundation
import simd

/// A capture session owned by the iOS app. Keep one instance until stop completes.
public final class ARVideoRecorder: NSObject, ARSessionDelegate {
    public let session = ARSession()
    private let queue = DispatchQueue(label: "com.jimssak.arkit-recorder")
    private var take: Take?
    private var captureFailure: Error?
    private var finishing = false

    public enum RecordingError: Error {
        case unsupportedDevice, missingCameraPermission, alreadyRecording, invalidRotation
        case noFrames, formatChanged, writerFailed, interrupted, trackingReset, outputExists
    }

    private struct Frame: Encodable {
        let frame_index: Int
        let timestamp_us: Int64
        let camera_to_world: [[Float]]
        let intrinsics: [[Float]]
    }

    private struct Metadata: Encodable {
        let schema_version = 1
        let video_file = "capture.mov"
        let video_id: String
        let image_size_px: [Int]
        let pixel_rotation_ccw: Int
        let frames: [Frame]
    }

    private final class Take {
        let directory: URL
        let rotation: Int
        let id = UUID().uuidString
        var writer: AVAssetWriter?
        var adaptor: AVAssetWriterInputPixelBufferAdaptor?
        var size: [Int] = []
        var timestamp: TimeInterval?
        var frames: [Frame] = []

        init(directory: URL, rotation: Int) {
            self.directory = directory
            self.rotation = rotation
        }
    }

    public override init() {
        super.init()
        session.delegate = self
        session.delegateQueue = queue
    }

    /// Record native camera pixels. Specify a fixed quarter-turn count for later viewing.
    /// Call from outside the session's delegate queue, after camera permission is granted.
    public func start(in directory: URL, pixelRotationCCW: Int = 0) throws {
        try queue.sync {
            guard take == nil, !finishing else { throw RecordingError.alreadyRecording }
            guard ARWorldTrackingConfiguration.isSupported else {
                throw RecordingError.unsupportedDevice
            }
            guard AVCaptureDevice.authorizationStatus(for: .video) == .authorized else {
                throw RecordingError.missingCameraPermission
            }
            guard (0...3).contains(pixelRotationCCW) else { throw RecordingError.invalidRotation }
            guard !FileManager.default.fileExists(atPath: directory.path) else {
                throw RecordingError.outputExists
            }
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
            captureFailure = nil
            take = Take(directory: directory, rotation: pixelRotationCCW)
            let configuration = ARWorldTrackingConfiguration()
            configuration.worldAlignment = .gravity
            session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
        }
    }

    /// Returns the folder only after both the movie and JSON have been finalized.
    public func stop(completion: @escaping (Result<URL, Error>) -> Void) {
        queue.async {
            guard let take = self.take else {
                self.deliver(.failure(self.captureFailure ?? RecordingError.noFrames), completion)
                return
            }
            self.take = nil
            self.session.pause()
            guard let writer = take.writer, let adaptor = take.adaptor, !take.frames.isEmpty else {
                take.writer?.cancelWriting()
                self.deliver(.failure(RecordingError.noFrames), completion)
                return
            }
            self.finishing = true
            adaptor.assetWriterInput.markAsFinished()
            writer.finishWriting {
                self.queue.async {
                    self.finishing = false
                    do {
                        guard writer.status == .completed else {
                            throw writer.error ?? RecordingError.writerFailed
                        }
                        let metadata = Metadata(
                            video_id: take.id, image_size_px: take.size,
                            pixel_rotation_ccw: take.rotation, frames: take.frames
                        )
                        let data = try JSONEncoder().encode(metadata)
                        try data.write(
                            to: take.directory.appendingPathComponent("camera.json"), options: .atomic
                        )
                        self.deliver(.success(take.directory), completion)
                    } catch {
                        self.deliver(.failure(error), completion)
                    }
                }
            }
        }
    }

    public func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard let take else { return }
        switch frame.camera.trackingState {
        case .normal: break
        case .limited(.relocalizing):
            fail(RecordingError.trackingReset)
            return
        default: return
        }
        do {
            try append(frame, to: take)
        } catch {
            fail(error)
        }
    }

    public func sessionWasInterrupted(_ session: ARSession) {
        fail(RecordingError.interrupted)
    }

    public func session(_ session: ARSession, didFailWithError error: Error) {
        fail(error)
    }

    private func append(_ frame: ARFrame, to take: Take) throws {
        let image = frame.capturedImage
        let size = [CVPixelBufferGetWidth(image), CVPixelBufferGetHeight(image)]
        if take.writer == nil {
            let writer = try AVAssetWriter(
                outputURL: take.directory.appendingPathComponent("capture.mov"), fileType: .mov
            )
            let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: size[0], AVVideoHeightKey: size[1],
                AVVideoCompressionPropertiesKey: [AVVideoAllowFrameReorderingKey: false],
            ])
            input.expectsMediaDataInRealTime = true
            let adaptor = AVAssetWriterInputPixelBufferAdaptor(
                assetWriterInput: input,
                sourcePixelBufferAttributes: [
                    kCVPixelBufferPixelFormatTypeKey as String: CVPixelBufferGetPixelFormatType(image),
                    kCVPixelBufferWidthKey as String: size[0],
                    kCVPixelBufferHeightKey as String: size[1],
                ]
            )
            guard writer.canAdd(input) else { throw RecordingError.writerFailed }
            writer.add(input)
            guard writer.startWriting() else { throw writer.error ?? RecordingError.writerFailed }
            writer.startSession(atSourceTime: .zero)
            take.writer = writer
            take.adaptor = adaptor
            take.size = size
        }
        let cameraSize = frame.camera.imageResolution
        guard take.size == size, Int(cameraSize.width) == size[0],
              Int(cameraSize.height) == size[1] else {
            throw RecordingError.formatChanged
        }
        if let writer = take.writer, writer.status == .failed {
            throw writer.error ?? RecordingError.writerFailed
        }
        guard let adaptor = take.adaptor, adaptor.assetWriterInput.isReadyForMoreMediaData else {
            return
        }
        let origin = take.timestamp ?? frame.timestamp
        let timestamp = Int64(((frame.timestamp - origin) * 1_000_000).rounded())
        if let previous = take.frames.last, timestamp <= previous.timestamp_us { return }
        let presentationTime = CMTime(value: timestamp, timescale: 1_000_000)
        guard adaptor.append(image, withPresentationTime: presentationTime) else {
            throw take.writer?.error ?? RecordingError.writerFailed
        }
        take.timestamp = origin
        let pose = frame.camera.transform
        let k = frame.camera.intrinsics
        take.frames.append(Frame(
            frame_index: take.frames.count, timestamp_us: timestamp,
            camera_to_world: (0..<4).map { row in (0..<4).map { pose[$0][row] } },
            intrinsics: (0..<3).map { row in (0..<3).map { k[$0][row] } }
        ))
    }

    private func fail(_ error: Error) {
        guard let take else { return }
        take.writer?.cancelWriting()
        captureFailure = error
        self.take = nil
        session.pause()
    }

    private func deliver(_ result: Result<URL, Error>, _ reply: @escaping (Result<URL, Error>) -> Void) {
        DispatchQueue.main.async { reply(result) }
    }
}
#endif
