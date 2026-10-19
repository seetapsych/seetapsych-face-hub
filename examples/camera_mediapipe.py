"""Display MediaPipe face landmarks with an asynchronous mirrored camera preview."""

import os
import platform
import time
from typing import Any

import cv2
import numpy
from camera_window import CameraWindow, draw_text, fit_canvas, fit_geometry
from seetapsych_lib.runtime.factory import Factory
from seetapsych_lib.runtime.parallel.future import Future
from seetapsych_lib.runtime.parallel_runner import ParallelRunner as Runner
from seetapsych_lib.runtime.pipeline import Pipeline

override_modules = [
    os.path.join(os.path.dirname(__file__), "../seetapsych_face_hub/modules"),
    os.path.join(os.path.dirname(__file__), "../../seetapsych-lib/seetapsych_lib/modules"),
]
PACKAGES = [
    "56d8a9a9-0ad9-4d90-ad3f-ffc2f51c1597",  # FaceDetection-MediaPipe
    "8e646eec-e50f-4102-a658-1449d04296fb",  # FaceMesh-MediaPipe
]
WINDOW_NAME = "Camera MediaPipe Face Mesh"
LANDMARK_ATTRIBUTE = "face_mesh"
LANDMARK_FIELD = "normalized_3d_landmarks"
LANDMARK_DIMENSIONS = 3
NORMALIZED_COORDINATES = True


def open_camera(camera_id: int = 0) -> cv2.VideoCapture:
    """Open a platform camera backend, falling back to automatic selection.

    Args:
        camera_id: Camera device index.

    Returns:
        Capture whose open state must be checked and which must be released.
    """
    backend = {
        "Windows": cv2.CAP_DSHOW,
        "Linux": cv2.CAP_V4L2,
        "Darwin": cv2.CAP_AVFOUNDATION,
    }.get(platform.system(), cv2.CAP_ANY)
    cap = cv2.VideoCapture(camera_id, backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_id)
    return cap


def draw_results(
    frame: numpy.ndarray, report: dict[str, Any], display_size: tuple[int, int], status: str
) -> numpy.ndarray:
    """Draw source-coordinate results at the current viewport resolution.

    Args:
        frame: Already mirrored camera frame, matching the inference orientation.
        report: Last completed report; no-face results clear previous landmarks.
        display_size: Target viewport width and height.
        status: Inference activity and timing text.

    Returns:
        Annotated, aspect-preserving BGR canvas without modifying inputs.
    """
    source_height, source_width = frame.shape[:2]
    left, top, width, height = fit_geometry((source_width, source_height), display_size)
    canvas = fit_canvas(frame, display_size).copy()
    scale_x, scale_y = width / source_width, height / source_height
    ui_scale = min(display_size[0] / 960, display_size[1] / 720)
    radius = max(1, round(ui_scale))
    detections = report.get("face_detection", [])
    for index, bbox in enumerate(detections):
        x1, y1, x2, y2 = bbox["xyxy"]
        start = (left + round(x1 * scale_x), top + round(y1 * scale_y))
        end = (left + round(x2 * scale_x), top + round(y2 * scale_y))
        color = (80, 230, 100) if index == 0 else (220, 100, 220)
        cv2.rectangle(canvas, start, end, color, max(1, round(2 * ui_scale)), cv2.LINE_AA)
        draw_text(
            canvas,
            f"Face {index + 1} | {bbox.get('score', 0.0):.2f}",
            (max(0, start[0]), max(round(24 * ui_scale), start[1] - round(8 * ui_scale))),
            color,
            scale=0.6 * ui_scale,
        )
    point_count = 0
    for face in report.get(LANDMARK_ATTRIBUTE, []) if detections else []:
        points = numpy.asarray(face.get(LANDMARK_FIELD, []), dtype=numpy.float32)
        if points.size == 0:
            continue
        points = points.reshape(-1, LANDMARK_DIMENSIONS)[:, :2].copy()
        if NORMALIZED_COORDINATES:
            points *= (source_width, source_height)
        for px, py in points:
            if not numpy.isfinite((px, py)).all():
                continue
            if not (0 <= px < source_width and 0 <= py < source_height):
                continue
            point = (left + round(float(px) * scale_x), top + round(float(py) * scale_y))
            cv2.circle(canvas, point, radius + 1, (35, 35, 35), -1, cv2.LINE_AA)
            cv2.circle(canvas, point, radius, (255, 210, 80), -1, cv2.LINE_AA)
            point_count += 1
    footer_height = min(canvas.shape[0], max(1, round(76 * ui_scale)))
    footer = canvas[-footer_height:]
    footer[:] = cv2.addWeighted(footer, 0.65, numpy.full_like(footer, (28, 25, 23)), 0.35, 0)
    draw_text(
        canvas,
        f"{status} | Faces {len(detections)} | Points {point_count}",
        (round(16 * ui_scale), canvas.shape[0] - round(46 * ui_scale)),
        scale=0.6 * ui_scale,
    )
    draw_text(
        canvas,
        "Mirrored preview | F = Fullscreen | Esc / Q / X = Exit",
        (round(16 * ui_scale), canvas.shape[0] - round(16 * ui_scale)),
        scale=0.6 * ui_scale,
    )
    return canvas


def run_camera(cap: cv2.VideoCapture) -> None:
    """Keep preview frames moving while one landmark request runs at a time.

    Frames are mirrored before both display and inference. Completed results
    remain visible until replaced; inference errors propagate after cleanup.

    Args:
        cap: Open camera owned by the caller.
    """
    factory = Factory()
    for root in override_modules:
        factory.load_dir_modules(root)
    pipeline = Pipeline(factory, packages=PACKAGES)
    pipeline.solve()
    pipeline.install_requirements()
    pipeline.cache_models()
    runner = Runner(pipeline)
    pending: Future[dict[str, Any]] | None = None
    report: dict[str, Any] = {}
    submitted_at = 0.0
    inference_fps = 0.0
    completed = False
    try:
        window = CameraWindow(WINDOW_NAME, (960, 720))
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped delivering frames. Check the connection and camera permissions.")
                break
            frame = cv2.flip(frame, 1)
            now = time.monotonic()
            if pending is not None and pending.done():
                report = pending.get()
                pending = None
                inference_fps = 1.0 / max(now - submitted_at, 1e-6)
                completed = True
            if pending is None:
                submitted_at = now
                pending = runner.run_async(data={"default": frame.copy()}, timestamp=time.time())
            state = "No face detected" if completed and not report.get("face_detection") else "Processing"
            status = f"{state} | AI {inference_fps:.1f} FPS | Pending {now - submitted_at:.1f}s"
            window.show(draw_results(frame, report, window.size(), status))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q"), ord("x"), ord("X")):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            window.handle_key(key)
    finally:
        if pending is not None:
            pending.cancel()
        runner.dispose()


def main() -> None:
    """Open the camera before model preparation and release it on every exit."""
    cap = open_camera(0)
    try:
        if not cap.isOpened():
            print("Could not open camera 0. Check the device index, camera permissions, and other camera apps.")
            return
        run_camera(cap)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
