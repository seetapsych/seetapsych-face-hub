# -*- coding: utf-8 -*-

import os
import platform
import time
from typing import Any

import cv2
import numpy
from camera_window import CameraWindow, draw_text, fit_geometry
from seetapsych_lib.runtime.factory import Factory
from seetapsych_lib.runtime.parallel.future import Future
from seetapsych_lib.runtime.parallel_runner import ParallelRunner as Runner
from seetapsych_lib.runtime.pipeline import Pipeline

override_modules = [
    os.path.join(os.path.dirname(__file__), "../seetapsych_face_hub/modules"),
    os.path.join(os.path.dirname(__file__), "../../seetapsych-lib/seetapsych_lib/modules"),
]

COLOR_FACE_SELECTED: tuple[int, int, int] = (0, 255, 0)
COLOR_PID_LABEL: tuple[int, int, int] = (0, 0, 255)
COLOR_HINT: tuple[int, int, int] = (0, 255, 255)
COLOR_LABEL_TEXT: tuple[int, int, int] = (255, 255, 255)
LINE_STYLE_SOLID: int = cv2.LINE_8
LABEL_BG_ALPHA: float = 0.45


def open_camera(camera_id: int = 0) -> cv2.VideoCapture:
    """Open a video capture with an OS-appropriate backend, falling back to default.

    Args:
        camera_id: Camera device index.

    Returns:
        Open ``cv2.VideoCapture`` instance; caller must ``release()`` when done.
    """
    system = platform.system()

    backend = {
        "Windows": cv2.CAP_DSHOW,
        "Linux": cv2.CAP_V4L2,
        "Darwin": cv2.CAP_AVFOUNDATION,
    }.get(system, cv2.CAP_ANY)

    cap = cv2.VideoCapture(camera_id, backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_id)

    return cap


def put_text_with_shadow(
    image: numpy.ndarray,
    text: str,
    org: tuple[int, int],
    font: int,
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
    *,
    shadow_color: tuple[int, int, int] = (96, 96, 96),
    shadow_base_offset_px: int = 1,
    line_type: int = cv2.LINE_AA,
) -> None:
    """Draw text with a down-right shadow that scales with font size.

    Args:
        image: Target image (in-place).
        text: String to render.
        org: Top-left anchor (x, y) of primary text.
        font: OpenCV FONT_HERSHEY_* constant.
        font_scale: Font scale, same semantics as ``cv2.putText``.
        color: Primary text BGR color.
        thickness: Stroke thickness for shadow and text.
        shadow_color: Shadow BGR color. Defaults to mid-gray.
        shadow_base_offset_px: Shadow offset at ``font_scale == 1.0``.
            Actual offset = ``round(base_offset * max(font_scale, 1.0))``.
        line_type: ``cv2.putText`` line type. Defaults to anti-aliased.
    """
    offset = int(round(shadow_base_offset_px * max(font_scale, 1.0)))
    x, y = org
    cv2.putText(image, text, (x + offset, y + offset), font, font_scale, shadow_color, thickness, line_type)
    cv2.putText(image, text, org, font, font_scale, color, thickness, line_type)


def draw_label_rect(
    image: numpy.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    *,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    alpha: float = LABEL_BG_ALPHA,
) -> None:
    """Draw a semi-transparent filled rectangle used as a label background.

    Args:
        image: Target image (in-place).
        pt1: Top-left corner (x, y).
        pt2: Bottom-right corner (x, y).
        bg_color: Fill BGR color. Defaults to black.
        alpha: Opacity of the fill (0 = fully transparent, 1 = fully opaque).
    """
    x1, y1 = pt1
    x2, y2 = pt2
    h, w = image.shape[:2]
    if x2 <= x1 or y2 <= y1 or x1 >= w or y1 >= h or x2 <= 0 or y2 <= 0:
        return
    rx1 = int(max(0, min(x1, w - 1)))
    ry1 = int(max(0, min(y1, h - 1)))
    rx2 = int(max(1, min(x2, w)))
    ry2 = int(max(1, min(y2, h)))
    overlay = image[ry1:ry2, rx1:rx2]
    if overlay.size == 0:
        return
    filled = numpy.full_like(overlay, bg_color, dtype=numpy.uint8)
    image[ry1:ry2, rx1:rx2] = cv2.addWeighted(overlay, 1.0 - alpha, filled, alpha, 0.0)


def draw_results(
    frame: numpy.ndarray, report: dict[str, Any], display_size: tuple[int, int] | None = None
) -> numpy.ndarray:
    """Draw the latest detections on a mirrored live preview.

    Args:
        frame: Current camera frame, not the inference input.
        report: Latest report; an empty dict hides stale annotations.
        display_size: Optional viewport bounds; inference coordinates stay in source pixels.

    Returns:
        Mirrored, annotated preview.
    """
    source_height, source_width = frame.shape[:2]
    if display_size is not None:
        _, _, rw, rh = fit_geometry((source_width, source_height), display_size)
        interpolation = cv2.INTER_AREA if rw < source_width else cv2.INTER_CUBIC
        frame = cv2.resize(frame, (rw, rh), interpolation=interpolation)
    canvas = cv2.flip(frame, 1)
    width = canvas.shape[1]
    scale_x, scale_y = width / source_width, canvas.shape[0] / source_height
    ui_scale = min(width / 640, canvas.shape[0] / 480)
    font = cv2.FONT_HERSHEY_SIMPLEX
    pid = report.get("face_selection", {}).get("pid", 0)
    for index, bbox in enumerate(report.get("face_detection", [])):
        x1, y1, x2, y2 = (
            int(round(v * scale)) for v, scale in zip(bbox["xyxy"], (scale_x, scale_y, scale_x, scale_y), strict=True)
        )
        left, right = width - 1 - x2, width - 1 - x1
        color = COLOR_FACE_SELECTED if index == 0 else (180, 180, 180)
        cv2.rectangle(canvas, (left, y1), (right, y2), color, max(1, round(2 * ui_scale)), cv2.LINE_AA)
        label = f"{bbox.get('score', 0.0):.2f}"
        if index == 0 and pid > 0:
            label = f"PID {pid} | {label}"
        (tw, th), _ = cv2.getTextSize(label, font, 0.6 * ui_scale, 1)
        label_y = max(th + 8, y1 - 5)
        draw_label_rect(canvas, (left, label_y - th - 6), (left + tw + 8, label_y + 4))
        put_text_with_shadow(canvas, label, (left + 4, label_y), font, 0.6 * ui_scale, COLOR_LABEL_TEXT)
    draw_label_rect(canvas, (0, canvas.shape[0] - round(72 * ui_scale)), (width, canvas.shape[0]))
    draw_text(
        canvas,
        "F = Fullscreen | R = Reset | Esc / Q / X = Exit",
        (round(16 * ui_scale), canvas.shape[0] - round(16 * ui_scale)),
        COLOR_HINT,
        scale=0.6 * ui_scale,
    )
    return canvas


def run_camera(cap: cv2.VideoCapture) -> None:
    """Run one asynchronous inference at a time while rendering every frame.

    Args:
        cap: Open camera owned by the caller.
    """
    factory = Factory()
    # load_dir_modules tolerates missing dirs — example runs fine without local overrides.
    for root in override_modules:
        factory.load_dir_modules(root)

    pipeline = Pipeline(
        factory,
        packages=[
            "c938b879-44db-45b0-9a5d-8377f0ace5e5",  # FaceDetection-RetinaFace(InsightFace)
            "bb212f54-aace-438f-9cb7-f6519f4fef48",  # FaceSelection
        ],
        attributes=[
            # 'face/detection',
            # 'face/selection',
        ],
    )

    pipeline.solve()
    pipeline.install_requirements()
    pipeline.cache_models()

    runner = Runner(pipeline)

    window_name = "Camera Face Selection"
    pending: Future[dict[str, Any]] | None = None
    report: dict[str, Any] = {}
    reset_requested = False
    submitted_at = result_at = 0.0
    inference_fps = 0.0
    try:
        window = CameraWindow(window_name, (960, 720))
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped delivering frames. Check the connection and camera permissions.")
                break
            now = time.monotonic()
            if pending is not None and pending.done():
                output = pending.get()
                pending = None
                if not reset_requested:
                    report = output
                    result_at = submitted_at
                    inference_fps = 1.0 / max(now - submitted_at, 1e-6)
            if pending is None:
                if reset_requested:
                    runner.reset()
                    report = {}
                    reset_requested = False
                submitted_at = now
                pending = runner.run_async(data={"default": frame.copy()}, timestamp=time.time())
            result_ttl = max(1.0, 2.0 / max(inference_fps, 0.01))
            canvas = draw_results(frame, report if now - result_at < result_ttl else {}, window.size())
            ui_scale = min(canvas.shape[1] / 640, canvas.shape[0] / 480)
            status = "Reset pending" if reset_requested else f"Inference: {inference_fps:.1f} FPS"
            draw_text(
                canvas,
                status,
                (round(16 * ui_scale), canvas.shape[0] - round(44 * ui_scale)),
                COLOR_LABEL_TEXT,
                scale=0.6 * ui_scale,
            )
            window.show(canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q"), ord("x"), ord("X")):
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            window.handle_key(key)
            if key in (ord("r"), ord("R")):
                reset_requested = True
                report = {}
    finally:
        if pending is not None:
            pending.cancel()
        runner.dispose()


def main() -> None:
    """Open the camera before preparing models and release resources on exit."""
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
