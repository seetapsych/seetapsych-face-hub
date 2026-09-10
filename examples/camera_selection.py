# -*- coding: utf-8 -*-

import os
import platform

import cv2
import numpy
from seetapsych_lib.runtime.factory import Factory
from seetapsych_lib.runtime.parallel_runner import ParallelRunner as Runner
from seetapsych_lib.runtime.pipeline import Pipeline

override_modules = [
    os.path.join(os.path.dirname(__file__), "../seetapsych_face_hub/modules"),
    os.path.join(os.path.dirname(__file__), "../../seetapsych-lib/seetapsych_lib/modules"),
]

COLOR_FACE_SELECTED: tuple[int, int, int] = (0, 255, 0)
COLOR_PID_LABEL: tuple[int, int, int] = (0, 0, 255)
COLOR_HINT: tuple[int, int, int] = (0, 255, 255)
LINE_STYLE_SOLID: int = cv2.LINE_8


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


def main():
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
    cap = open_camera(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    while cap.isOpened():
        ok = cap.grab()
        if not ok:
            break
        ok, frame = cap.retrieve()
        if not ok:
            break

        report = runner.run(data={"default": frame})

        frame = cv2.flip(frame, 1)
        f_h, f_w = frame.shape[:2]

        face_detection = report.get("face_detection", [])
        face_selection = report.get("face_selection", {})

        for bbox in face_detection:
            xyxy = [int(round(v)) for v in bbox["xyxy"]]
            x1, y1, x2, y2 = xyxy
            x1_flip = f_w - x2
            x2_flip = f_w - x1
            cv2.rectangle(frame, (x1_flip, y1), (x2_flip, y2), COLOR_FACE_SELECTED, 2, lineType=LINE_STYLE_SOLID)

            score = bbox.get("score", 0.0)
            label = f"{score:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs, thick = 0.55, 1
            (tw, th), _ = cv2.getTextSize(label, font, fs, thick)
            label_bg_h = th + 8
            bbox_gap = 2
            room_above = y1 - label_bg_h - bbox_gap
            if room_above >= 0:
                ly1 = room_above
                label_inside = False
            else:
                ly1 = y1 + 3
                label_inside = True
            ly2 = ly1 + label_bg_h
            lx = x1_flip + (0 if not label_inside else 3)
            cv2.rectangle(frame, (lx, ly1), (lx + tw + 8, ly2), (0, 0, 0), -1)
            put_text_with_shadow(
                frame,
                label,
                (lx + 4, ly1 + th + 4),
                font,
                fs,
                (255, 255, 255),
                thick,
                shadow_base_offset_px=1,
            )

            pid = bbox.get("pid") or face_selection.get("pid") or 0
            if pid > 0:
                pid_text = f"PID: {pid}"
                pid_fs, pid_thick = 0.9, 2
                (pid_w, pid_h), _ = cv2.getTextSize(pid_text, font, pid_fs, pid_thick)
                pid_bg_h = pid_h + 10
                pid_gap = 4
                pid_room_above = y1 - pid_bg_h - pid_gap
                if pid_room_above >= 0:
                    pid_y1 = pid_room_above
                    pid_inside = False
                else:
                    pid_y1 = y1 + 3
                    pid_inside = True
                pid_y2 = pid_y1 + pid_bg_h
                pid_x1 = x2_flip - pid_w - 12 - (0 if not pid_inside else -3)
                pid_x1 = max(x1_flip + (3 if pid_inside else 0), pid_x1)
                pid_x2 = pid_x1 + pid_w + 12
                cv2.rectangle(frame, (pid_x1, pid_y1), (pid_x2, pid_y2), (0, 0, 0), -1)
                put_text_with_shadow(
                    frame,
                    pid_text,
                    (pid_x1 + 6, pid_y1 + pid_h + 5),
                    font,
                    pid_fs,
                    COLOR_PID_LABEL,
                    pid_thick,
                    shadow_base_offset_px=2,
                )

        hint_lines = [
            "R = Reset tracker",
            "Esc / Q / X = Exit",
        ]
        hint_y = f_h - 16
        for line in reversed(hint_lines):
            (_, th), _ = cv2.getTextSize(line, font, 0.6, 1)
            put_text_with_shadow(frame, line, (16, hint_y), font, 0.6, COLOR_HINT, 1, shadow_base_offset_px=1)
            hint_y -= th + 10

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

        key &= 0xFF
        if key in {ord("q"), ord("x")}:
            break

        if key == ord("r"):
            runner.reset()
            _ = f_h

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE):
        cv2.destroyWindow(window_name)
    cap.release()
    runner.dispose()


if __name__ == "__main__":
    main()
