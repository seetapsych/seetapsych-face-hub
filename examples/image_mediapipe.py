# -*- coding: utf-8 -*-

import json
import os
from typing import Any

import cv2
import numpy
from seetapsych_lib.runtime.factory import Factory
from seetapsych_lib.runtime.parallel_runner import ParallelRunner as Runner
from seetapsych_lib.runtime.pipeline import Pipeline

override_modules = [
    os.path.join(os.path.dirname(__file__), "../seetapsych_face_hub/modules"),
    os.path.join(os.path.dirname(__file__), "../../seetapsych-lib/seetapsych_lib/modules"),
]

image_path = os.path.join(os.path.dirname(__file__), "ian-dooley.jpg")

COLOR_FACE_PRIMARY: tuple[int, int, int] = (0, 255, 0)
COLOR_FACE_SECONDARY: tuple[int, int, int] = (255, 0, 255)
COLOR_MESH: tuple[int, int, int] = (255, 0, 0)
COLOR_SCORE: tuple[int, int, int] = (255, 255, 255)
LINE_STYLE_SOLID: int = cv2.LINE_8
LINE_STYLE_DASH_GAP: int = 8


def fit_image(image: numpy.ndarray, max_width: int = 1280, max_height: int = 960) -> tuple[numpy.ndarray, float]:
    """Proportionally downscale image to fit within max dimensions.

    Args:
        image: Input BGR image.
        max_width: Width cap in pixels.
        max_height: Height cap in pixels.

    Returns:
        Resized image and applied scale (<= 1.0; 1.0 if no resize occurred).
    """
    h, w = image.shape[:2]
    if w <= max_width and h <= max_height:
        return image, 1.0
    scale = min(max_width / w, max_height / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


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


def draw_rectangle_dashed(
    image: numpy.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    gap: int = LINE_STYLE_DASH_GAP,
) -> None:
    """Draw a dashed rectangle by alternating line segments on each edge.

    Args:
        image: Target image (in-place).
        pt1: Top-left corner (x, y).
        pt2: Bottom-right corner (x, y).
        color: BGR stroke color.
        thickness: Line thickness.
        gap: On/off segment length in pixels.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    points: list[tuple[tuple[int, int], tuple[int, int]]] = []
    points.extend(((x, y1), (min(x + gap, x2), y1)) for x in range(x1, x2, gap * 2))
    points.extend(((x, y2), (min(x + gap, x2), y2)) for x in range(x1, x2, gap * 2))
    points.extend(((x1, y), (x1, min(y + gap, y2))) for y in range(y1, y2, gap * 2))
    points.extend(((x2, y), (x2, min(y + gap, y2))) for y in range(y1, y2, gap * 2))
    for a, b in points:
        cv2.line(image, a, b, color, thickness, lineType=LINE_STYLE_SOLID)


def draw_results(image: numpy.ndarray, report: dict[str, Any]) -> numpy.ndarray:
    """Render face bboxes and MediaPipe 468-point face mesh dots.

    Multi-face bbox styling convention:
    * Face 0 (primary) — green solid rectangle.
    * Face N (N >= 1) — magenta dashed rectangle.

    Args:
        image: Input BGR image.
        report: Algorithm result dict with per-face lists:
            ``face_detection``, ``face_mesh`` (``normalized_3d_landmarks``: list[468*3]).

    Returns:
        Annotated BGR image.
    """
    vis, scale = fit_image(image)
    fh_orig, fw_orig = image.shape[:2]
    fh_vis, fw_vis = vis.shape[:2]

    face_detection = report.get("face_detection", [])
    face_mesh = report.get("face_mesh", [])

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2

    for face_idx, bbox in enumerate(face_detection):
        xyxy = [int(round(v * scale)) for v in bbox["xyxy"]]
        pt1, pt2 = (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])
        if face_idx == 0:
            cv2.rectangle(vis, pt1, pt2, COLOR_FACE_PRIMARY, 2, lineType=LINE_STYLE_SOLID)
        else:
            draw_rectangle_dashed(vis, pt1, pt2, COLOR_FACE_SECONDARY, thickness=2)

        score = bbox.get("score", 0.0)
        label = f"{score:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        label_bg_h = text_h + 8
        bbox_gap = 2
        room_above = xyxy[1] - label_bg_h - bbox_gap
        if room_above >= 0:
            label_y1 = room_above
            label_inside = False
        else:
            label_y1 = xyxy[1] + 3
            label_inside = True
        label_y2 = label_y1 + label_bg_h
        label_x1 = xyxy[0] + (0 if not label_inside else 3)
        label_x2 = label_x1 + text_w + 8
        cv2.rectangle(vis, (label_x1, label_y1), (label_x2, label_y2), (0, 0, 0), -1)
        put_text_with_shadow(
            vis,
            label,
            (label_x1 + 4, label_y1 + text_h + 4),
            font,
            font_scale,
            COLOR_SCORE,
            thickness,
            shadow_base_offset_px=1,
        )

    for _face_idx, mesh in enumerate(face_mesh):
        raw = mesh.get("normalized_3d_landmarks", [])
        if len(raw) < 3:
            continue
        pts3d = numpy.asarray(raw, dtype=numpy.float32).reshape(-1, 3)
        pts2d = pts3d[:, :2]
        pts2d[:, 0] *= fw_orig
        pts2d[:, 1] *= fh_orig
        pts2d *= scale
        for px, py in pts2d:
            ix, iy = int(round(px)), int(round(py))
            if 0 <= ix < fw_vis and 0 <= iy < fh_vis:
                cv2.circle(vis, (ix, iy), 1, COLOR_MESH, -1, lineType=LINE_STYLE_SOLID)

    return vis


def main():
    factory = Factory()
    # load_dir_modules tolerates missing dirs — example runs fine without local overrides.
    for root in override_modules:
        factory.load_dir_modules(root)

    pipeline = Pipeline(
        factory,
        packages=[
            "56d8a9a9-0ad9-4d90-ad3f-ffc2f51c1597",  # FaceDetection-MediaPipe
            "8e646eec-e50f-4102-a658-1449d04296fb",  # FaceMesh-MediaPipe
        ],
        attributes=[
            # 'face/detection',
            # 'face/mesh',
        ],
    )

    pipeline.solve()
    pipeline.install_requirements()
    pipeline.cache_models()

    runner = Runner(pipeline)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    report = runner.run(data={"default": image})
    print(json.dumps(report, indent=2, ensure_ascii=False))

    vis = draw_results(image, report)

    stem, ext = os.path.splitext(image_path)
    out_path = f"{stem}_mpmesh_result{ext}"
    cv2.imwrite(out_path, vis)
    print(f"Saved result to: {out_path}")

    cv2.imshow("mediapipe_mesh", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    runner.dispose()


if __name__ == "__main__":
    main()
