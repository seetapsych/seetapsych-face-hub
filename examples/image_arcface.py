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
COLOR_LANDMARK: tuple[int, int, int] = (255, 0, 0)
COLOR_SCORE: tuple[int, int, int] = (255, 255, 255)
COLOR_SPECTRUM_LINE: tuple[int, int, int] = (0, 200, 255)
COLOR_SPECTRUM_FILL: tuple[int, int, int] = (0, 100, 180)
COLOR_ZERO_LINE: tuple[int, int, int] = (128, 128, 128)
COLOR_AXIS_LABEL: tuple[int, int, int] = (200, 200, 200)
COLOR_BAR_POS: tuple[int, int, int] = (0, 140, 255)
COLOR_BAR_NEG: tuple[int, int, int] = (255, 140, 60)
LINE_STYLE_SOLID: int = cv2.LINE_8
LINE_STYLE_DASH_GAP: int = 8
LABEL_BG_ALPHA: float = 0.45


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


def draw_feature_spectrum(
    canvas: numpy.ndarray,
    feature: list[float],
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> None:
    """Render L2-normalized face embedding as a vertical voiceprint-style spectrum plot.

    Layout is rotated 90 deg relative to a conventional spectrogram so the panel
    sits to the right of portrait images without extending total height:

    * Y axis — dimension index, running top (dim 0) to bottom (dim N-1). Ticks at
      every N/8 dims; last index included explicitly.
    * X axis — feature value, centered on a zero line; positive goes right,
      negative goes left, symmetric span around ``±1.15 * max(|v|)``.
    * Filled area between the polyline and the centered zero axis.
    * Right-hand heat strip — per-dimension sign-magnitude bars: orange grows
      right for positive values; blue grows left for negative ones.
    * Top label row with dim count and L2 norm sanity check.
    * Left value ticks for the feature magnitude axis.

    Args:
        canvas: Target BGR image (in-place).
        feature: N-dim L2-normalized embedding values.
        x0: Left pixel of plot region.
        y0: Top pixel of plot region.
        width: Available width in pixels.
        height: Available height in pixels.
    """
    if width <= 0 or height <= 0 or len(feature) == 0:
        return

    dims = len(feature)
    feat = numpy.asarray(feature, dtype=numpy.float32)

    font = cv2.FONT_HERSHEY_SIMPLEX
    label_font_scale = 0.55
    label_thickness = 1
    tick_font_scale = 0.45
    tick_thickness = 1

    label_text = f"ArcFace {dims}-dim Feature  L2={numpy.linalg.norm(feat):.4f}"
    (_, label_h), _ = cv2.getTextSize(label_text, font, label_font_scale, label_thickness)
    label_margin_v = 8
    label_total_h = label_h + label_margin_v * 2

    heatbar_w = 56
    heatbar_margin = 8
    topaxis_margin_top = tick_h = int(16)
    _ = topaxis_margin_top
    xaxis_margin_left = 64
    xaxis_margin_right = heatbar_w + heatbar_margin + 16

    plot_top: int = y0 + label_total_h
    plot_bottom: int = y0 + height - tick_h - 12
    plot_left: int = x0 + xaxis_margin_left
    plot_right: int = x0 + width - xaxis_margin_right

    if plot_bottom <= plot_top or plot_right <= plot_left:
        return

    plot_h = plot_bottom - plot_top
    plot_w = plot_right - plot_left

    cv2.rectangle(canvas, (x0, y0), (x0 + width - 1, y0 + height - 1), (60, 60, 60), 1)
    put_text_with_shadow(
        canvas,
        label_text,
        (x0 + 10, y0 + label_margin_v + label_h),
        font,
        label_font_scale,
        (230, 230, 230),
        label_thickness,
    )

    abs_max = float(numpy.max(numpy.abs(feat))) if dims > 0 else 1.0
    if abs_max < 1e-6:
        abs_max = 1e-6
    x_span = abs_max * 1.15
    x_zero = plot_left + plot_w // 2

    cv2.line(canvas, (x_zero, plot_top), (x_zero, plot_bottom), COLOR_ZERO_LINE, 1, lineType=LINE_STYLE_SOLID)

    def x_map(val: float) -> int:
        frac = val / x_span
        return int(round(x_zero + frac * (plot_w * 0.45)))

    def y_map(idx: int) -> int:
        if dims <= 1:
            return plot_top + plot_h // 2
        frac = idx / (dims - 1)
        return int(round(plot_top + frac * (plot_h - 1)))

    pts = numpy.zeros((dims, 2), dtype=numpy.int32)
    for i in range(dims):
        pts[i, 0] = x_map(feat[i])
        pts[i, 1] = y_map(i)

    fill_pts: list[tuple[int, int]] = []
    fill_pts.append((x_zero, pts[0, 1]))
    fill_pts.extend((int(p[0]), int(p[1])) for p in pts)
    fill_pts.append((x_zero, pts[-1, 1]))

    fill_arr_list = [numpy.array(fill_pts, dtype=numpy.int32)]
    overlay = canvas.copy()
    cv2.fillPoly(overlay, fill_arr_list, COLOR_SPECTRUM_FILL)
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, dst=canvas)

    cv2.polylines(canvas, [pts], False, COLOR_SPECTRUM_LINE, 2, lineType=LINE_STYLE_SOLID)

    x_labels = [
        (x_span, f"+{x_span:.2f}"),
        (x_span * 0.5, f"+{x_span * 0.5:.2f}"),
        (0.0, "0.00"),
        (-x_span * 0.5, f"-{x_span * 0.5:.2f}"),
        (-x_span, f"-{x_span:.2f}"),
    ]
    for val, label in x_labels:
        px = x_map(val)
        if plot_left <= px <= plot_right:
            cv2.line(canvas, (px, plot_bottom), (px, plot_bottom + 4), COLOR_AXIS_LABEL, 1)
            (tw, th), _ = cv2.getTextSize(label, font, tick_font_scale, tick_thickness)
            put_text_with_shadow(
                canvas,
                label,
                (px - tw // 2, plot_bottom + th + 8),
                font,
                tick_font_scale,
                COLOR_AXIS_LABEL,
                tick_thickness,
            )

    y_ticks = list(range(0, dims, max(1, dims // 8)))
    if y_ticks[-1] != dims - 1:
        y_ticks.append(dims - 1)
    for idx in y_ticks:
        py = y_map(idx)
        cv2.line(canvas, (x_zero - 4, py), (x_zero, py), COLOR_AXIS_LABEL, 1)
        label = str(idx)
        (tw, th), _ = cv2.getTextSize(label, font, tick_font_scale, tick_thickness)
        put_text_with_shadow(
            canvas,
            label,
            (x_zero - tw - 10, py + th // 3),
            font,
            tick_font_scale,
            COLOR_AXIS_LABEL,
            tick_thickness,
        )

    bar_left = plot_right + heatbar_margin
    bar_right = bar_left + heatbar_w
    if bar_right <= x0 + width - 8:
        cv2.rectangle(canvas, (bar_left - 1, plot_top - 1), (bar_right + 1, plot_bottom + 1), (80, 80, 80), 1)
        bar_inner_left = bar_left
        bar_inner_right = bar_right
        bar_w = max(1, bar_inner_right - bar_inner_left)
        bar_mid = (bar_inner_left + bar_inner_right) // 2
        for i in range(dims):
            by0 = y_map(i)
            if i < dims - 1:
                by1 = y_map(i + 1)
            else:
                by1 = plot_bottom
            bh = max(1, by1 - by0)
            val = feat[i]
            mag = min(1.0, abs(val) / x_span)
            if val >= 0:
                c = tuple(int(v) for v in COLOR_BAR_POS)
                x_start = bar_mid
                x_end = bar_mid + max(1, int(round((bar_w // 2) * mag)))
            else:
                c = tuple(int(v) for v in COLOR_BAR_NEG)
                x_end = bar_mid + 1
                x_start = bar_mid - max(1, int(round((bar_w // 2) * mag)))
            color = (
                int(max(0, min(255, c[0] * (0.35 + 0.65 * mag)))),
                int(max(0, min(255, c[1] * (0.35 + 0.65 * mag)))),
                int(max(0, min(255, c[2] * (0.35 + 0.65 * mag)))),
            )
            cv2.rectangle(canvas, (x_start, by0), (x_end - 1, by0 + bh - 1), color, -1)

        legend_x = bar_left
        legend_y = y0 + height - 4
        sw, sh = 8, 14
        cv2.rectangle(canvas, (legend_x, legend_y - sh), (legend_x + sw, legend_y), COLOR_BAR_POS, -1)
        put_text_with_shadow(
            canvas,
            "+",
            (legend_x + sw + 3, legend_y),
            font,
            tick_font_scale,
            COLOR_AXIS_LABEL,
            tick_thickness,
        )
        lx2 = legend_x + 30
        cv2.rectangle(canvas, (lx2, legend_y - sh), (lx2 + sw, legend_y), COLOR_BAR_NEG, -1)
        put_text_with_shadow(
            canvas,
            "-",
            (lx2 + sw + 3, legend_y),
            font,
            tick_font_scale,
            COLOR_AXIS_LABEL,
            tick_thickness,
        )


def draw_results(image: numpy.ndarray, report: dict[str, Any]) -> numpy.ndarray:
    """Render face bboxes, landmarks, and a right-side vertical feature spectrum.

    Canvas layout strategy:
    * For medium/large images (any dim >= 320 px after fit_image): extend the
      canvas horizontally with a ``feature_panel_w`` column to the right so
      portrait images retain their full height.
    * For smaller images: draw the spectrum as a semi-transparent overlay in
      the bottom-right corner, preserving the original output dimensions.

    Args:
        image: Input BGR image.
        report: Algorithm result dict with per-face lists:
            ``face_detection``, ``face_landmarks``, ``face_feature``.

    Returns:
        Annotated BGR image.
    """
    base, scale = fit_image(image)
    bh, bw = base.shape[:2]

    face_detection = report.get("face_detection", [])
    face_landmarks = report.get("face_landmarks", [])
    face_feature = report.get("face_feature", [])

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2

    feature_panel_w = 380
    overlay_mode = min(bh, bw) < 320 or not face_feature
    if overlay_mode:
        canvas = base
    else:
        canvas = numpy.zeros((bh, bw + feature_panel_w, 3), dtype=numpy.uint8)
        canvas[:, :bw] = base

    for face_idx, bbox in enumerate(face_detection):
        xyxy = [int(round(v * scale)) for v in bbox["xyxy"]]
        pt1, pt2 = (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3])
        if face_idx == 0:
            cv2.rectangle(canvas, pt1, pt2, COLOR_FACE_PRIMARY, 2, lineType=LINE_STYLE_SOLID)
        else:
            draw_rectangle_dashed(canvas, pt1, pt2, COLOR_FACE_SECONDARY, thickness=2)

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
        draw_label_rect(canvas, (label_x1, label_y1), (label_x2, label_y2))
        put_text_with_shadow(
            canvas,
            label,
            (label_x1 + 4, label_y1 + text_h + 4),
            font,
            font_scale,
            COLOR_SCORE,
            thickness,
            shadow_base_offset_px=1,
        )

    for _face_idx, lm in enumerate(face_landmarks):
        landmarks = lm.get("landmarks", [])
        if len(landmarks) >= 10:
            pts = [(int(round(landmarks[i] * scale)), int(round(landmarks[i + 1] * scale))) for i in range(0, 10, 2)]
            for px, py in pts:
                cv2.circle(canvas, (px, py), 3, COLOR_LANDMARK, -1, lineType=LINE_STYLE_SOLID)

    if face_feature:
        if overlay_mode:
            panel_w = min(bw - 8, 360)
            panel_h = min(bh - 8, 260)
            px0 = max(4, bw - panel_w - 4)
            py0 = max(4, bh - panel_h - 4)
            sub = canvas[py0 : py0 + panel_h, px0 : px0 + panel_w]
            shaded = cv2.addWeighted(sub, 0.55, numpy.zeros_like(sub), 0.45, 12)
            canvas[py0 : py0 + panel_h, px0 : px0 + panel_w] = shaded
            draw_feature_spectrum(canvas, face_feature[0], px0, py0, panel_w, panel_h)
        else:
            panel_x0 = bw
            panel_w = feature_panel_w
            panel_y0 = 4
            panel_h = bh - 8
            draw_feature_spectrum(canvas, face_feature[0], panel_x0, panel_y0, panel_w, panel_h)

    return canvas


def main():
    factory = Factory()
    # load_dir_modules tolerates missing dirs — example runs fine without local overrides.
    for root in override_modules:
        factory.load_dir_modules(root)

    pipeline = Pipeline(
        factory,
        packages=[
            "c938b879-44db-45b0-9a5d-8377f0ace5e5",  # FaceDetection-RetinaFace(InsightFace)
            "a761cb3a-dba8-4b73-bc2b-6a1c02e55a27",  # FaceFeature-ArcFace(InsightFace)
        ],
        attributes=[
            # 'face/detection',
            # 'face/landmarks',
            # 'face/feature',
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
    out_path = f"{stem}_arcface_result{ext}"
    cv2.imwrite(out_path, vis)
    print(f"Saved result to: {out_path}")

    cv2.imshow("arcface", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    runner.dispose()


if __name__ == "__main__":
    main()
