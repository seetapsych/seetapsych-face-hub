# -*- coding: utf-8 -*-

from typing import Any, Literal

import numpy
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

from seetapsych_lib import api


class Instance(api.Instance):
    def __init__(self, task_path: str, device: api.Device,
                 running_mode: Literal['IMAGE', 'VIDEO', 'LIVE_STREAM'] = 'IMAGE',
                 num_faces: int = 1,
                 min_face_detection_confidence: float = 0.5,
                 min_face_presence_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 output_face_blendshapes: bool = False,
                 output_facial_transformation_matrixes: bool = False):
        # use_gpu = device.type in {'cuda', 'gpu'}
        running_mode_map = {
            'IMAGE': VisionTaskRunningMode.IMAGE,
            'VIDEO': VisionTaskRunningMode.VIDEO,
            'LIVE_STREAM': VisionTaskRunningMode.LIVE_STREAM,
        }

        base_options = python.BaseOptions(model_asset_path=task_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode_map[running_mode],
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=output_face_blendshapes,
            output_facial_transformation_matrixes=output_facial_transformation_matrixes,
        )

        detector = vision.FaceLandmarker.create_from_options(options)

        self.__detector = detector

    def inference(self, *,
                  data: dict[str, Any],
                  report: dict[str, Any],
                  **kwargs) -> dict[str, Any]:
        input_data = data['default']
        input_data = numpy.ascontiguousarray(input_data)  # [H, W, C] format

        rgb_data = input_data[:, :, ::-1]
        # image = mp.Image(mp.ImageFormat.SRGB, numpy.ascontiguousarray(rgb_data))   # convert to RGB input

        # preload face_detection
        face_detection = report.get('face_detection', [])

        face_mesh_landmarks = []

        height, width = input_data.shape[:2]
        for the_detection in face_detection:
            left, top, right, bottom = the_detection['xyxy']
            cx = (left + right) / 2
            cy = (bottom + top) / 2
            size = max(right - left, bottom - top) * 1.2

            left, top, right, bottom = cx -  size / 2, cy - size / 2, cx + size / 2, cy + size / 2

            left = int(left)
            right = int(right)
            top = int(top)
            bottom = int(bottom)

            left = max(0, min(width - 1, left))
            top = max(0, min(height - 1, top))
            right = max(left, min(width - 1, right))
            bottom = max(top, min(height - 1, bottom))

            roi = rgb_data[top:bottom, left:right]
            roi_image = mp.Image(mp.ImageFormat.SRGB, numpy.ascontiguousarray(roi))

            detection_result = self.__detector.detect(roi_image)

            if detection_result.face_landmarks:
                the_landmarks = detection_result.face_landmarks[0]

                point3ds = numpy.asarray([v for p in the_landmarks for v in (p.x, p.y, p.z)]).reshape([-1, 3])
                point3ds *= [(right - left) / width, (bottom - top) / height, 1]
                point3ds += [left / width, top / height, 0]

                face_mesh_landmarks.append({
                    'normalized_3d_landmarks': point3ds.flatten().tolist(),
                })
            else:
                face_mesh_landmarks.append({
                    'normalized_3d_landmarks': [],
                })
        return {
            'face_mesh': face_mesh_landmarks,
        }

    def dispose(self):
        self.__detector.close()


class Package(api.Package):
    def create(self, *,
               models: list[api.UsageModel],
               parameters: dict[str, Any],
               device: api.Device | None,
               **kwargs) -> Instance:
        assert len(models) >= 1, api.MissingModelError('At least one model required')

        running_mode = parameters.get('running_mode', 'IMAGE')
        num_faces = parameters.get('num_faces', 1)
        min_face_detection_confidence = parameters.get('min_face_detection_confidence', 0.5)
        min_face_presence_confidence = parameters.get('min_face_presence_confidence', 0.5)
        min_tracking_confidence = parameters.get('min_tracking_confidence', 0.5)
        output_face_blendshapes = parameters.get('output_face_blendshapes', False)
        output_facial_transformation_matrixes = parameters.get('output_facial_transformation_matrixes', False)

        model_path = models[0].cache()
        return Instance(
            model_path,
            api.Device('cpu') if device is None else device,
            running_mode=running_mode,
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=output_face_blendshapes,
            output_facial_transformation_matrixes=output_facial_transformation_matrixes,
        )


def load() -> api.Package:
    return Package()


def main():
    pass


if __name__ == '__main__':
    main()
