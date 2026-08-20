# -*- coding: utf-8 -*-

from typing import Any, Literal

import numpy
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

from fabopsy_lib import api


class Instance(api.Instance):
    def __init__(self, model_path: str, device: api.Device,
                 running_mode: Literal['IMAGE', 'VIDEO', 'LIVE_STREAM'] = 'IMAGE',
                 min_detection_confidence: float = 0.5,
                 min_suppression_threshold: float = 0.3):
        # use_gpu = device.type in {'cuda', 'gpu'}
        running_mode_map = {
            'IMAGE': VisionTaskRunningMode.IMAGE,
            'VIDEO': VisionTaskRunningMode.VIDEO,
            'LIVE_STREAM': VisionTaskRunningMode.LIVE_STREAM,
        }

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=running_mode_map[running_mode],
            min_detection_confidence=min_detection_confidence,
            min_suppression_threshold=min_suppression_threshold,
        )

        detector = vision.FaceDetector.create_from_options(options)

        self.__detector = detector

    def inference(self, *,
                  data: dict[str, Any],
                  report: dict[str, Any],
                  **kwargs) -> dict[str, Any]:
        input_data = data['default']
        input_data = numpy.ascontiguousarray(input_data)  # [H, W, C] format

        image = mp.Image(mp.ImageFormat.SRGB, numpy.ascontiguousarray(input_data[:, :, ::-1]))   # convert to RGB input

        detection_result = self.__detector.detect(image)

        face_detection = []

        for detection in detection_result.detections:
            x1, y1 = detection.bounding_box.origin_x, detection.bounding_box.origin_y
            x2, y2 = x1 + detection.bounding_box.width, y1 + detection.bounding_box.height
            score = detection.categories[0].score
            face_detection.append({
                'xyxy': [x1, y1, x2, y2],
                'score': score,
            })

        return {
            'face_detection': face_detection,
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
        min_detection_confidence = parameters.get('min_detection_confidence', 0.5)
        min_suppression_threshold = parameters.get('min_suppression_threshold', 0.3)

        model_path = models[0].cache()
        return Instance(
            model_path,
            api.Device('cpu') if device is None else device,
            running_mode=running_mode,
            min_detection_confidence=min_detection_confidence,
            min_suppression_threshold=min_suppression_threshold,
        )


def load() -> api.Package:
    return Package()


def main():
    pass


if __name__ == '__main__':
    main()
