# -*- coding: utf-8 -*-

from typing import Any

import numpy

from fabopsy_lib import api
from fabopsy_lib.onnx.session import OnnxSession

from .arcface_onnx import ArcFaceONNX
from .common import Face

class Instance(api.Instance):
    def __init__(self, model_path: str, device: api.Device):
        self.__session = OnnxSession(model_path, device)
        self.__arcface = ArcFaceONNX(model_path, session=self.__session.session)

    def inference(self, *,
                  data: dict[str, Any],
                  report: dict[str, Any],
                  **kwargs) -> dict[str, Any]:
        input_data = data['default']
        input_data = numpy.ascontiguousarray(input_data)  # [H, W, C] format

        # got landmarks
        # face_detection = report.get('face_detection', [])
        face_landmark_5 = report.get('face_landmark_5', [])

        face_feature = []


        for face_landmarks in face_landmark_5:
            # xyxy = face_box.get('xyxy', [])
            # score = face_box.get('score', 0)
            landmarks = face_landmarks.get('landmarks', [])

            # xyxy = numpy.asarray(xyxy)
            landmarks = numpy.asarray(landmarks).reshape((-1, 2))
            face = Face(bbox=None, kps=landmarks, det_score=None)

            feat = self.__arcface.get(input_data, face)
            feat = numpy.asarray(feat).reshape([-1])
            norm_feat = feat / (numpy.linalg.norm(feat) + 1e-12)
            feat: list[float] = norm_feat.tolist()

            face_feature.append(feat)

        return {
            'face_feature': face_feature,
        }


def format_input_size(size: float | list[float]) -> tuple[float, float]:
    match size:
        case int(x):
            return x, x
        case float(x):
            return x, x
        case []:
            raise RuntimeError('input size could not be []')
        case [x]:
            return x, x
        case [x, y]:
            return x, y
        case list(v):
            return v[0], v[1]
        case _:
            raise RuntimeError(f'input size could not be {size}')


class Package(api.Package):
    def create(self, *,
               models: list[api.UsageModel],
               parameters: dict[str, Any],
               device: api.Device | None,
               **kwargs) -> Instance:
        assert len(models) >= 1, api.MissingModelError('At least one model required')

        input_size = format_input_size(parameters.get('input_size', [640, 640]))

        model_path = models[0].cache()
        return Instance(
            model_path,
            api.Device('cpu') if device is None else device,
        )


def load() -> api.Package:
    return Package()


def main():
    pass


if __name__ == '__main__':
    main()
