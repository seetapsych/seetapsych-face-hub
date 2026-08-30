# -*- coding: utf-8 -*-

from typing import Any

import numpy
from seetapsych_lib import api
from seetapsych_lib.onnx.session import OnnxSession

from .lib.retinaface.retinaface import RetinaFace


class Instance(api.Instance):
    def __init__(self, model_path: str, device: api.Device, input_size: tuple[float, float] | None = None):
        self.__session = OnnxSession(model_path, device)
        self.__retinaface = RetinaFace(None, session=self.__session.session)
        self.__input_size = [640, 640] if not input_size else input_size

    def inference(self, *, data: dict[str, Any], report: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        input_data = data["default"]
        input_data = numpy.ascontiguousarray(input_data)  # [H, W, C] format

        dets, kpss = self.__retinaface.detect(input_data, input_size=self.__input_size)

        face_detection = []
        face_landmarks = []

        for det, kps in zip(dets, kpss, strict=False):
            det = det.tolist()
            kps = kps.tolist()
            face_detection.append({"xyxy": det[:4], "score": det[4]})
            face_landmarks.append({"landmarks": [v for p in kps for v in p]})

        return {
            "face_detection": face_detection,
            "face_landmarks": face_landmarks,
        }


def format_input_size(size: float | list[float]) -> tuple[float, float]:
    match size:
        case int(x):
            return x, x
        case float(x):
            return x, x
        case []:
            raise RuntimeError("input size could not be []")
        case [x]:
            return x, x
        case [x, y]:
            return x, y
        case list(v):
            return v[0], v[1]
        case _:
            raise RuntimeError(f"input size could not be {size}")


class Package(api.Package):
    def create(
        self, *, models: list[api.UsageModel], parameters: dict[str, Any], device: api.Device | None, **kwargs: Any
    ) -> Instance:
        assert len(models) >= 1, api.MissingModelError("At least one model required")

        input_size: tuple[float, float] = format_input_size(parameters.get("input_size", (640, 640)))

        model_path = models[0].cache()
        return Instance(
            model_path,
            api.Device("cpu") if device is None else device,
            input_size=input_size,
        )


def load() -> api.Package:
    return Package()


def main():
    pass


if __name__ == "__main__":
    main()
