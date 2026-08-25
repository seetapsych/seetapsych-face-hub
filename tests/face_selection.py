# -*- coding: utf-8 -*-
import json
import os

import cv2
import numpy

from seetapsych_lib.runtime.factory import Factory
from seetapsych_lib.runtime.pipeline import Pipeline
from seetapsych_lib.runtime.runner import Runner

module_root = os.path.join(os.path.dirname(__file__), '../seetapsych_face_hub/modules')


def main():
    factory = Factory()
    factory.load_builtin_modules()
    factory.load_dir_modules(module_root)

    pipeline = Pipeline(factory, packages=[
        'c938b879-44db-45b0-9a5d-8377f0ace5e5', # insightface's face/detection
        'bb212f54-aace-438f-9cb7-f6519f4fef48', # face/selection
        '8e646eec-e50f-4102-a658-1449d04296fb', # face/mesh
    ])
    pipeline.set_models(
        'c938b879-44db-45b0-9a5d-8377f0ace5e5',
        ['1deb39b1-1074-4f6e-b81e-a6a843d011eb']    # using minimum retinaface model
    )

    # print(pipeline.config.model_dump_json(indent=2, exclude_none=True))

    pipeline.solve()

    # print(pipeline.config.model_dump_json(indent=2, exclude_none=True))

    print(pipeline.problem())
    print(pipeline.satisfied())

    runner = Runner(pipeline)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print('Could not open camera')
        exit(1)

    while True:
        ok = cap.grab()
        if not ok:
            break
        ok, frame = cap.retrieve()
        if not ok:
            break
        report = runner.run(data={
            'default': frame
        })
        frame_height, frame_width = frame.shape[:2]
        face_detection = report.get('face_detection', [])
        face_selection = report.get('face_selection', {'pid': 0})
        face_mesh = report.get('face_mesh', [])
        for bbox, mesh in zip(face_detection, face_mesh):
            xyxy = bbox['xyxy']
            # score = bbox['score']
            xyxy = list(map(int, xyxy))

            point3ds = mesh['normalized_3d_landmarks']
            point3ds = numpy.asarray(point3ds).reshape([-1, 3])
            point2ds = point3ds[:, :2]
            point2ds *= numpy.asarray([frame_width, frame_height], dtype=float)

            cv2.rectangle(frame, xyxy[:2], xyxy[2:], (255, 0, 0), 2)
            for p in point2ds:
                p = list(map(int, p))
                cv2.circle(frame, p, 2, (0, 255, 0), -1)

        # print(face_detection)

        frame: numpy.ndarray = cv2.flip(frame, 1)

        if face_detection:
            xyxy = face_detection[0]['xyxy']
            xyxy = list(map(int, xyxy))
            p = [frame.shape[1] - xyxy[2], xyxy[1]]
            pid = face_selection['pid']

            cv2.putText(frame, f'{pid}', p, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('face', frame)
        key = cv2.waitKey(1)
        if key >= 0:
            break


if __name__ == '__main__':
    main()
