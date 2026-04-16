# -*- coding: utf-8 -*-
import json
import os

import cv2

from fabopsy_lib.runtime.factory import Factory
from fabopsy_lib.runtime.pipeline import Pipeline
from fabopsy_lib.runtime.runner import Runner

module_root = os.path.join(os.path.dirname(__file__), '../fabopsy_face/modules')


def main():
    factory = Factory()
    factory.load_dir_modules(module_root)

    pipeline = Pipeline(factory, attributes=['face/feature'])

    print(pipeline.config.model_dump_json(indent=2, exclude_none=True))

    pipeline.solve()

    print(pipeline.config.model_dump_json(indent=2, exclude_none=True))

    print(pipeline.problem())
    print(pipeline.satisfied())

    runner = Runner(pipeline)

    image = cv2.imread('data/a.jpg')
    report = runner.run(data={
        'default': image
    })
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
