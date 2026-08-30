# -*- coding: utf-8 -*-

import cv2
import seetapsych_lib.runtime.actions
import seetapsych_lib.runtime.model
import seetapsych_lib.runtime.module
from seetapsych_lib import api


def main():
    module_schema = seetapsych_lib.runtime.module.load_local_module("../seetapsych_face_hub/modules/retinaface.yml")

    # select package you want
    package_schema = module_schema.module.packages[0]
    package = seetapsych_lib.runtime.actions.load_package(package_schema)

    # select model you want
    model_schema = package_schema.models[0]

    # load model
    model = seetapsych_lib.runtime.model.build_model(model_schema, cache_dir=".cache")

    # create instance
    instance = package.create(models=[model], parameters={}, device=api.Device("cpu"))

    # read image and forward
    image = cv2.imread("data/a.jpg")
    report = {}

    # inference
    update = instance.inference(data={"default": image}, report=report)

    # merge report
    if update:
        report.update(update)

    print(report)


if __name__ == "__main__":
    main()
