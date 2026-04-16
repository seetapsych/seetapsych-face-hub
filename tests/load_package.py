# -*- coding: utf-8 -*-

import cv2

import fabopsy_lib.runtime.actions
import fabopsy_lib.runtime.module
import fabopsy_lib.runtime.model
from fabopsy_lib import api

def main():
    module_schema = fabopsy_lib.runtime.module.load_local_module('../fabopsy_face/modules/retinaface.yml')

    # select package you want
    package_schema = module_schema.module.packages[0]
    package = fabopsy_lib.runtime.actions.load_package(package_schema)

    # select model you want
    model_schema = package_schema.models[0]

    # load model
    model = fabopsy_lib.runtime.model.build_model(model_schema, cache_dir='.cache')

    # create instance
    instance = package.create(models=[model], parameters={}, device=api.Device('cpu'))

    # read image and forward
    image = cv2.imread('data/a.jpg')
    report = {}

    # inference
    update = instance.inference(data={
        'default': image
    }, report=report)

    # merge report
    if update:
        report.update(update)

    print(report)


if __name__ == '__main__':
    main()
