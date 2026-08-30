# -*- coding: utf-8 -*-
import json
import os

import cv2
from seetapsych_lib.runtime.factory import Factory
from seetapsych_lib.runtime.pipeline import Pipeline
from seetapsych_lib.runtime.runner import Runner

module_root = os.path.join(os.path.dirname(__file__), "../seetapsych_face_hub/modules")


def main():
    factory = Factory(disable_default=True)
    # 加载内置的算法模块
    # factory.load_builtin_modules()
    # 加载额外的算法模块
    factory.load_dir_modules(module_root)

    # 快速构建工作流，声明需要计算的属性是人脸特征 'face/feature'
    pipeline = Pipeline(factory, attributes=["face/feature"])

    # 检测是否有需要 solve 解决的依赖或缺失问题
    print(pipeline.problem())
    # 解决工作流依赖，自动添加上人脸检测和对应的模型
    pipeline.solve()

    # 检测是否有需要 install 和 download 解决的运行环境问题
    print(pipeline.satisfied())
    # 安装当前 pipeline 运行缺失的依赖
    pipeline.install_requirements()
    # 下载 pipeline 运行缺失的模型
    pipeline.cache_models()

    # 构建执行器
    runner = Runner(pipeline)

    # 运行算法
    report = runner.run(data={"default": cv2.imread("data/a.jpg")})

    # 打印运行结果
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
