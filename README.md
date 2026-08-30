# SeetaPsych Face Hub

> Community open-source face modules for SeetaPsych

## Usage

This project is already included in the seetapsych-lib default configuration. Download and use it via `seetapsych-manager download`.

For usage, refer to [SeetaPsych](https://github.com/seetapsych/seetapsych-lib).

You can additionally add this algorithm module using the following methods.

### WebUI

Run `seetapsych-webui` with the `--dirs` argument to use it.

```
seetapsych-webui --files seetapsych_face_hub/modules/insightface/retinaface.yml
```

### Programmatic Usage

Add the following code in your program to use this algorithm module.
```python
from seetapsych_lib.runtime.factory import Factory
from seetapsych_lib.runtime.pipeline import Pipeline

factory = Factory()
factory.load_file_modules("seetapsych_face_hub/modules/insightface/retinaface.yml")

pipeline = Pipeline(factory, ...)

pipeline.add_attributes("face/detection")
```

## Introduction

### Retinaface(InsightFace)

A face detection model from InsightFace's Buffalo release, capable of detecting faces and 5-point facial landmarks simultaneously. ONNX-based with configurable input size (default 640x640) for speed/accuracy tradeoff.

Module config: [retinaface.yml](seetapsych_face_hub/modules/insightface/retinaface.yml).
Provide Attributes: `face/detection` `face/landmarks`.

Available models: `det_500m.onnx` (recommended, 500M FLOPs, lightweight), `det_2.5g.onnx` (2.5G FLOPs, balanced), `det_10g.onnx` (10G FLOPs, high accuracy).

### ArcFace(InsightFace)

A face feature extraction (recognition) model from InsightFace's Buffalo release, capable of extracting L2-normalized 512-dim face embedding vectors. Requires 5-point face landmarks for similarity transform alignment before feature extraction.

Module config: [arcface.yml](seetapsych_face_hub/modules/insightface/arcface.yml).
Provide Attributes: `face/feature`.

Available models: `w600k_r50.onnx` (recommended, ResNet-50 backbone, higher accuracy), `w600k_mbf.onnx` (MobileFaceNet backbone, lightweight, faster).

### FaceMesh(MediaPipe)

A suite of MediaPipe Tasks vision solutions including BlazeFace face detector and Face Landmarker. Face Mesh extracts 468 normalized 3D face landmarks covering the entire facial geometry. Detection and mesh are provided as two separate packages:

- **Face Detector**: Provides `face/detection`. Supports BlazeFace models in short-range, full-range, and sparse variants.
- **Face Mesh**: Provides `face/mesh`, requires `face/detection`. Outputs normalized 3D landmarks with optional face blendshapes and facial transformation matrices.

Module config: [mediapipe.yml](seetapsych_face_hub/modules/mediapipe.yml).
Provide Attributes: `face/detection` `face/mesh`.

Detector models: `blaze_face_full_range.tflite` (recommended), `blaze_face_full_range_sparse.tflite`, `blaze_face_short_range.tflite`.
Mesh model: `face_landmarker.task` (recommended).

### Retinaface(Ver. PyTorch)

PyTorch implementation of RetinaFace (Biubug6), a single-stage dense face localisation model. Detects faces and 5-point landmarks with configurable confidence threshold (default 0.6). Models are distributed in safetensors format.

Module config: [retinaface.yml](seetapsych_face_hub/modules/retinaface.yml).
Provide Attributes: `face/detection` `face/landmarks`.

Available models: `mobilenet0.25_Final.safetensors` (recommended, MobileNet-0.25 backbone, lightweight), `Resnet50_Final.safetensors` (ResNet-50 backbone, higher accuracy).

## References

- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide)
- [InsightFace](https://github.com/deepinsight/insightface)
- [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
