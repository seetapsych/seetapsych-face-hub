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

### Module Catalog

| Module YML | Packages |
|---|---|
| [insightface/retinaface.yml](seetapsych_face_hub/modules/insightface/retinaface.yml) | FaceDetection-RetinaFace(InsightFace) |
| [insightface/arcface.yml](seetapsych_face_hub/modules/insightface/arcface.yml) | FaceFeature-ArcFace(InsightFace) |
| [mediapipe.yml](seetapsych_face_hub/modules/mediapipe.yml) | FaceDetection-MediaPipe, FaceMesh-MediaPipe |
| [retinaface.yml](seetapsych_face_hub/modules/retinaface.yml) | FaceDetection-RetinaFace(PyTorch) |

### InsightFace (RetinaFace)

> InsightFace Buffalo RetinaFace detector: face bounding box + 5-point landmark detection with 500M / 2.5G / 10G FLOPs model variants.

Module config: [insightface/retinaface.yml](seetapsych_face_hub/modules/insightface/retinaface.yml)

| Package Name | Provides Attributes | Requires Attributes |
|---|---|---|
| FaceDetection-RetinaFace(InsightFace) | `face/detection`, `face/landmarks` | *(none)* |

**Description**: InsightFace RetinaFace detecting face boxes + 5-point landmarks with configurable input size; 3 accuracy/speed model tiers

**Parameters**

| Name | Type | Default | Description & Tuning |
|---|---|---|---|
| input_size | number[] | [640, 640] | Network input [width, height] in pixels. Larger sizes improve small-face recall but slow inference. Square inputs are recommended. |

**Models**

| Name | Recommended | Notes |
|---|---|---|
| det_500m.onnx | ✓ | |
| det_2.5g.onnx | | |
| det_10g.onnx | | |

**Output Attributes**
- `face/detection` — [spec](https://github.com/seetapsych/seetapsych-attributes#facedetection).
- `face/landmarks` — [spec](https://github.com/seetapsych/seetapsych-attributes#facelandmarks).

### InsightFace (ArcFace)

> InsightFace Buffalo face feature extraction via ArcFace models producing L2-normalized 512-dim embeddings.

Module config: [insightface/arcface.yml](seetapsych_face_hub/modules/insightface/arcface.yml)

| Package Name | Provides Attributes | Requires Attributes |
|---|---|---|
| FaceFeature-ArcFace(InsightFace) | `face/feature` | `face/landmarks` |

**Description**: Extract L2-normalized 512-dim face embeddings for recognition, clustering, or similarity search from 5-point aligned faces

**Parameters**: *(none)*

**Models**

| Name | Recommended | Notes |
|---|---|---|
| w600k_r50.onnx | ✓ | |
| w600k_mbf.onnx | | |

**Output Attributes**
- `face/feature` — [spec](https://github.com/seetapsych/seetapsych-attributes#facefeature).

### MediaPipe

> MediaPipe Tasks face detection and 468-point 3D face mesh landmarking via BlazeFace short/full-range models.

Module config: [mediapipe.yml](seetapsych_face_hub/modules/mediapipe.yml)

| Package Name | Provides Attributes | Requires Attributes |
|---|---|---|
| FaceDetection-MediaPipe | `face/detection` | *(none)* |
| FaceMesh-MediaPipe | `face/mesh` | `face/detection` |

#### Package: FaceDetection-MediaPipe

**Description**: Face bounding box detection via MediaPipe BlazeFace detector with short/full-range/sparse model variants

**Parameters**

| Name | Type | Default | Description & Tuning |
|---|---|---|---|
| running_mode | selection (IMAGE, VIDEO, LIVE_STREAM) | IMAGE | Task execution mode. IMAGE for single static frames; VIDEO/LIVE_STREAM for time-series smoothing across frames. |
| min_detection_confidence | number | 0.5 | Minimum confidence score for a detection box to be returned. Lower values catch more faces at the cost of false positives. |
| min_suppression_threshold | number | 0.3 | IoU threshold for non-maximum suppression. Higher values keep more overlapping boxes; lower values prune aggressively. |

**Models**

| Name | Recommended | Notes |
|---|---|---|
| blaze_face_full_range.tflite | ✓ | |
| blaze_face_full_range_sparse.tflite | | |
| blaze_face_short_range.tflite | | |

**Output Attributes**
- `face/detection` — [spec](https://github.com/seetapsych/seetapsych-attributes#facedetection).

#### Package: FaceMesh-MediaPipe

**Description**: 468-point normalized 3D face mesh landmark extraction from detected face boxes with optional blendshapes and pose matrices

**Parameters**

| Name | Type | Default | Description & Tuning |
|---|---|---|---|
| running_mode | selection (IMAGE, VIDEO, LIVE_STREAM) | IMAGE | Task execution mode. VIDEO/LIVE_STREAM enable temporal landmark smoothing when num_faces = 1. |
| num_faces | integer | 1 | Maximum number of faces to landmark simultaneously. Smoothing is only active when num_faces is 1. |
| min_face_detection_confidence | number | 0.5 | Minimum confidence for the built-in face detector to succeed when initializing landmark placement. |
| min_face_presence_confidence | number | 0.5 | Minimum confidence that a face is present in the frame before landmark output; filters intermittent dropouts in video. |
| min_tracking_confidence | number | 0.5 | Minimum tracking confidence to reuse the last known face position vs. re-running full detection. Higher values reduce drift but trigger re-detection more often. |
| output_face_blendshapes | boolean | false | Whether to output 52 face blendshape coefficients for 3D avatar/AR rendering. Adds minor compute overhead. |
| output_facial_transformation_matrixes | boolean | false | Whether to output 4x4 facial pose transformation matrices for rigid head pose estimation. |

**Models**

| Name | Recommended | Notes |
|---|---|---|
| face_landmarker.task | ✓ | |

**Output Attributes**
- `face/mesh` — [spec](https://github.com/seetapsych/seetapsych-attributes#facemesh).

## References

- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide)
- [InsightFace](https://github.com/deepinsight/insightface)
- [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
