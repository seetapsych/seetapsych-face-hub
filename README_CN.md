# SeetaPsych Face Hub

> SeetaPsych 社区开源人脸模块集合。

简体中文 | [English](README.md)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](pyproject.toml)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)

## 使用说明

本项目已包含在 seetapsych-lib 的默认配置中。通过 `seetapsych-manager download` 即可下载并使用。

具体用法参见 [SeetaPsych](https://github.com/seetapsych/seetapsych-lib)。

也可通过以下方式额外加载本算法模块。

### WebUI

运行 `seetapsych-webui` 时使用 `--files` 参数加载：

```
seetapsych-webui --files seetapsych_face_hub/modules/insightface/retinaface.yml
```

### 编程使用

在程序中添加以下代码以使用本算法模块：

```python
from seetapsych_lib.runtime.factory import Factory
from seetapsych_lib.runtime.pipeline import Pipeline

factory = Factory()
factory.load_file_modules("seetapsych_face_hub/modules/insightface/retinaface.yml")

pipeline = Pipeline(factory, ...)

pipeline.add_attributes("face/detection")
```

完整的端到端示例（含可视化）参见：

* [examples/image_retinaface.py](examples/image_retinaface.py) — 静态图像人脸检测 + 基于 InsightFace RetinaFace 的 5 点关键点。
* [examples/image_mediapipe.py](examples/image_mediapipe.py) — 静态图像人脸检测 + 基于 MediaPipe Tasks 的 468 点 3D 网格。
* [examples/image_arcface.py](examples/image_arcface.py) — 静态图像 RetinaFace + 512 维 ArcFace 嵌入向量，含声纹风格频谱可视化。
* [examples/camera_selection.py](examples/camera_selection.py) — 实时摄像头人脸检测 + 基于 PID 的目标选择与跟踪。

### 模块目录

| 模块 YML | 算法包 |
|---|---|
| [insightface/retinaface.yml](seetapsych_face_hub/modules/insightface/retinaface.yml) | FaceDetection-RetinaFace(InsightFace) |
| [insightface/arcface.yml](seetapsych_face_hub/modules/insightface/arcface.yml) | FaceFeature-ArcFace(InsightFace) |
| [mediapipe.yml](seetapsych_face_hub/modules/mediapipe.yml) | FaceDetection-MediaPipe, FaceMesh-MediaPipe |
| [retinaface.yml](https://github.com/seetapsych/seetapsych-face-hub/blob/main/seetapsych_face_hub/modules/retinaface.yml) | FaceDetection-RetinaFace(PyTorch) |

### InsightFace (RetinaFace)

> InsightFace Buffalo RetinaFace 检测器：人脸边界框 + 5 点关键点检测，提供 500M / 2.5G / 10G FLOPs 三档模型变体。

<div align="center" id="figure-retinaface-result">
  <img src="assets/example-retinaface.jpg" alt="InsightFace RetinaFace 可视化：样本人像上的人脸框、置信度与 5 点关键点" height="480"/>
  <p><em><strong>图 1</strong> InsightFace RetinaFace 输出可视化 — 检测到的人脸、置信度评分与 5 点关键点。</em></p>
</div>

模块配置：[insightface/retinaface.yml](seetapsych_face_hub/modules/insightface/retinaface.yml)

| 算法包名称 | 提供属性 | 依赖属性 |
|---|---|---|
| FaceDetection-RetinaFace(InsightFace) | `face/detection`, `face/landmarks` | *(无)* |

**说明**：InsightFace RetinaFace 检测人脸框 + 5 点关键点，输入尺寸可配置；提供 3 级精度/速度档位。

**参数**

| 名称 | 类型 | 默认值 | 说明与调优建议 |
|---|---|---|---|
| input_size | number[] | [640, 640] | 网络输入尺寸 [宽, 高]，单位像素。更大尺寸提升小人脸召回率但降低推理速度；建议使用正方形输入。 |

**模型**

| 名称 | 推荐 | 备注 |
|---|---|---|
| det_500m.onnx | ✓ | |
| det_2.5g.onnx | | |
| det_10g.onnx | | |

**输出属性**
- `face/detection` — [规格](https://github.com/seetapsych/seetapsych-attributes#facedetection)。
- `face/landmarks` — [规格](https://github.com/seetapsych/seetapsych-attributes#facelandmarks)。

### InsightFace (ArcFace)

> InsightFace Buffalo 人脸特征提取，基于 ArcFace 模型产出 L2 归一化的 512 维嵌入向量。

<div align="center" id="figure-arcface-result">
  <img src="assets/example-arcface.jpg" alt="InsightFace ArcFace 可视化：RetinaFace 检测结果与纵向声纹风格的 512 维嵌入频谱" height="480"/>
  <p><em><strong>图 2</strong> InsightFace ArcFace 输出可视化 — 检测到的人脸，以及带符号幅度热度条的纵向声纹风格 512 维嵌入频谱。</em></p>
</div>

模块配置：[insightface/arcface.yml](seetapsych_face_hub/modules/insightface/arcface.yml)

| 算法包名称 | 提供属性 | 依赖属性 |
|---|---|---|
| FaceFeature-ArcFace(InsightFace) | `face/feature` | `face/landmarks` |

**说明**：从 5 点对齐的人脸裁剪图像中提取 L2 归一化的 512 维人脸嵌入向量，用于识别、聚类或相似度检索。

**参数**：*(无)*

**模型**

| 名称 | 推荐 | 备注 |
|---|---|---|
| w600k_r50.onnx | ✓ | |
| w600k_mbf.onnx | | |

**输出属性**
- `face/feature` — [规格](https://github.com/seetapsych/seetapsych-attributes#facefeature)。

### MediaPipe

> MediaPipe Tasks 人脸检测与 468 点 3D 人脸网格关键点提取，基于 BlazeFace 短距/全距模型。

<div align="center" id="figure-mediapipe-result">
  <img src="assets/example-mediapipe.jpg" alt="MediaPipe Tasks 可视化：样本人像上的人脸检测框与 468 点 3D FaceMesh 关键点" height="480"/>
  <p><em><strong>图 3</strong> MediaPipe Tasks 输出可视化 — 人脸检测边界框与 468 点 3D FaceMesh 关键点。</em></p>
</div>

模块配置：[mediapipe.yml](seetapsych_face_hub/modules/mediapipe.yml)

| 算法包名称 | 提供属性 | 依赖属性 |
|---|---|---|
| FaceDetection-MediaPipe | `face/detection` | *(无)* |
| FaceMesh-MediaPipe | `face/mesh` | `face/detection` |

#### 算法包：FaceDetection-MediaPipe

**说明**：基于 MediaPipe BlazeFace 检测器的人脸边界框检测，提供短距/全距/稀疏三档模型变体。

**参数**

| 名称 | 类型 | 默认值 | 说明与调优建议 |
|---|---|---|---|
| running_mode | selection (IMAGE, VIDEO, LIVE_STREAM) | IMAGE | 任务执行模式。IMAGE 用于单帧静态图像；VIDEO / LIVE_STREAM 用于帧间时序平滑。 |
| min_detection_confidence | number | 0.5 | 返回检测框的最低置信度阈值。值越低捕获人脸越多，但误报率也随之升高。 |
| min_suppression_threshold | number | 0.3 | 非极大值抑制（NMS）的 IoU 阈值。越高保留越多重叠框；越低剪枝越激进。 |

**模型**

| 名称 | 推荐 | 备注 |
|---|---|---|
| blaze_face_full_range.tflite | ✓ | |
| blaze_face_full_range_sparse.tflite | | |
| blaze_face_short_range.tflite | | |

**输出属性**
- `face/detection` — [规格](https://github.com/seetapsych/seetapsych-attributes#facedetection)。

#### 算法包：FaceMesh-MediaPipe

**说明**：从检测到的人脸框中提取 468 点归一化 3D 人脸网格关键点，可选输出 blendshape 与姿态矩阵。

**参数**

| 名称 | 类型 | 默认值 | 说明与调优建议 |
|---|---|---|---|
| running_mode | selection (IMAGE, VIDEO, LIVE_STREAM) | IMAGE | 任务执行模式。当 num_faces = 1 时，VIDEO / LIVE_STREAM 启用时序关键点平滑。 |
| num_faces | integer | 1 | 同时提取关键点的最大人脸数量。仅当 num_faces = 1 时平滑机制生效。 |
| min_face_detection_confidence | number | 0.5 | 初始化关键点定位时，内置人脸检测器所需的最低置信度。 |
| min_face_presence_confidence | number | 0.5 | 输出关键点前帧内存在人脸的最低置信度；用于过滤视频中的偶发丢帧。 |
| min_tracking_confidence | number | 0.5 | 是否复用上一人脸位置而非重新执行完整检测的跟踪置信度阈值。越高漂移越少，但重检测更频繁。 |
| output_face_blendshapes | boolean | false | 是否输出 52 个人脸 blendshape 系数，用于 3D 虚拟形象 / AR 渲染。会增加少量计算开销。 |
| output_facial_transformation_matrixes | boolean | false | 是否输出 4×4 人脸姿态变换矩阵，用于刚性头部姿态估计。 |

**模型**

| 名称 | 推荐 | 备注 |
|---|---|---|
| face_landmarker.task | ✓ | |

**输出属性**
- `face/mesh` — [规格](https://github.com/seetapsych/seetapsych-attributes#facemesh)。

## 参考文献

- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide)
- [InsightFace](https://github.com/deepinsight/insightface)
- [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
