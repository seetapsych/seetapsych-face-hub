# SeetaPsych Face Hub

> SeetaPsych 社区维护的开源人脸算法模块，覆盖检测、关键点、人脸网格与特征提取。

简体中文 | [English](README.md)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](pyproject.toml)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)

## 使用方式

本项目已纳入 seetapsych-lib 默认配置，直接通过 `seetapsych-manager download` 即可下载使用。

具体用法请参考 [SeetaPsych](https://github.com/seetapsych/seetapsych-lib) 主库文档。

如需额外加载本项目的算法模块，可通过以下方式：

### WebUI

运行 `seetapsych-webui` 时通过 `--files` 参数加载：

```
seetapsych-webui --files seetapsych_face_hub/modules/insightface/retinaface.yml
```

### 编程调用

在代码中添加以下内容以加载并使用本算法模块：

```python
from seetapsych_lib.runtime.factory import Factory
from seetapsych_lib.runtime.pipeline import Pipeline

factory = Factory()
factory.load_file_modules("seetapsych_face_hub/modules/insightface/retinaface.yml")

pipeline = Pipeline(factory, ...)

pipeline.add_attributes("face/detection")
```

完整的端到端示例（含可视化效果）参见：

* [examples/image_retinaface.py](examples/image_retinaface.py) — 静态图像人脸检测 + 基于 InsightFace RetinaFace 的 5 点关键点提取。
* [examples/image_mediapipe.py](examples/image_mediapipe.py) — 静态图像人脸检测 + 基于 MediaPipe Tasks 的 468 点 3D 人脸网格。
* [examples/image_arcface.py](examples/image_arcface.py) — 静态图像 RetinaFace 检测 + 512 维 ArcFace 嵌入向量提取，含声纹风格频谱可视化。
* [examples/camera_selection.py](examples/camera_selection.py) — 实时摄像头人脸检测 + 基于 PID 的目标筛选与跟踪。

### 模块列表

| 模块 YML | 算法包 |
|---|---|
| [insightface/retinaface.yml](seetapsych_face_hub/modules/insightface/retinaface.yml) | FaceDetection-RetinaFace(InsightFace) |
| [insightface/arcface.yml](seetapsych_face_hub/modules/insightface/arcface.yml) | FaceFeature-ArcFace(InsightFace) |
| [mediapipe.yml](seetapsych_face_hub/modules/mediapipe.yml) | FaceDetection-MediaPipe, FaceMesh-MediaPipe |
| [retinaface.yml](https://github.com/seetapsych/seetapsych-face-hub/blob/main/seetapsych_face_hub/modules/retinaface.yml) | FaceDetection-RetinaFace(PyTorch) |

### InsightFace (RetinaFace)

> InsightFace Buffalo 系列 RetinaFace 检测器：输出人脸边界框与 5 点关键点，提供 500M / 2.5G / 10G FLOPs 三档计算量的模型变体。

<div align="center" id="figure-retinaface-result">
  <img src="assets/example-retinaface.jpg" alt="InsightFace RetinaFace 可视化：样本人像上的人脸框、置信度与 5 点关键点" height="480"/>
  <p><em><strong>图 1</strong> InsightFace RetinaFace 输出可视化 — 检测到的人脸、置信度评分及 5 点关键点。</em></p>
</div>

模块配置：[insightface/retinaface.yml](seetapsych_face_hub/modules/insightface/retinaface.yml)

| 算法包名称 | 提供属性 | 依赖属性 |
|---|---|---|
| FaceDetection-RetinaFace(InsightFace) | `face/detection`, `face/landmarks` | *(无)* |

**说明**：InsightFace RetinaFace 检测器，输出人脸边界框与 5 点关键点，输入尺寸可配置；提供 3 档精度与速度权衡的模型。

**参数**

| 名称 | 类型 | 默认值 | 说明与调优建议 |
|---|---|---|---|
| input_size | number[] | [640, 640] | 网络输入尺寸 [宽, 高]，单位为像素。尺寸越大，小人脸召回率越高，但推理速度越慢；推荐使用正方形输入。 |

**模型**

| 名称 | 推荐 | 备注 |
|---|---|---|
| det_500m.onnx | ✓ | |
| det_2.5g.onnx | | |
| det_10g.onnx | | |

**输出属性**
- `face/detection` — [规格定义](https://github.com/seetapsych/seetapsych-attributes#facedetection)。
- `face/landmarks` — [规格定义](https://github.com/seetapsych/seetapsych-attributes#facelandmarks)。

### InsightFace (ArcFace)

> InsightFace Buffalo 系列人脸特征提取器，基于 ArcFace 模型生成 L2 归一化的 512 维人脸嵌入向量。

<div align="center" id="figure-arcface-result">
  <img src="assets/example-arcface.jpg" alt="InsightFace ArcFace 可视化：RetinaFace 检测结果与纵向声纹风格的 512 维嵌入频谱" height="480"/>
  <p><em><strong>图 2</strong> InsightFace ArcFace 输出可视化 — 检测到的人脸，以及带符号幅度热度条的纵向声纹风格 512 维嵌入频谱。</em></p>
</div>

模块配置：[insightface/arcface.yml](seetapsych_face_hub/modules/insightface/arcface.yml)

| 算法包名称 | 提供属性 | 依赖属性 |
|---|---|---|
| FaceFeature-ArcFace(InsightFace) | `face/feature` | `face/landmarks` |

**说明**：基于 5 点对齐后的人脸裁剪图像，提取 L2 归一化的 512 维人脸嵌入向量，可用于人脸识别、聚类或相似度检索等下游任务。

**参数**：*(无)*

**模型**

| 名称 | 推荐 | 备注 |
|---|---|---|
| w600k_r50.onnx | ✓ | |
| w600k_mbf.onnx | | |

**输出属性**
- `face/feature` — [规格定义](https://github.com/seetapsych/seetapsych-attributes#facefeature)。

### MediaPipe

> MediaPipe Tasks 人脸检测与 468 点 3D 人脸网格关键点提取，基于 BlazeFace 短距 / 全距系列模型。

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

**说明**：基于 MediaPipe BlazeFace 检测器的人脸边界框检测，提供短距、全距、稀疏三档模型变体。

**参数**

| 名称 | 类型 | 默认值 | 说明与调优建议 |
|---|---|---|---|
| running_mode | selection (IMAGE, VIDEO, LIVE_STREAM) | IMAGE | 任务执行模式。`IMAGE` 适用于单帧静态图像；`VIDEO` / `LIVE_STREAM` 适用于需要帧间时序平滑的视频场景。 |
| min_detection_confidence | number | 0.5 | 返回检测结果的最低置信度阈值。阈值越低，检出的人脸越多，但误报率也越高。 |
| min_suppression_threshold | number | 0.3 | 非极大值抑制（NMS）的 IoU 阈值。阈值越高，保留的重叠框越多；阈值越低，剪枝越激进。 |

**模型**

| 名称 | 推荐 | 备注 |
|---|---|---|
| blaze_face_full_range.tflite | ✓ | |
| blaze_face_full_range_sparse.tflite | | |
| blaze_face_short_range.tflite | | |

**输出属性**
- `face/detection` — [规格定义](https://github.com/seetapsych/seetapsych-attributes#facedetection)。

#### 算法包：FaceMesh-MediaPipe

**说明**：从检测到的人脸边界框中提取 468 点归一化 3D 人脸网格关键点，支持可选输出 blendshape 系数与姿态矩阵。

**参数**

| 名称 | 类型 | 默认值 | 说明与调优建议 |
|---|---|---|---|
| running_mode | selection (IMAGE, VIDEO, LIVE_STREAM) | IMAGE | 任务执行模式。当 `num_faces = 1` 时，`VIDEO` / `LIVE_STREAM` 模式会对关键点启用时序平滑。 |
| num_faces | integer | 1 | 同时提取关键点的人脸数量上限。仅当该值为 1 时，时序平滑机制才会生效。 |
| min_face_detection_confidence | number | 0.5 | 初始化关键点定位时，内置人脸检测器所需的最低置信度。 |
| min_face_presence_confidence | number | 0.5 | 输出关键点前帧内存在人脸的最低置信度；用于过滤视频序列中的偶发丢帧。 |
| min_tracking_confidence | number | 0.5 | 跟踪置信度阈值。高于该阈值时复用已知人脸位置，否则重新执行完整检测。值越高，位置漂移越少，但重检测触发更频繁。 |
| output_face_blendshapes | boolean | false | 是否输出 52 个人脸 blendshape 系数，用于 3D 虚拟形象或 AR 渲染。启用后会增加少量计算开销。 |
| output_facial_transformation_matrixes | boolean | false | 是否输出 4×4 人脸姿态变换矩阵，用于刚性头部姿态估计。 |

**模型**

| 名称 | 推荐 | 备注 |
|---|---|---|
| face_landmarker.task | ✓ | |

**输出属性**
- `face/mesh` — [规格定义](https://github.com/seetapsych/seetapsych-attributes#facemesh)。

## 参考文献

- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide)
- [InsightFace](https://github.com/deepinsight/insightface)
- [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
