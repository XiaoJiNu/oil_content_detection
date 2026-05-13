# 花椒含油率 DINOv3 ROI 推理架构

本文档描述当前推荐的 RGB 图像 + ENVI 高光谱 cube 推理工程包：`artifacts/huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.1`。该 artifact 使用 DINOv3 coarse polygon 定位，再在 polygon 内执行 gray_otsu 二次精分割，最终用 refined mask 聚合 cube 光谱。

## 目标与非目标

目标：

1. 应用团队上传同视角、同尺寸 RGB 图像和 ENVI `.hdr/.dat` cube。
2. 系统自动调用 DINOv3 ROI 分割模型得到 coarse polygon mask，并在 polygon 内执行 gray_otsu refinement 得到最终花椒 mask。
3. 系统从 cube 的 mask 区域提取平均光谱和 ROI 质量特征。
4. 系统调用含油率回归模型输出结构化 JSON。

非目标：

1. 不做 RGB/cube 自动配准。
2. 不做 resize 来强行对齐输入。
3. 不接收应用侧上传任意 checkpoint 路径。
4. 不把需要 `weight_g`、`batch`、`sid_num` 的精确复现模型暴露为“只上传 RGB+cube”的部署模型。

## 关键 artifact

| Artifact | 用途 | 说明 |
| --- | --- | --- |
| `artifacts/huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.1` | 当前推荐含油率部署候选模型 | `feature_source=dinov3-roi-cube`，`roi_backend=dinov3`，`post_refinement=gray_otsu` |
| `artifacts/huajiao-dinov3-roi-seg-v0.1.0` | DINOv3 ROI 分割模型 | Val mean Dice≈0.975292，mean IoU≈0.951880 |
| `artifacts/huajiao-report-20260505-snv-sg1-pls10-batch-quality-huber` | 精确复现/内部对照 | 包含 `weight_g/batch/sid_num`，不可作为应用上传部署模型 |

当前含油率部署模型指标：

| Split | n | R² | RMSE | MAE |
| --- | ---: | ---: | ---: | ---: |
| Train | 157 | 0.639776 | 0.265788 | 0.205741 |
| Val | 40 | 0.777996 | 0.212330 | 0.170691 |

## 模块数据流

```text
RGB image (.png/.jpg) ─┐
                       ├─ HuajiaoOilPredictor.predict()
ENVI cube (.hdr/.dat) ─┘
        │
        ├─ load_rgb_image_bgr()
        ├─ load_envi_cube()
        ├─ validate_same_view_shape()
        │     └─ 不一致时抛出 CUBE_IMAGE_SHAPE_MISMATCH
        │
        ├─ DINOv3 ROI backend
        │     ├─ roi_config.json: backend=dinov3
        │     ├─ 解析 dinov3.artifact_dir，相对 oil artifact 目录
        │     ├─ DinoV3ROIPredictor.from_artifact()
        │     ├─ sha256 校验 best_model.pt 与 backbone weights
        │     ├─ predict_image() -> 原图尺寸 coarse polygon boolean mask
        │     ├─ dinov3.post_refinement.enabled=true
        │     └─ refine_mask_in_roi_if_configured(gray_otsu) -> final refined mask
        │
        ├─ extract_roi_spectrum_features()
        │     ├─ mask 区域均值光谱 wl_*
        │     ├─ coverage_ratio
        │     ├─ log_valid_pixel_count
        │     └─ valid_pixel_ratio
        │
        ├─ feature_row_to_frame()
        ├─ model.joblib predict()
        └─ PredictionResult.to_dict()
```

## 主要源码

| 路径 | 责任 |
| --- | --- |
| `src/oil_content_detection/inference/predictor.py` | 公共 Predictor；根据 `roi_config.backend` 分流 traditional/DINOv3 ROI |
| `src/oil_content_detection/inference/artifact.py` | 加载 `model.joblib`、`manifest.json`、`feature_schema.json`、`roi_config.json`，拒绝部署禁用特征 |
| `src/oil_content_detection/inference/features.py` | 校验 RGB/cube/mask 形状，提取 ROI 光谱和质量特征 |
| `src/oil_content_detection/inference/roi.py` | 传统 ROI wrapper、DINOv3 polygon 内 post-refinement helper 与 ROI 质量门 |
| `src/oil_content_detection/inference/heatmap.py` | 使用回归模型对 ROI 像素光谱生成模型响应热力图和 overlay |
| `src/oil_content_detection/service/app.py` | FastAPI Web/API 薄适配层，处理上传、调用 predictor、返回 URL/JSON |
| `src/oil_content_detection/service/storage.py` | 上传文件落盘、扩展名/大小检查、`.hdr` 同名 `.dat` 兼容处理 |
| `src/oil_content_detection/segmentation/predictor.py` | DINOv3 ROI predictor、checkpoint/hash 校验、mask/overlay 输出 |
| `scripts/package_huajiao_model.py` | 从 train/val cube 重提取 DINOv3 ROI 特征并生成含油率 artifact |
| `scripts/huajiao_predict.py` | CLI 推理入口 |
| `scripts/huajiao_web.py` | Web 展示应用启动入口 |

## Web/API 展示层

```text
Browser / curl multipart upload
        │
        ├─ POST /api/predict
        │     ├─ 保存 image/hdr/dat 到 runtime/huajiao_web/uploads/<job_id>/
        │     ├─ 将上传的 dat/raw 复制为 hdr 同名 .dat
        │     ├─ 调用 HuajiaoOilPredictor.predict()
        │     ├─ 写 prediction.json 和 ROI debug 图
        │     ├─ 可选 generate_oil_response_heatmap()
        │     └─ 返回 prediction、visualization URL 和下载 URL
        │
        ├─ GET /api/health
        └─ GET /outputs/<job_id>/...
```

Web 层只负责文件上传、调用核心 predictor、组织静态文件 URL 和展示页面；不允许用户通过请求覆盖模型 artifact、DINOv3 checkpoint 或 backbone 权重路径。热力图是模型响应可视化，不是独立标定的逐像素真实含油率。

## Artifact 文件契约

含油率 artifact 必须包含：

```text
model.joblib
manifest.json
feature_schema.json
preprocess_config.json
roi_config.json
metrics.json
model_card.md
train_predictions.csv
val_predictions.csv
training_features/train_dinov3_roi_features.csv
training_features/val_dinov3_roi_features.csv
```

`roi_config.json` 中 DINOv3 路径必须相对于含油率 artifact 目录保存，例如：

```json
{
  "backend": "dinov3",
  "require_same_shape": true,
  "dinov3": {
    "artifact_dir": "../huajiao-dinov3-roi-seg-v0.1.0",
    "device": "cpu",
    "mask_source": "dinov3_roi_segmentation",
    "post_refinement": {
      "enabled": true,
      "segmentation_mode": "gray_otsu",
      "pepper_open_size": 5,
      "pepper_close_size": 5,
      "hole_fill_max_area": 300,
      "label_fallback_enabled": false,
      "label_max_area_ratio": 0.0
    }
  }
}
```

## 部署输入契约

1. `image_path`: OpenCV/PIL 可读取的 RGB 图像。
2. `hdr_path`: ENVI `.hdr`，旁边必须有匹配 `.dat`。
3. 图像高宽必须等于 cube `(lines, samples)`。
4. cube band 数必须等于 `feature_schema.json` 中波段列数。
5. 设备和 DINOv3 权重路径由受信任 artifact 固定；应用请求不得覆盖。

## 输出契约

成功输出由 `PredictionResult.to_dict()` 生成，核心字段：

```json
{
  "sample_id": "0346_11",
  "target_name": "oil_ml_per_100g",
  "prediction": 1.883649238062088,
  "unit": "ml/100g",
  "model_version": "huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.1",
  "artifact_sha256": "51e05592b632f362226b3683d7b1a58d977822abc60f6f042780058079651206",
  "input_mode": "envi_cube_with_rgb_image",
  "roi": {
    "mask_source": "dinov3_roi_gray_otsu_refined",
    "pepper_pixels": 194791,
    "coverage_ratio": 0.14532130323720396,
    "valid_pixel_count": 194791,
    "valid_pixel_ratio": 1.0,
    "mask_path": "/tmp/.../0346_11_dinov3_pepper_mask.png",
    "overlay_path": "/tmp/.../0346_11_dinov3_roi_overlay.png"
  },
  "qc": {"status": "ok", "warnings": []}
}
```

CLI 错误输出：

```json
{
  "qc": {"status": "error", "warnings": []},
  "error": {"code": "CUBE_IMAGE_SHAPE_MISMATCH", "message": "...", "details": {}}
}
```

## 安全与可维护性约束

1. PyTorch ROI checkpoint 只从内部 artifact 加载；加载前校验 SHA256。
2. `torch.load(..., weights_only=True)` 只加载 head/state-dict。
3. `feature_schema.json` 拒绝 `distill_ml`、`oil_ml_per_gram`、`oil_ml_per_100g`、`weight_g`、`batch`、`sid_num` 等部署禁用字段。
4. Debug mask/overlay 仅写入调用者指定的 `debug_dir`。
5. 质量门和文档必须明确：DINOv3 coarse polygon 分割指标高，不等价于下游含油率指标一定高；下游指标以重新提取后的 train/val metrics 为准。
6. 输出 JSON 必须严格兼容标准 JSON；ROI debug info 中的 NaN/Inf 必须转为 null，`PredictionResult.to_json()` 使用 `allow_nan=False`。
