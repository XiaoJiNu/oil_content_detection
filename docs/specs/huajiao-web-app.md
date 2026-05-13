# 花椒含油率 Web 应用规格

版本：2026-05-12

适用 artifact：`artifacts/huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.1`

## 1. 目标

本规格定义一个可直接部署给应用方使用的展示 Web 应用：用户在浏览器上传 RGB 图像、ENVI `.hdr` 和 ENVI 数据文件 `.dat/.raw`，系统返回花椒含油率预测，并展示 ROI mask、ROI overlay、模型响应热力图等可视化结果。

该 Web 层只是 `HuajiaoOilPredictor` 的薄适配器；模型输入约束、DINOv3 ROI、gray_otsu 二次精分割、feature schema 和错误码仍以 `docs/specs/huajiao-oil-model-serving.md` 为准。

## 2. 输入契约

`POST /api/predict` 使用 `multipart/form-data`：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `image` | file | 是 | RGB 图像，允许 `.png/.jpg/.jpeg/.bmp`。 |
| `hdr` | file | 是 | ENVI header，必须是 `.hdr`。 |
| `dat` | file | 是 | ENVI 二进制数据，允许 `.dat/.raw`。服务端会把该文件复制为与 `.hdr` 同名 `.dat`，以兼容当前 ENVI reader。 |
| `sample_id` | string | 否 | 只用于结果展示和 debug 文件命名，不作为模型特征。 |
| `generate_heatmap` | bool | 否 | 默认 true；false 时只输出预测和 ROI 可视化，不生成热力图。 |

约束：

1. RGB 图像高宽必须与 ENVI cube `(lines, samples)` 一致。
2. cube band 数必须与 artifact `feature_schema.json` 中的 224 个 `wl_*` 波段一致。
3. 应用请求不能覆盖模型路径、DINOv3 checkpoint 路径或任意权重路径；`--model-dir`/`HUAJIAO_MODEL_DIR` 只允许由受信任运维在服务启动时配置。
4. 默认单文件上传上限由 `HUAJIAO_MAX_UPLOAD_MB` 或 `--max-upload-mb` 控制，默认 2048 MiB。

## 3. HTTP API

### 3.1 `GET /api/health`

成功示例：

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.1",
  "device": "cpu"
}
```

模型加载失败时返回 HTTP 503，`model_loaded=false`，对外只返回固定错误摘要 `model health check failed`；详细异常进入服务日志，避免把内部路径暴露给浏览器或调用方。

### 3.2 `POST /api/predict`

成功响应示例：

```json
{
  "job_id": "7d93a0790c44",
  "prediction": {
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
      "pixel_count": 194791,
      "valid_pixel_count": 194791,
      "coverage_ratio": 0.14532130323720396,
      "valid_pixel_ratio": 1.0,
      "mask_path": "runtime/huajiao_web/outputs/.../debug/0346_11_dinov3_pepper_mask.png",
      "overlay_path": "runtime/huajiao_web/outputs/.../debug/0346_11_dinov3_roi_overlay.png",
      "info": {}
    },
    "qc": {"status": "ok", "warnings": []}
  },
  "visualizations": {
    "image_url": "/outputs/<job_id>/input/0346_11.png",
    "mask_url": "/outputs/<job_id>/debug/0346_11_dinov3_pepper_mask.png",
    "overlay_url": "/outputs/<job_id>/debug/0346_11_dinov3_roi_overlay.png",
    "heatmap_url": "/outputs/<job_id>/visualizations/oil_heatmap.png",
    "heatmap_overlay_url": "/outputs/<job_id>/visualizations/oil_heatmap_overlay.png"
  },
  "downloads": {
    "prediction_json_url": "/outputs/<job_id>/prediction.json",
    "response_json_url": "/outputs/<job_id>/response.json"
  },
  "notes": ["heatmap_is_model_response_not_pixel_ground_truth"]
}
```

错误响应示例：

```json
{
  "job_id": "7d93a0790c44",
  "qc": {"status": "error", "warnings": []},
  "error": {
    "code": "CUBE_IMAGE_SHAPE_MISMATCH",
    "message": "RGB image and hyperspectral cube must be the same view and same size.",
    "details": {"image_shape_hw": [4, 5], "cube_shape_hw": [5, 5]}
  }
}
```

HTTP 状态：

| HTTP | code | 含义 |
| ---: | --- | --- |
| 200 | 无 | 成功预测。 |
| 400 | `INVALID_UPLOAD` | 文件扩展名、空文件或大小限制错误。 |
| 400 | 继承 predictor 错误码 | 如 `CUBE_IMAGE_SHAPE_MISMATCH`、`ROI_EMPTY`、`WAVELENGTH_SCHEMA_MISMATCH`。 |
| 500 | `UNEXPECTED_ERROR` | 未预期服务端异常，应查看 `runtime/huajiao_web/outputs/<job_id>/error.json` 和服务日志。 |

## 4. 可视化语义

1. `mask_url`：最终用于含油率预测的 ROI mask；当前推荐模型应为 `dinov3_roi_gray_otsu_refined`。
2. `overlay_url`：ROI mask 叠加到 RGB 图像的检查图，用于人工确认 ROI 是否合理。
3. `heatmap_url`：透明背景 PNG，显示模型对 ROI 内像素光谱的相对响应。
4. `heatmap_overlay_url`：模型响应热力图叠加到 RGB 图像。

重要限制：热力图是“样本级回归模型对逐像素光谱的响应可视化”，用于展示和 QC，不是经过独立标定的逐像素真实含油率地面真值。

## 5. 持久化输出

默认目录：

```text
runtime/huajiao_web/uploads/<job_id>/
runtime/huajiao_web/outputs/<job_id>/
  input/<uploaded-image>
  debug/*_dinov3_pepper_mask.png
  debug/*_dinov3_roi_overlay.png
  visualizations/oil_heatmap.png
  visualizations/oil_heatmap_overlay.png
  visualizations/oil_heatmap_stats.json
  prediction.json
  response.json
  error.json  # 仅失败时
```

`runtime/` 已加入 `.gitignore`，生产部署应把 `output_root` 放在受控磁盘/挂载点，并通过定时清理或对象存储生命周期策略管理历史上传文件，避免长期运行后磁盘无限增长。

## 6. 验收标准

1. `GET /api/health` 返回 HTTP 200 且 `model_loaded=true`。
2. 用验证集真实样本上传 RGB + `.hdr` + `.dat`，`POST /api/predict` 返回 HTTP 200。
3. 响应 JSON 可用 `json.dumps(payload, allow_nan=False)` 严格序列化。
4. 响应包含预测值、模型版本、ROI 质量指标和可打开的图片 URL。
5. `prediction.json`、`response.json`、ROI mask/overlay 和热力图文件写入 job 输出目录。
6. 输入 shape 不一致时返回结构化错误，不能静默 resize 或配准。
