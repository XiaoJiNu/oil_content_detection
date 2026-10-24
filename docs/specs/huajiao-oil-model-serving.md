# 花椒含油率模型 Serving 规格

版本：2026-05-09

当前 serving 候选 artifact：`artifacts/huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.2`

ROI artifact：`artifacts/huajiao-dinov3-roi-seg-v0.1.0`

## 1. 背景

本规格面向应用团队集成“上传 RGB 图像 + ENVI 高光谱 cube，返回花椒含油率预测”的展示/工程调用场景。

仓库中存在两类模型，必须区分：

1. 精确复现模型：`artifacts/huajiao-report-20260505-snv-sg1-pls10-batch-quality-huber`
   - 用于复现 2026-05-05 报告指标。
   - 输入包含 `weight_g`、`batch`、`sid_num`。
   - 不适合应用团队只上传 RGB+cube 的 serving。
2. 部署候选模型：`artifacts/huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.2`
   - train/val 特征通过 DINOv3 coarse polygon + gray_otsu refined ROI serving 路径重新从 cube 提取。
   - 输入只使用 `wl_*` ROI 平均光谱和 ROI 质量特征；v0.3.2 的模型内部再派生光谱 summary/PLS 特征，不要求应用侧上传额外字段。
   - 禁止使用重量、批次、样本编号和目标/标签字段。

## 2. 输入契约

### 2.1 必填参数

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `image_path` | path/string | RGB 图像，PIL/OpenCV 可读取。 |
| `hdr_path` | path/string | ENVI `.hdr` 文件；旁边必须有对应 `.dat`。 |
| `sample_id` | string/null | 可选；只用于输出和 debug 文件命名，不作为模型特征。 |
| `debug_dir` | path/string/null | 可选；提供时输出 ROI mask 和 overlay。 |

### 2.2 输入数据要求

1. RGB 图像和 cube 必须同视角；不支持旋转、平移、裁剪或复杂配准。
2. RGB 高宽与 cube `(lines, samples)` 完全一致时直接推理；任一维度相差不超过 5 像素时，推理端自动将 RGB resize 到 cube 高宽后再做 ROI、热力图和含油率预测；超过 5 像素返回 `CUBE_IMAGE_SHAPE_MISMATCH`。
3. 推理代码不 resize cube；cube `(lines, samples)` 始终是光谱聚合基准。
4. cube band 数必须与 artifact `feature_schema.json` 中 `wavelength_columns` 数量一致；当前为 224 个 `wl_*` 波段。
5. ENVI header 中若缺少 wavelength list，允许按 band 顺序映射，但 `qc.warnings` 会记录警告。
6. 应用请求不得指定任意模型 checkpoint、DINOv3 backbone 权重或 artifact 之外的文件路径作为模型权重来源。

## 3. 输出 JSON schema

### 3.1 成功响应

```json
{
  "sample_id": "0346_11",
  "target_name": "oil_ml_per_100g",
  "prediction": 1.883649238062088,
  "unit": "ml/100g",
  "model_version": "huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.2",
  "artifact_sha256": "dbdafd38e061f581b583b9e21f8550d830df4bc7e0e55a56845a21486e43b2c0",
  "input_mode": "envi_cube_with_rgb_image",
  "roi": {
    "mask_source": "dinov3_roi_gray_otsu_refined",
    "pepper_pixels": 194791,
    "pixel_count": 194791,
    "valid_pixel_count": 194791,
    "coverage_ratio": 0.14532130323720396,
    "valid_pixel_ratio": 1.0,
    "mask_path": "/tmp/huajiao_debug/0346_11_dinov3_pepper_mask.png",
    "overlay_path": "/tmp/huajiao_debug/0346_11_dinov3_roi_overlay.png",
    "info": {
      "backend": "dinov3",
      "dinov3_artifact_dir": ".../artifacts/huajiao-dinov3-roi-seg-v0.1.0",
      "dinov3_threshold": 0.5,
      "dinov3_coverage_ratio": 0.4697825152788388,
      "post_refinement_enabled": 1.0,
      "post_refinement_mode": "gray_otsu",
      "dinov3_polygon_coverage_ratio": 0.4697825152788388,
      "post_refined_coverage_ratio": 0.14532130323720396,
      "shape_alignment": {
        "enabled": true,
        "method": "resize_rgb_to_cube_hw",
        "resized": false,
        "image_shape_hw": [1310, 1024],
        "cube_shape_hw": [1310, 1024],
        "delta_hw": [0, 0],
        "max_delta_px": 5,
        "aligned_image_path": "/path/to/original-or-debug/aligned_rgb.png"
      }
    }
  },
  "qc": {
    "status": "ok",
    "warnings": []
  }
}
```

### 3.2 字段说明

| 字段 | 说明 |
| --- | --- |
| `prediction` | 含油率预测值。 |
| `unit` | 当前为 `ml/100g`。 |
| `artifact_sha256` | `model.joblib` SHA256，用于追溯。 |
| `roi.mask_source` | 当前推荐候选固定输出最终 refined mask 来源：`dinov3_roi_gray_otsu_refined`。 |
| `roi.pepper_pixels` | mask 内像素数。 |
| `roi.coverage_ratio` | mask 像素数 / 图像总像素数。 |
| `roi.valid_pixel_count` | mask 内有限光谱像素数。 |
| `roi.valid_pixel_ratio` | 有效光谱像素数 / mask 像素数。 |
| `roi.mask_path` | debug mask 路径；未传 `debug_dir` 时为 null。 |
| `roi.overlay_path` | debug overlay 路径；未传 `debug_dir` 时为 null。 |
| `qc.status` | 成功时 `ok`，CLI 捕获错误时 `error`。 |
| `qc.warnings` | 非致命质量/波长/自动 RGB resize 警告。 |
| `roi.info.shape_alignment` | RGB/cube 高宽校验与自动 resize 审计信息；`resized=true` 时 `aligned_image_path` 指向服务端生成的对齐 RGB 图。 |

## 4. 错误响应和错误码

CLI 错误响应形态：

```json
{
  "qc": {"status": "error", "warnings": []},
  "error": {
    "code": "CUBE_IMAGE_SHAPE_MISMATCH",
    "message": "RGB image and hyperspectral cube must be the same view; automatic RGB resize is allowed only when height and width differ by no more than 5 pixels.",
    "details": {"image_shape_hw": [1316, 1024], "cube_shape_hw": [1310, 1024], "delta_hw": [6, 0], "max_allowed_delta_px": 5}
  }
}
```

| 错误码 | 含义 | 应用侧建议 |
| --- | --- | --- |
| `INPUT_FILE_NOT_FOUND` | RGB、HDR 或 DAT 文件不存在/不可读 | 检查上传路径和文件完整性 |
| `UNSUPPORTED_SPECTRAL_FORMAT` | cube 不是 `(height,width,bands)` | 检查 ENVI 文件格式 |
| `WAVELENGTH_SCHEMA_MISMATCH` | cube band 数或特征 schema 不匹配 | 确认使用同一采集设备/波段配置 |
| `UNSUPPORTED_FEATURE_SCHEMA` | artifact 需要 serving 无法生成的字段 | 更换合规 artifact |
| `ROI_EMPTY` | ROI mask 为空或无有效光谱像素 | 检查图像内容/背景/曝光 |
| `ROI_LOW_COVERAGE` | ROI 像素数或覆盖率低于质量门 | 检查样本摆放或 ROI 模型适配性 |
| `CUBE_IMAGE_SHAPE_MISMATCH` | RGB 与 cube 任一维度差值 >5 像素，或 ROI mask 与 cube 对齐失败 | 确认同视角采集；重新导出/裁剪到更接近的高宽；<=5px 可由服务自动 resize RGB |
| `MODEL_ARTIFACT_MISSING` | 模型文件或 DINOv3 artifact 缺失 | 检查部署包完整性 |
| `ROI_SEGMENTATION_FAILED` | DINOv3 ROI 推理失败 | 检查 GPU/CPU 环境、权重 hash、输入图像 |
| `MODEL_PREDICTION_FAILED` | 回归模型输出形状异常 | 检查 artifact 与代码版本 |
| `UNEXPECTED_ERROR` | CLI 未预期异常 | 查看日志并升级为明确错误码 |

## 5. 当前指标和验收

当前 DINOv3 ROI + gray_otsu refined 部署候选模型：

| Split | n | R² | RMSE | MAE |
| --- | ---: | ---: | ---: | ---: |
| Train | 157 | 0.645112 | 0.263812 | 0.203680 |
| Val | 40 | 0.800960 | 0.201049 | 0.162341 |

DINOv3 ROI 分割模型：

| Split | n | mean Dice | mean IoU |
| --- | ---: | ---: | ---: |
| Val | 40 | 0.975292 | 0.951880 |

验收标准：

1. `HuajiaoOilPredictor.from_artifact()` 能加载当前 artifact。
2. CLI 能对真实样本输出 JSON，且 `qc.status=ok`。
3. Debug 目录能输出 DINOv3 mask 和 overlay。
4. `roi.mask_source=dinov3_roi_gray_otsu_refined`。
5. `feature_schema.json` 不包含 `weight_g`、`batch`、`sid_num`、目标/标签字段。
6. RGB/cube 高宽完全一致或每维差值 <=5px 时必须成功进入 ROI/预测；发生自动 resize 时 `roi.info.shape_alignment.resized=true` 且 `qc.warnings` 记录 resize。
7. RGB/cube 任一维度差值 >5px 时必须返回 `CUBE_IMAGE_SHAPE_MISMATCH`。
8. 成功响应必须能用 `json.dumps(payload, allow_nan=False)` 序列化，ROI info 中非有限数必须输出为 `null`。

## 6. 非功能约束

1. 推理环境推荐 `/home/yr/anaconda3/envs/hj`。
2. GPU 可用时 DINOv3 ROI 使用 `cuda`；部署环境无 GPU 时可将 `roi_config.json` 中 device 设为 `cpu` 重新发布，但必须重跑 smoke。
3. PyTorch checkpoint 和 backbone weights 必须通过 `manifest.json` SHA256 校验。
4. 大文件 artifact 不应随意提交；交付前应确认仓库/存储策略。
