# 花椒含油率推理 Runbook

适用 artifact：`artifacts/huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.1`

适用环境：`/home/yr/anaconda3/envs/hj`

## 1. 前置检查

```bash
cd /home/yr/yr/code/cv/oil_content_detection
/home/yr/anaconda3/envs/hj/bin/python - <<'PY'
from pathlib import Path
for p in [
    'artifacts/huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.1/model.joblib',
    'artifacts/huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.1/manifest.json',
    'artifacts/huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.1/feature_schema.json',
    'artifacts/huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.1/roi_config.json',
    'artifacts/huajiao-dinov3-roi-seg-v0.1.0/best_model.pt',
    'artifacts/huajiao-dinov3-roi-seg-v0.1.0/manifest.json',
]:
    path = Path(p)
    print(p, 'OK' if path.exists() else 'MISSING')
PY
```

检查 GPU（可选）：

```bash
/home/yr/anaconda3/envs/hj/bin/python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('device_count', torch.cuda.device_count())
print('device0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
PY
```

## 2. CLI 推理

```bash
/home/yr/anaconda3/envs/hj/bin/python scripts/huajiao_predict.py \
  --model-dir artifacts/huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.1 \
  --image /path/to/sample.png \
  --hdr /path/to/capture/REFLECTANCE_sample.hdr \
  --sample-id sample-id \
  --out /tmp/huajiao_prediction.json \
  --debug-dir /tmp/huajiao_debug
```

输出文件：

- `/tmp/huajiao_prediction.json`: 结构化预测结果。
- `/tmp/huajiao_debug/<sample>_dinov3_pepper_mask.png`: DINOv3 polygon 内 gray_otsu refined 二值 mask。
- `/tmp/huajiao_debug/<sample>_dinov3_roi_overlay.png`: ROI overlay。

## 3. 当前真实样本 smoke

已验证样本：`0346_11`

命令：

```bash
mkdir -p /home/yr/yr/tmp/huajiao_dinov3_gray_otsu_refined_smoke_0346_11_v4/debug
/home/yr/anaconda3/envs/hj/bin/python scripts/huajiao_predict.py \
  --model-dir artifacts/huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.1 \
  --image '/home/yr/yr/data/科研数据/湖北金宏椒2/0346_11/0346_11.png' \
  --hdr '/home/yr/yr/data/科研数据/湖北金宏椒2/0346_11/capture/REFLECTANCE_0346_11.hdr' \
  --sample-id 0346_11 \
  --out /home/yr/yr/tmp/huajiao_dinov3_gray_otsu_refined_smoke_0346_11_v4/prediction.json \
  --debug-dir /home/yr/yr/tmp/huajiao_dinov3_gray_otsu_refined_smoke_0346_11_v4/debug
```

验证结果摘要：

- prediction: `1.883649238062088 ml/100g`
- `qc.status`: `ok`
- `roi.mask_source`: `dinov3_roi_gray_otsu_refined`
- `roi.coverage_ratio`: `0.14532130323720396`
- `roi.valid_pixel_count`: `194791`
- strict JSON: `json.dumps(payload, allow_nan=False)` 通过
- JSON: `/home/yr/yr/tmp/huajiao_dinov3_gray_otsu_refined_smoke_0346_11_v4/prediction.json`
- Mask: `/home/yr/yr/tmp/huajiao_dinov3_gray_otsu_refined_smoke_0346_11_v4/debug/0346_11_dinov3_pepper_mask.png`
- Overlay: `/home/yr/yr/tmp/huajiao_dinov3_gray_otsu_refined_smoke_0346_11_v4/debug/0346_11_dinov3_roi_overlay.png`

## 4. Python API

```python
from oil_content_detection.inference import HuajiaoOilPredictor

predictor = HuajiaoOilPredictor.from_artifact(
    'artifacts/huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.1'
)
result = predictor.predict(
    image_path='/path/to/sample.png',
    hdr_path='/path/to/capture/REFLECTANCE_sample.hdr',
    sample_id='sample-id',
    debug_dir='/tmp/huajiao_debug',
)
print(result.to_json())
```

## 5. 常见问题排查

### 5.1 `INPUT_FILE_NOT_FOUND`

原因：图像、`.hdr` 或 `.dat` 文件缺失/路径错误。

处理：确认上传目录包含：

```text
sample.png
capture/REFLECTANCE_sample.hdr
capture/REFLECTANCE_sample.dat
```

### 5.2 `CUBE_IMAGE_SHAPE_MISMATCH`

原因：RGB 图像高宽与 cube `(lines, samples)` 不一致。

处理：在采集/上传侧保证 RGB 与 cube 同视角同尺寸。当前推理代码故意不 resize，以避免 mask 和光谱像素错位。

### 5.3 `WAVELENGTH_SCHEMA_MISMATCH`

原因：cube band 数不等于 artifact `feature_schema.json` 中 224 个波段。

处理：确认使用同一高光谱设备和同一导出流程；不要用缺 band 或重新采样的 cube 调用当前模型。

### 5.4 `ROI_EMPTY` 或 `ROI_LOW_COVERAGE`

原因：DINOv3 coarse polygon 未定位到花椒，或 polygon 内 gray_otsu refinement 后 mask 过小。

处理：检查 debug overlay；确认背景、拍摄距离、曝光、样本摆放与训练数据接近。若 coarse polygon 正常但 refined mask 异常，优先检查 gray_otsu/morphology 配置；必要时扩充 ROI 标注或重新调参。

### 5.5 `ROI_SEGMENTATION_FAILED`

原因：DINOv3 artifact 缺失、hash 不匹配、GPU/CPU 环境问题、输入图像损坏。

处理：

1. 检查 `artifacts/huajiao-dinov3-roi-seg-v0.1.0/manifest.json`。
2. 确认 `best_model.pt` 和 backbone weights 存在且 SHA256 匹配。
3. 若部署机无 GPU，用 CPU 重新发布或把 `roi_config.json` 中 device 设置为 `cpu` 后重新 smoke。

## 6. 质量门

推理变更后至少运行：

```bash
/home/yr/anaconda3/envs/hj/bin/python -m pytest -q tests/inference
/home/yr/anaconda3/envs/hj/bin/python -m pytest -q tests/inference/test_heatmap.py tests/service/test_huajiao_web_app.py
/home/yr/anaconda3/envs/hj/bin/python scripts/huajiao_predict.py --help
/home/yr/anaconda3/envs/hj/bin/python scripts/huajiao_web.py --help
```

Web 展示应用启动、上传 smoke 和热力图检查见 `docs/runbooks/huajiao-web-app.md`。发布前运行完整质量门见 `docs/runbooks/model-release.md`。
