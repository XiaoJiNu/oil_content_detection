# 花椒含油率 Web 应用 Runbook

适用 artifact：`artifacts/huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.2`

适用环境：`/home/yr/anaconda3/envs/hj`

## 1. 安装/检查依赖

仓库当前开发环境已经安装 FastAPI、uvicorn、python-multipart、httpx、OpenCV、PyTorch 等依赖。新服务器部署时可使用：

```bash
cd /home/yr/yr/code/cv/oil_content_detection
/home/yr/anaconda3/envs/hj/bin/python -m pip install -r requirements-serving.txt
```

如果部署服务器使用 GPU，需要按服务器 CUDA/driver 版本安装匹配的 `torch`/`torchvision` wheel。当前模型也可以 CPU 推理，但 DINOv3 ROI 会更慢。

## 2. 启动服务

`--model-dir` 是受信任运维启动参数，不应暴露成浏览器/API 可传字段；应用请求只能上传样本文件，不能选择任意模型或 checkpoint。

开发/单机演示启动：

```bash
cd /home/yr/yr/code/cv/oil_content_detection
/home/yr/anaconda3/envs/hj/bin/python scripts/huajiao_web.py \
  --model-dir artifacts/huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.2 \
  --host 0.0.0.0 \
  --port 8000 \
  --upload-root runtime/huajiao_web/uploads \
  --output-root runtime/huajiao_web/outputs \
  --heatmap-max-pixels 200000
```

浏览器打开：

```text
http://<server-ip>:8000/
```

健康检查：

```bash
curl -s http://127.0.0.1:8000/api/health | jq .
```

## 3. 命令行上传 smoke

以验证集真实样本 `0286_11` 为例：

```bash
curl -s -X POST http://127.0.0.1:8000/api/predict \
  -F sample_id=0286_11 \
  -F generate_heatmap=true \
  -F image=@'/home/yr/yr/data/科研数据/枸椒5/0286_11/0286_11.png' \
  -F hdr=@'/home/yr/yr/data/科研数据/枸椒5/0286_11/capture/REFLECTANCE_0286_11.hdr' \
  -F dat=@'/home/yr/yr/data/科研数据/枸椒5/0286_11/capture/REFLECTANCE_0286_11.dat' \
  | jq .
```

成功时应看到：

- `prediction.qc.status = "ok"`
- `prediction.roi.mask_source = "dinov3_roi_gray_otsu_refined"`
- `visualizations.mask_url`、`overlay_url`、`heatmap_url`、`heatmap_overlay_url` 非空
- `downloads.prediction_json_url` 和 `downloads.response_json_url` 可访问

下载/查看热力图：

```bash
curl -O http://127.0.0.1:8000/outputs/<job_id>/visualizations/oil_heatmap_overlay.png
```

## 4. Python TestClient smoke

无需常驻服务的本地验证命令：

```bash
/home/yr/anaconda3/envs/hj/bin/python - <<'PY'
import json
from pathlib import Path
from fastapi.testclient import TestClient
from oil_content_detection.service.app import create_app
from oil_content_detection.service.settings import HuajiaoWebSettings

root = Path('/home/yr/yr/code/cv/oil_content_detection')
settings = HuajiaoWebSettings(
    model_dir=root / 'artifacts/huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.2',
    upload_root=root / 'runtime/huajiao_web_smoke/uploads',
    output_root=root / 'runtime/huajiao_web_smoke/outputs',
    heatmap_max_pixels=50000,
)
client = TestClient(create_app(settings=settings))
files = {
    'image': ('0286_11.png', Path('/home/yr/yr/data/科研数据/枸椒5/0286_11/0286_11.png').read_bytes(), 'image/png'),
    'hdr': ('REFLECTANCE_0286_11.hdr', Path('/home/yr/yr/data/科研数据/枸椒5/0286_11/capture/REFLECTANCE_0286_11.hdr').read_bytes(), 'text/plain'),
    'dat': ('REFLECTANCE_0286_11.dat', Path('/home/yr/yr/data/科研数据/枸椒5/0286_11/capture/REFLECTANCE_0286_11.dat').read_bytes(), 'application/octet-stream'),
}
response = client.post('/api/predict', data={'sample_id': '0286_11', 'generate_heatmap': 'true'}, files=files)
print(response.status_code)
payload = response.json()
json.dumps(payload, ensure_ascii=False, allow_nan=False)
print(json.dumps({
    'job_id': payload.get('job_id'),
    'prediction': payload.get('prediction', {}).get('prediction'),
    'qc': payload.get('prediction', {}).get('qc'),
    'roi': payload.get('prediction', {}).get('roi', {}),
    'visualizations': payload.get('visualizations'),
}, ensure_ascii=False, indent=2))
PY
```

## 5. 常见问题排查

### 5.1 服务启动后 `/api/health` 为 503

常见原因：

1. `--model-dir` 路径错误。
2. DINOv3 ROI artifact 缺失。
3. PyTorch / torchvision / CUDA 环境不匹配。
4. checkpoint 或 backbone SHA256 校验失败。

处理：先运行 `docs/runbooks/huajiao-inference.md` 中的 artifact 前置检查和 CLI smoke。

### 5.2 上传后返回 `INVALID_UPLOAD`

原因：文件扩展名不在白名单、文件为空、或超过 `--max-upload-mb`。

处理：确认浏览器/调用端传的是 `.png/.jpg/.jpeg/.bmp`、`.hdr`、`.dat/.raw` 三个文件。不要把整个目录或压缩包直接作为 `dat` 字段上传。

### 5.3 上传后返回 `CUBE_IMAGE_SHAPE_MISMATCH`

原因：RGB 图像高宽和 ENVI cube `(lines, samples)` 的任一维度差值超过 5 像素，或自动对齐后 ROI mask 与 cube 仍不一致。

处理：确认上传的 RGB 与 `.hdr/.dat` 是同一次采集/同视角；1~5 像素的小差异会由服务自动 resize RGB 到 cube 高宽，响应中的 `prediction.roi.info.shape_alignment` 会记录原始高宽、cube 高宽、差值和 `aligned_image_path`；超过 5 像素需要重新导出或裁剪源数据，服务不会 resize cube 或做复杂配准。

### 5.4 热力图未生成但预测成功

热力图是非关键展示产物；生成失败时预测仍可返回，`prediction.qc.warnings` 会包含 `HEATMAP_GENERATION_FAILED`。优先检查：

1. ROI mask 文件是否存在。
2. `heatmap_max_pixels` 是否过低或内存不足。
3. cube band 数和 feature schema 是否匹配。

## 6. 质量门

开发变更后运行：

```bash
/home/yr/anaconda3/envs/hj/bin/python -m pytest -q tests/inference/test_heatmap.py tests/service/test_huajiao_web_app.py
/home/yr/anaconda3/envs/hj/bin/python scripts/huajiao_web.py --help
```

交付前再用至少一个验证集真实样本执行 Web 接口 smoke，并确认输出目录中的 `response.json`、`prediction.json`、ROI mask/overlay、heatmap/heatmap_overlay 都存在。
