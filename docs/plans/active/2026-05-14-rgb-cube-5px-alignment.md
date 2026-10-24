# RGB/cube 5 像素容差自动对齐实施计划

> **For Hermes:** 按 harness 方法执行：先更新控制面，再用 TDD 修改推理与 Web 服务，最后同步部署移交包。

**Goal:** 将花椒含油率部署推理从“RGB 高宽必须与 ENVI cube `(lines, samples)` 完全一致”改为“仍要求同视角；若 RGB 高/宽与 cube 高/宽分别相差不超过 5 像素，服务自动将 RGB resize 到 cube 尺寸后再进行 ROI、光谱聚合和含油率预测；超过 5 像素仍返回结构化错误”。

**项目类型:** 组合型 harness 批次：software-product（推理 API/Web 行为变更）+ deployment-platform（移交包部署契约更新）+ data-pipeline（RGB/cube 输入契约与质量门）。

**Architecture:** 不改 cube，不改模型权重，不重新训练模型。推理入口读取 cube 后，以 cube `(lines, samples)` 为基准：如果 RGB 尺寸一致则原路径运行；如果每个维度差值 `<= 5`，生成一张 `*_aligned_to_cube.png` 的服务端临时 RGB 图，所有 ROI 分割、mask/overlay、热力图都使用该对齐图；如果任一维度差值 `> 5`，保留 `CUBE_IMAGE_SHAPE_MISMATCH` 错误码并在 details 里返回尺寸、差值和阈值。输出 JSON 的 `roi.info.shape_alignment` 和 `qc.warnings` 记录是否发生自动 resize。

**Tech Stack:** Python 3.10+，OpenCV resize，FastAPI，pytest；运行环境使用 `/home/yr/anaconda3/envs/hj/bin/python`。

---

## 方案要点

1. 输入契约
   - 仍然要求 RGB 与 cube 是同视角采集；本次只处理小幅高宽差，不做复杂特征配准、旋转、平移、裁剪或 cube resample。
   - 默认容差：`max_delta_px=5`，高和宽分别判断，即 `abs(image_h - cube_h) <= 5` 且 `abs(image_w - cube_w) <= 5`。
   - 超过容差仍返回 `CUBE_IMAGE_SHAPE_MISMATCH`。

2. 算法行为
   - 以 cube 尺寸为源头标准，因为光谱聚合必须按 cube 像素索引取值。
   - resize RGB 到 `(cube_samples, cube_lines)`，不 resize cube。
   - DINOv3 ROI：传入 resize 后的临时 RGB 文件，确保输出 mask 与 cube 一致。
   - Traditional ROI：直接使用 resize 后的 BGR ndarray。
   - 光谱聚合继续严格校验 `mask.shape[:2] == cube.shape[:2]`，防止下游 silent mismatch。
   - 热力图读取 `roi.info.shape_alignment.aligned_image_path`，否则用原始 image_path。

3. 输出和审计
   - `roi.info.shape_alignment` 记录：`enabled`、`method`、`resized`、`image_shape_hw`、`cube_shape_hw`、`delta_hw`、`max_delta_px`、`aligned_image_path`。
   - `qc.warnings` 添加 `RGB image was resized to ENVI cube shape ...`。
   - README/spec/runbook/handoff README 同步说明该策略。

## TDD 任务

### Task 1: 新增 5px 容差 resize 行为测试

**Files:**
- Modify: `tests/inference/test_huajiao_serving_mode_b.py`

**Steps:**
1. 添加 traditional predictor 测试：cube 为 `(180, 140)`，RGB 为 `(183, 145)`，预测应成功；ROI mask/overlay 尺寸应为 `(180, 140)`；`qc.warnings` 与 `roi.info.shape_alignment` 应记录 resize。
2. 添加超过容差测试：cube 为 `(180, 140)`，RGB 为 `(186, 140)`，预测应抛 `CUBE_IMAGE_SHAPE_MISMATCH`，details 包含 `max_allowed_delta_px=5`。
3. 添加 DINOv3 fake predictor 测试：输入 RGB 与 cube 差值 <=5 时，fake DINOv3 读取到的 `image_path` 应已经是 cube 尺寸。
4. 运行：`/home/yr/anaconda3/envs/hj/bin/python -m pytest -q tests/inference/test_huajiao_serving_mode_b.py::<new_tests>`，预期先失败。

### Task 2: 实现 RGB/cube shape alignment helper

**Files:**
- Modify: `src/oil_content_detection/inference/features.py`
- Modify: `src/oil_content_detection/inference/predictor.py`

**Steps:**
1. 保留 `validate_same_view_shape()` 作为 mask/cube 严格校验。
2. 新增 `RGB_SHAPE_ALIGNMENT_DEFAULT_MAX_DELTA_PX = 5` 和 `validate_rgb_cube_shape_for_alignment()` / `resize_rgb_to_cube_shape()`。
3. Predictor 在 load image/cube 后调用 alignment helper；如需 resize，在 `debug_dir` 下写 `*_aligned_to_cube.png` 并后续使用该路径和 BGR ndarray。
4. 将 alignment info 合并到 ROI info；将 warning 合并进 `QCResult.warnings`。
5. 运行 Task 1 测试，预期通过。

### Task 3: 修复 Web 热力图使用原图的问题

**Files:**
- Modify: `src/oil_content_detection/service/app.py`
- Modify: `tests/service/test_huajiao_web_app.py`

**Steps:**
1. 新增测试：fake predictor 返回 `roi.info.shape_alignment.aligned_image_path`，Web 生成热力图时应读取该 aligned image，而不是原始上传图。
2. 在 `app.py` 中选择 heatmap image path：优先 `prediction_payload["roi"]["info"]["shape_alignment"]["aligned_image_path"]`，否则原 `image_path`。
3. 运行 service 测试。

### Task 4: 同步 artifact config 与文档

**Files:**
- Modify: `artifacts/huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.1/roi_config.json`
- Modify: `artifacts/huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.1/model_card.md`
- Modify: `README.md`, `ARCHITECTURE.md`, `docs/specs/*.md`, `docs/runbooks/*.md`

**Steps:**
1. 在 `roi_config.json` 增加 `shape_alignment`：enabled=true, max_delta_px=5, method=resize_rgb_to_cube_hw。
2. 文档明确：<=5px 自动 resize RGB；>5px 报 `CUBE_IMAGE_SHAPE_MISMATCH`；不做复杂配准、不 resize cube。
3. 运行 `git diff --check`。

### Task 5: 同步部署移交包

**Files:**
- Modify under `/home/yr/yr/tmp/huajiao_web_handoff_20260513_180107/`

**Steps:**
1. 将修改后的 runtime 代码、测试、docs、artifact config/model_card 复制到 handoff 包对应路径。
2. 更新 `/home/yr/yr/tmp/huajiao_web_handoff_20260513_180107/README_HANDOFF.md`，写清输入新契约、部署命令、可直接复制文件清单。
3. 重新生成 `HANDOFF_MANIFEST.json`，运行 `python scripts/check_handoff.py`。
4. 在 handoff 包中运行 scoped pytest 和 `scripts/huajiao_web.py --help`。

## Done Definition

- 定向 inference/service 测试通过。
- source repo docs 与 handoff README 不再声称“不会自动 resize”。
- 部署包代码和 docs 已同步，manifest 自检通过。
- 最终答复列出可以直接复制给部署同事的目录/文件、部署命令、验证命令和超过 5px 时的错误行为。

## 执行结果（2026-05-14）

状态：已完成实现、文档和部署移交包同步。

关键变更：

1. `HuajiaoOilPredictor.predict()` 新增 RGB/cube shape alignment：
   - RGB 与 cube 高宽一致：直接推理。
   - 每维差值 `<=5px`：自动 resize RGB 到 cube `(lines, samples)`，并在 debug 目录写 `*_aligned_to_cube.png`。
   - 任一维差值 `>5px`：返回/抛出 `CUBE_IMAGE_SHAPE_MISMATCH`，details 包含原始尺寸、cube 尺寸、delta 和阈值。
2. DINOv3 ROI、traditional ROI、热力图 overlay 均使用实际对齐后的 RGB 图；cube 不 resize。
3. 输出 JSON 新增/保留 `roi.info.shape_alignment`，自动 resize 时 `qc.warnings` 记录原因。
4. `roi_config.json` 增加：`shape_alignment.enabled=true`、`max_delta_px=5`、`method=resize_rgb_to_cube_hw`。
5. 已同步 README、ARCHITECTURE、serving/web specs、runbooks、artifact model_card，以及 `/home/yr/yr/tmp/huajiao_web_handoff_20260513_180107/README_HANDOFF.md` 和 `HANDOFF_MANIFEST.json`。

验证结果：

```bash
/home/yr/anaconda3/envs/hj/bin/python -m pytest -q \
  tests/inference/test_huajiao_serving_mode_b.py \
  tests/service/test_huajiao_web_app.py
# 17 passed

/home/yr/anaconda3/envs/hj/bin/python scripts/huajiao_web.py --help
/home/yr/anaconda3/envs/hj/bin/python scripts/huajiao_predict.py --help
# ok

git diff --check
# ok

cd /home/yr/yr/tmp/huajiao_web_handoff_20260513_180107
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/yr/anaconda3/envs/hj/bin/python -m pytest -q -p no:cacheprovider \
  tests/inference/test_huajiao_serving_mode_b.py \
  tests/service/test_huajiao_web_app.py
# 17 passed

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/yr/anaconda3/envs/hj/bin/python scripts/check_handoff.py
# handoff package ok; file_count=241; total_size_bytes=97214958
```
