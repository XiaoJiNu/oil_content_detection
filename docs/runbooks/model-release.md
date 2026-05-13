# 花椒含油率模型发布 Runbook

本文档说明如何重新生成、校验和发布 DINOv3 coarse polygon + gray_otsu refined ROI + 含油率 deployable artifact。

当前推荐候选发布目标：`artifacts/huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.1`

## 1. 发布原则

1. 应用交付模型只能使用上传文件可直接得到的特征：`wl_*` ROI 光谱均值、`coverage_ratio`、`log_valid_pixel_count`、`valid_pixel_ratio`。
2. 禁止使用 `weight_g`、`batch`、`sid_num`、`distill_ml`、`oil_ml_per_gram`、`oil_ml_per_100g` 等部署不可用或目标/标签字段作为输入特征。
3. 若 ROI 后端改为 DINOv3 或 DINOv3 post-refinement 配置改变，必须用同一 serving ROI 路径重新提取 train/val cube 特征后再训练回归模型。
4. DINOv3 ROI artifact 和含油率 artifact 分开版本化；含油率 `roi_config.json` 中记录 DINOv3 artifact 的相对路径。
5. DINOv3 checkpoint 和 backbone weights 必须通过 SHA256 校验。

## 2. 前置 artifact

DINOv3 ROI artifact：

```text
artifacts/huajiao-dinov3-roi-seg-v0.1.0/
  best_model.pt
  config.json
  manifest.json
  metrics.json
```

关键指标：

- Val samples: 40
- Val mean Dice: 0.975292
- Val mean IoU: 0.951880
- Backbone: `dinov3_vits16`
- image_size: 448
- patch_size: 16
- freeze_backbone: true

## 3. 生成 DINOv3 ROI 含油率 artifact

```bash
cd /home/yr/yr/code/cv/oil_content_detection
/home/yr/anaconda3/envs/hj/bin/python scripts/package_huajiao_model.py \
  --data-dir data/processed/huajiao_refined \
  --out-dir artifacts/huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.1 \
  --model-version huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.1 \
  --feature-source auto-roi-cube \
  --roi-backend dinov3 \
  --dinov3-roi-artifact artifacts/huajiao-dinov3-roi-seg-v0.1.0 \
  --roi-device cpu \
  --dinov3-post-refine-mode gray_otsu \
  --dinov3-refine-open-size 5 \
  --dinov3-refine-close-size 5 \
  --dinov3-refine-hole-fill-max-area 300 \
  --n-components 12 \
  --regressor ridge \
  --ridge-alpha 0.01
```

说明：

- 首次运行会读取 157 个 train cube 和 40 个 val cube，使用 DINOv3 coarse polygon + gray_otsu refined ROI 提取特征。
- 缓存写入：
  - `training_features/train_dinov3_roi_features.csv`
  - `training_features/val_dinov3_roi_features.csv`
- 后续同一 out-dir 重跑会复用缓存。
- 如部署机无 GPU，可用 `--roi-device cpu` 验证流程，但正式 GPU 发布后仍需在目标设备上重跑 smoke。

## 4. 当前发布输出

当前 `v0.3.1` 包含：

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

当前 metrics：

| Split | n | R² | RMSE | MAE |
| --- | ---: | ---: | ---: | ---: |
| Train | 157 | 0.639776 | 0.265788 | 0.205741 |
| Val | 40 | 0.777996 | 0.212330 | 0.170691 |

当前 `model.joblib` SHA256：

```text
51e05592b632f362226b3683d7b1a58d977822abc60f6f042780058079651206
```

## 5. 校验 artifact

### 5.1 读取 manifest/metrics

```bash
/home/yr/anaconda3/envs/hj/bin/python - <<'PY'
import json
from pathlib import Path
root = Path('artifacts/huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.1')
for name in ['manifest.json', 'metrics.json', 'roi_config.json']:
    print('\n---', name, '---')
    print(json.dumps(json.loads((root/name).read_text(encoding='utf-8')), ensure_ascii=False, indent=2))
PY
```

### 5.2 禁用字段检查

```bash
/home/yr/anaconda3/envs/hj/bin/python - <<'PY'
import json
from pathlib import Path
schema = json.loads(Path('artifacts/huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.1/feature_schema.json').read_text(encoding='utf-8'))
forbidden = {'distill_ml', 'oil_ml_per_gram', 'oil_ml_per_100g', 'weight_g', 'batch', 'sid_num'}
selected = set(schema['feature_columns'])
bad = sorted(forbidden & selected)
print('forbidden_selected=', bad)
assert not bad
print('feature_count=', len(schema['feature_columns']))
print('wavelength_count=', len(schema['wavelength_columns']))
print('quality_columns=', schema['quality_columns'])
PY
```

### 5.3 真实样本 smoke

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

验收：

```bash
/home/yr/anaconda3/envs/hj/bin/python - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path('/home/yr/yr/tmp/huajiao_dinov3_gray_otsu_refined_smoke_0346_11_v4/prediction.json').read_text(encoding='utf-8'))
assert payload['qc']['status'] == 'ok'
assert payload['roi']['mask_source'] == 'dinov3_roi_gray_otsu_refined'
assert Path(payload['roi']['mask_path']).exists()
assert Path(payload['roi']['overlay_path']).exists()
import json as _json
_json.dumps(payload, allow_nan=False)
print(payload['prediction'], payload['unit'], payload['roi']['coverage_ratio'])
PY
```

## 6. 测试与质量门

小范围推理测试：

```bash
/home/yr/anaconda3/envs/hj/bin/python -m pytest -q tests/inference -vv
```

完整质量门：

```bash
/home/yr/anaconda3/envs/hj/bin/python -m pytest -q
/home/yr/anaconda3/envs/hj/bin/python -m ruff check src tests scripts
/home/yr/anaconda3/envs/hj/bin/python -m black --check src tests scripts
```

若修改了模型打包脚本，至少单跑：

```bash
/home/yr/anaconda3/envs/hj/bin/python -m pytest -q tests/inference/test_huajiao_model_packaging.py -vv
```

若修改了 Predictor ROI backend，至少单跑：

```bash
/home/yr/anaconda3/envs/hj/bin/python -m pytest -q tests/inference/test_huajiao_serving_mode_b.py::test_predictor_applies_dinov3_post_refinement_inside_polygon -vv
```

## 7. 发布/回滚

发布：

1. 确认 `metrics.json`、`manifest.json`、`model_card.md` 与实际 artifact 一致。
2. 确认真实样本 smoke 成功。
3. 确认文档中的 artifact 路径和指标更新。
4. 确认应用团队使用 `huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.1`，而不是 coarse polygon v0.3.0 或精确复现 artifact。

回滚：

1. 若 gray_otsu refinement 现场 overlay 不满足演示，可临时回滚到 coarse DINOv3 artifact `artifacts/huajiao-oil-content-dinov3-roi-v0.3.0` 做定位对照；若 DINOv3 路径整体不满足，再回滚到传统 ROI deployable artifact：`artifacts/huajiao-oil-content-auto-roi-v0.2.1`。
2. 回滚时仍需执行 `docs/runbooks/huajiao-inference.md` 中的 smoke。
3. 不允许回滚到含 `weight_g/batch/sid_num` 的精确复现 artifact 作为应用 serving。

## 8. 已知风险

1. DINOv3 coarse polygon 分割指标高，但 coarse mask coverage≈0.44 会污染下游光谱；发布说明必须保留 v0.3.1 使用 gray_otsu refinement 的原因。
2. 当前 validation-top 组合 Val R²≈0.778，但 OOF R²≈0.409，说明小样本下存在固定验证集调参敏感风险。
3. 当前数据量较小，验证集不是严格跨批次/跨产地泛化评估。
4. 上游拍摄背景、光照、摆放和设备波段变化会影响 coarse polygon 与 refined mask；新现场必须先验收 overlay 和 smoke。
