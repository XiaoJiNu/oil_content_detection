# 花椒含油率 DINOv3 refined ROI 部署模型优化记录（2026-05-14）

## 目标

在应用侧只能上传 RGB 图像 + 同视角 ENVI hyperspectral cube 的部署约束下，继续优化当前 DINOv3 coarse ROI + gray_otsu in-polygon refinement 口径的含油率回归模型，同时保留 2026-05-14 新增的 RGB/cube 5px 尺寸容差策略：

- RGB 高宽与 cube `(lines, samples)` 完全一致：直接预测。
- 每个维度差值 `<=5px`：只将 RGB resize 到 cube 高宽，cube 不重采样，然后再做 ROI、光谱聚合、热力图和含油率预测。
- 任一维度差值 `>5px`：返回 `CUBE_IMAGE_SHAPE_MISMATCH`。

## 生成结果

新部署候选 artifact：

```text
artifacts/huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.2
```

模型类：

```text
oil_content_detection.inference.advanced_training.AdvancedHuajiaoRegressor
```

该模型的 serving 输入 schema 仍然只包含上传文件可推导字段：

- `wl_*` ROI 平均光谱。
- `coverage_ratio`。
- `log_valid_pixel_count`。
- `valid_pixel_ratio`。

未使用也不允许使用 `weight_g`、`batch`、`sid_num`、蒸馏标签、目标值等应用上传不可得字段。

## 正式 train/val 指标

训练只使用 train split，val 为正式验证集：

| Artifact | Split | n | R² | RMSE | MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| v0.3.2 | Train | 157 | 0.645112 | 0.263812 | 0.203680 |
| v0.3.2 | Val | 40 | 0.800960 | 0.201049 | 0.162341 |
| v0.3.1 | Val | 40 | 0.777996 | 0.212330 | 0.170691 |
| 历史手工/report best | Val | 40 | 0.850229 | 0.174400 | 0.144887 |

结论：v0.3.2 在部署可用约束内相比 v0.3.1 有提升（Val R² 从 0.777996 到 0.800960，RMSE 从 0.212330 到 0.201049），但仍低于历史手工/report best；主要差距来自自动 ROI 与手工/历史特征口径不同，以及历史复现模型使用了应用上传不可得字段。

## 选择的候选配置

```json
{
  "candidate_id": "cand_0284",
  "preprocess": "snv_sg0",
  "spectral_feature_set": "summary",
  "n_components": 12,
  "regressor_type": "ridge",
  "ridge_alpha": 1.0
}
```

完整搜索摘要：

```text
reports/huajiao_dinov3_refined_oil_optimization_20260514/advanced_summary.json
reports/huajiao_dinov3_refined_oil_optimization_20260514/diagnostics.json
reports/huajiao_dinov3_refined_oil_optimization_20260514/advanced_stage1_val_grid.csv
reports/huajiao_dinov3_refined_oil_optimization_20260514/advanced_stage2_oof_grid.csv
```

## 部署注意

部署包如果使用 v0.3.2，必须同时移交以下新增运行时代码，否则 `model.joblib` 无法反序列化：

```text
src/oil_content_detection/inference/advanced_training.py
```

启动服务时推荐模型目录改为：

```bash
python scripts/huajiao_web.py \
  --model-dir artifacts/huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.2 \
  --host 0.0.0.0 \
  --port 8000
```
