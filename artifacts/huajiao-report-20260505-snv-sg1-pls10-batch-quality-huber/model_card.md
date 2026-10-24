# 花椒含油率模型复现卡片：2026-05-05 报告模型

## 复现对象

- 来源报告：`docs/reference_docs/实验记录/花椒含油率模型实验进展报告_20260505.md`
- 模型路线：SNV + SG一阶导(window=13, polyorder=2) + PLSR(10) + batch/quality 特征 + HuberRegressor
- 目标列：`oil_ml_per_100g`
- 单位：`ml/100g`

## 输入特征

- 光谱波段：224 个 `wl_*` 均值光谱列
- PLS 表征：10 维
- 元数据/质量特征：`batch` one-hot、`coverage_ratio`、`log_valid_pixel_count`、`valid_pixel_ratio`、`weight_g`、`sid_num`
- `batch` 类别口径：为精确复现历史报告预测文件，one-hot 类别空间按正式 train+val 划分中的原始目录级 batch 名称固定；这不同于报告统计章节中按产地/品种归并展示的大类口径，也不是严格 train-only 部署训练口径。
- 最终特征数：48

## 指标

| Split | N | R² | RMSE | MAE |
| --- | ---: | ---: | ---: | ---: |
| Train | 157 | 0.752985 | 0.220095 | 0.148464 |
| Val | 40 | 0.850229 | 0.174400 | 0.144887 |

## 使用限制

该模型是对报告中正式验证划分下工程最优结果的精确复现。它使用 `weight_g`、`batch` 和 `sid_num`，因此适合复现实验和内部对比；如果应用展示端不能提供重量，不能直接把该模型作为无重量部署模型使用，应使用单独的禁重量部署模型。
