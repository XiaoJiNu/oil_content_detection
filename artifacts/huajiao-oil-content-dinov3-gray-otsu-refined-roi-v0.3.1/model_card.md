# huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.1 Model Card

## 用途

花椒含油率展示/工程调用模型。输入为同视角 RGB 图像与 ENVI 高光谱 cube；RGB 高宽与 cube `(lines, samples)` 完全一致时直接推理，任一维度相差不超过 5 像素时服务自动将 RGB resize 到 cube 高宽后推理。系统用 DINOv3 ROI 分割模型先定位 coarse polygon，再在 polygon 内执行 gray_otsu 二次精分割得到 refined 花椒区域，聚合 ROI 内平均光谱，并输出 `oil_ml_per_100g` 预测值。

## 来源实验

- 来源报告：`docs/reference_docs/实验记录/花椒含油率模型实验进展报告_20260505.md`
- 报告推荐路线：`SNV + SG一阶导(window=13, polyorder=2) + PLSR(10) + HuberRegressor`
- 本 artifact 的部署约束：不使用 `weight_g`、`batch`、`sid_num`；只使用上传文件可直接得到的波段均值和 ROI 质量特征。
- 本 artifact 的特征来源：`dinov3-roi-cube`；ROI 后端：`dinov3`；post_refinement：`gray_otsu`。默认发布版使用 DINOv3 coarse polygon + gray_otsu refined ROI 从 cube 重新提取训练/验证特征，使训练口径与展示推理口径一致。

## 数据与特征

- 训练/验证数据：`/home/yr/yr/code/cv/oil_content_detection/data/processed/huajiao_refined`
- 波段特征数：224
- ROI 质量特征：coverage_ratio, log_valid_pixel_count, valid_pixel_ratio
- 禁止字段：distill_ml, oil_ml_per_gram, oil_ml_per_100g, weight_g, batch, sid_num

## 指标

| Split | R² | RMSE | MAE |
| --- | ---: | ---: | ---: |
| Train | 0.639776 | 0.265788 | 0.205741 |
| Val | 0.777996 | 0.212330 | 0.170691 |

注意：该指标是当前正式 Train/Val 划分下的工程指标。本 artifact 为了满足部署约束，去掉了报告最优模型中的重量、批次和样本编号相关特征，因此指标可能低于报告中的 `Val R²=0.8502` 最优记录。

## 输入契约

1. `image_path`: RGB/BGR 图像文件，OpenCV 可读取。
2. `hdr_path`: ENVI `.hdr` 文件，旁边必须有同名 `.dat`。
3. RGB 图像与 cube 必须同视角；RGB 高宽与 cube `(lines, samples)` 完全一致时直接推理，任一维度相差不超过 5 像素时自动将 RGB resize 到 cube 高宽后推理；超过 5 像素返回 `CUBE_IMAGE_SHAPE_MISMATCH`。不 resize cube，不做旋转、平移、裁剪或复杂配准。

## 输出

- `prediction`: 含油率预测值。
- `unit`: `ml/100g`。
- `roi`: ROI 像素数、覆盖率、有效光谱像素数、可选 mask/overlay 路径。
- `qc`: 状态和 warning。

## 风险与限制

- ROI 分割依赖当前拍摄背景/红纸/标签风格；DINOv3 coarse polygon 与 gray_otsu refinement 任一环节受影响都可能改变 coverage，换背景或光照后必须先做 ROI overlay 验收。
- 当前数据量较小，不代表严格跨产地、跨批次泛化能力。
- 不应把该模型用于超出当前花椒样本、设备格式和拍摄流程的生产决策，除非补充验证。
