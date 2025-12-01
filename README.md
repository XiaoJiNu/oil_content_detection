# 油茶籽含油率检测复现项目

本仓库用于复现论文《基于高光谱成像的油茶籽含油率检测方法》中的最佳方案（光谱集Ⅱ + 遗传算法 + PLSR），当前提供模拟数据与完整代码骨架，便于后续替换为真实高光谱/含油率测定数据。

## 项目目标

- 构建油茶籽高光谱数据的预处理、波段筛选与回归建模流程。
- 通过遗传算法筛选特征波长并训练 PLS 回归模型，预测含油率。
- 为未来接入真实设备与实验数据提供可扩展的工程框架与文档。

## 快速开始

1. **准备环境**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **生成模拟数据（如需重新随机化）**

   ```bash
   python scripts/generate_simulated_set_II.py
   ```

   脚本会在 `data/processed/set_II/` 下生成新的高光谱立方体与 ROI 均值光谱文件。
3. **运行复现流程（使用模拟数据）**

   ```bash
   # 基础运行
   python scripts/run_best_method.py

   # 显示 GA 训练过程
   python scripts/run_best_method.py --verbose

   # 保存结果到指定目录
   python scripts/run_best_method.py --output-dir results/experiment_01

   # 查看所有参数选项
   python scripts/run_best_method.py --help
   ```

   输出包括：遗传算法筛选出的波长列表、训练/测试集 R² 与 RMSE 指标。
4. **可视化结果**

   ```bash
   # 生成所有可视化图表（GA历史、光谱选择、预测结果）
   python scripts/visualize_results.py results/experiment_01

   # 可视化油茶籽含油分布图像（论文中的空间分布图）
   python scripts/visualize_oil_distribution.py results/experiment_01 --mode summary

   # 可视化特定样本的详细含油分布
   python scripts/visualize_oil_distribution.py results/experiment_01 --mode single --sample-indices 0 1 2

   # 指定输出目录
   python scripts/visualize_results.py results/experiment_01 --output-dir figures/
   ```
5. **运行测试**

   ```bash
   pytest -v
   ```
6. **更换为真实数据**

   - 将实测光谱整理为与 `data/processed/set_II/mean_spectra.csv` 相同的列格式（`sample_id`、`wl_<波长>`、`oil_content`）。
   - 使用 `--data` 参数指向新数据：
     ```bash
     python scripts/run_best_method.py --data path/to/your/data.csv
     ```

## 使用指南（数据准备、训练、测试）

1. **数据准备**

   - 仓库已附带模拟数据：`data/processed/set_II/mean_spectra.csv`（ROI 均值光谱）与 `data/processed/set_II/simulated_set_II_cube.npz`（光谱立方体+掩膜）。
   - 如需重新随机化模拟数据，运行：
     ```bash
     python scripts/generate_simulated_set_II.py --output data/processed --seed 2024
     ```
   - 替换为真实数据时，请准备一个 CSV，满足：
     - 列名以 `wl_` 开头表示波长列（如 `wl_900`、`wl_905`），油含量列名为 `oil_content`，可选的样本标识列 `sample_id`。
     - 反射率建议归一化到 0~1，油含量在 0~100 之间且无缺失值。
     - 示例格式：
       ```csv
       sample_id,wl_900,wl_905,wl_910,oil_content
       sample_000,0.41,0.42,0.40,33.5
       sample_001,0.36,0.37,0.39,28.2
       ```

     将文件放到任意目录（如 `data/processed/your_set/mean_spectra.csv`），后续用 `--data` 指向即可。
2. **训练并评估模型**

   - 直接使用默认模拟数据：
     ```bash
     python scripts/run_best_method.py --output-dir results/demo_run --verbose
     ```
   - 更换为自定义数据：
     ```bash
     python scripts/run_best_method.py --data data/processed/your_set/mean_spectra.csv --output-dir results/your_run --verbose
     ```
   - 关键可调参数：`--test-size`（测试集占比）、`--ga-generations` 与 `--ga-population`（遗传算法迭代与种群大小）、`--target-features`（期望波段数）、`--seed`（可复现随机种子）。运行后会打印训练/测试 R² 和 RMSE，并在指定 `--output-dir` 下生成：
     - `results.json`：指标摘要与配置
     - `selected_wavelengths.json`：被选中的波长列表
     - `plsr_model.pkl`：训练好的 PLSR 模型
     - `ga_history.json` 与 `feature_support.npy`：遗传算法历史与特征掩码
3. **测试与验证**

   - 模型训练脚本自带留出集评估（默认 66% 训练 / 34% 测试），运行日志即为模型测试结果。
   - 运行单元测试验证代码功能：
     ```bash
     pytest -q
     # 或仅跑核心流水线测试
     pytest -q tests/test_plsr_best.py
     ```

## 真实数据（花椒光谱）流程

- **准备原始文件**
  - 光谱文件：放置在 `/home/yr/yr/data/科研数据/<批次>/<样本编号>/capture/REFLECTANCE_<编号>.hdr/.dat`，文件名包含样本编号（如 `0164_11`），已完成黑白板校正。
  - 样本图片：与样本目录同级的 `<编号>.png`（如 `0164_11.png`），用于 ROI 可视化叠加。
  - 标签文件：`docs/2025年8月花椒挥发油测定结果.xls`，使用 sheet `云南竹叶椒` 与 `云南藤椒1`，列名支持组合见 `LabelConfig`（高光谱图件编号 / 蒸馏量（初）ml / 重量(g) 等）。依赖 `xlrd` 读取 `.xls`。
- **生成 train/val 样本清单（8:2，自动匹配原始目录）**
  ```bash
  python scripts/build_splits.py \
    --excel docs/2025年8月花椒挥发油测定结果.xls \
    --sheets 云南竹叶椒和云南藤椒1 \
    --raw-root /home/yr/yr/data/科研数据 \
    --out-dir data/labels \
    --train-ratio 0.8 \
    --seed 2024
  ```

  - 输出：`data/labels/train.txt`、`data/labels/val.txt`，每行字段：`高光谱图件编号  重量(g)  蒸馏量(初)ml  ml/100g  样本目录路径`。仅保留同时存在标签与 `REFLECTANCE_<编号>.hdr/.dat` 的样本。
- **按清单构建特征表并导出 ROI 可视化**
  ```bash
  # 生成训练集特征/元数据 + ROI 叠加图
  python scripts/build_huajiao_from_split.py \
    --split data/labels/train.txt \
    --output-dir data/processed/huajiao/train \
    --roi-dir data/processed/ROI \
    --primary-stat trimmed_mean --include-stats trimmed_mean \
    --nir-target 800 --red-target 650 \
    --ratio-quantile 0.90 --ratio-floor 1.05 --intensity-quantile 0.15 \
    --trim-fraction 0.10

  # 验证集同理
  python scripts/build_huajiao_from_split.py \
    --split data/labels/val.txt \
    --output-dir data/processed/huajiao/val \
    --roi-dir data/processed/ROI \
    --primary-stat trimmed_mean --include-stats trimmed_mean
  ```

  - 输出：`data/processed/huajiao/<train|val>/huajiao_spectra.parquet`（或 CSV）+ `huajiao_metadata.parquet`。ROI 叠加图保存到 `data/processed/ROI/<编号>_roi.png`。
  - ROI 逻辑：NIR/Red 比值 + 强度分位阈值 → 形态学处理 → 极亮/极暗像素剔除；若样本图片存在，自动将掩膜缩放叠加，否则使用光谱立方体均值图作底图。
- **训练与验证（PLSR + 可选 GA 选波长）**
  ```bash
  python - <<'PY'
  import re
  import pandas as pd
  import numpy as np
  from oil_content_detection.models.plsr_pipeline import PLSRExperimentConfig, run_plsr_experiment
  from oil_content_detection.preprocessing import PreprocessStep
  from oil_content_detection.feature_selection.ga_selector import GAConfig

  df = pd.read_parquet("data/processed/huajiao/train/huajiao_spectra.parquet")
  feature_cols = [c for c in df.columns if re.match(r"^wl_\\d+$", c)]  # 只用主统计量的波长列
  wavelengths = [int(c.split("_")[1]) for c in feature_cols]
  X = df[feature_cols].to_numpy(dtype=float)
  y = df["oil_ml_per_gram"].to_numpy(dtype=float)

  cfg = PLSRExperimentConfig(
      preprocess=(PreprocessStep("snv"), PreprocessStep("savgol", {"window_length": 9, "polyorder": 2, "deriv": 1})),  # SNV + 一阶导平滑
      use_ga=True,
      ga_config=GAConfig(generations=10, population_size=12, min_features=8, max_features=25, target_features=18, random_state=2024),
      test_size=0.25,
      random_state=2024,
      cv_splits=5,
      max_components=12,
  )

  result = run_plsr_experiment(X, y, wavelengths=wavelengths, config=cfg)
  print("Train/Test R2:", result.train_r2, result.test_r2)
  print("RMSEC/RMSEP:", result.rmsec, result.rmsep)
  print("RMSECV/R2cv:", result.rmsecv, result.r2cv)
  print("Selected wavelengths:", result.selected_wavelengths)
  PY
  ```

  - 评估指标：`train_r2/test_r2`、`rmsec/rmsep`（训练/测试 RMSE）、`rmsecv/r2cv`（交叉验证），若开启 GA 还会返回 `selected_wavelengths` 与 `support_mask`。
  - 数据前提：光谱矩阵应为反射率（已黑白校正）；波段列表传入后可直接记录选波长；样本较少时可调大 `test_size` 或改为 `cv_splits` 小一些。
- **验证与排查**
  - 简查数据：`python - <<'PY' ... print(df.isna().sum().sum())` 确认无缺失，`df["oil_ml_per_gram"].describe()` 检查标签范围。
  - 运行关键单测：`pytest -q tests/test_huajiao_dataset.py tests/test_plsr_pipeline_extended.py`。
  - 若 ROI 数量为 0 或覆盖率极低，可调 `HuajiaoROIConfig` 的 `ratio_floor/intensity_quantile/min_area`。

## 仓库结构

- `src/oil_content_detection/`：核心源码
  - `feature_selection/ga_selector.py`：遗传算法波段筛选
  - `models/plsr_best.py`：GA + PLSR 管线
- `scripts/run_best_method.py`：一键执行最佳方案
- `data/processed/set_II/mean_spectra.csv`：模拟光谱数据（ROI 均值光谱）
- `data/processed/set_II/simulated_set_II_cube.npz`：模拟高光谱立方体与 ROI 掩膜
- `docs/reference_docs/`：需求文档、复现计划与实验记录
- `AGENTS.md`：贡献者指南

## 文档与记录

- `docs/reference_docs/功能需求文档/复现计划.md`：整体复现计划与架构说明。
- `docs/reference_docs/实验记录/模拟复现记录.md`：模拟数据实验过程与指标。

## 下一步建议

- 接入真实高光谱采集系统，补充数据预处理与含油率标定脚本。
- 将模型训练、可视化等步骤拓展为 CLI/Notebook 形式，便于批量实验。
- 引入自动化测试覆盖关键模块（特征选择、PLSR 拟合）。
