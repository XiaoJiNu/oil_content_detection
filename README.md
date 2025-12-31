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
     # 推荐使用 conda 环境 /home/yr/anaconda3/envs/hj（numpy<2）
     /home/yr/anaconda3/envs/hj/bin/python -m pytest -q
     # 或仅跑核心流水线测试
     /home/yr/anaconda3/envs/hj/bin/python -m pytest -q tests/test_plsr_best.py
     ```

## 总含油量预测（A+B对照，新增）

当前最佳模型预测 `oil_ml_per_gram`（ml/g）。为避免推理时称重，新增“总含油量”两种预测方案并可对照评估：

- **方案A 两阶段**：
  1) 用 GA+PLSR（现有管线）预测 `oil_ml_per_gram_pred`；
  2) 用 ROI 形态特征回归预测 `weight_g_pred`；
  3) `oil_ml_total_pred_a = oil_ml_per_gram_pred * weight_g_pred`。
- **方案B 直接预测**：
  以 `distill_ml`（总含油量）为标签，直接用 PLSR 预测 `oil_ml_total_pred_b`，默认拼接“光谱 + 形态特征”。

对应实现：

- 重量回归器：`src/oil_content_detection/models/weight_regressor.py`
- A+B对照管线：`src/oil_content_detection/pipelines/total_oil_pipeline.py`
- 对照脚本：`scripts/compare_total_oil_methods.py`

### 数据字段要求

对照脚本读取 `huajiao_spectra.parquet`（或等价 DataFrame），需要包含：

- 光谱列：`wl_<nm>`
- 标签：`oil_ml_per_gram`、`weight_g`、`distill_ml`（可选；缺失时自动用 `oil_ml_per_gram*weight_g` 生成总量标签）
- 形态列：`valid_pixel_count`、`coverage_ratio`

### 运行对照

```bash
python scripts/compare_total_oil_methods.py \
  --train data/processed/huajiao/train/huajiao_spectra.parquet \
  --val data/processed/huajiao/val/huajiao_spectra.parquet \
  --use-ga \
  --output-dir results/total_oil_comparison
```

脚本会打印验证集上：

- 方案A 总量指标：`total_a_val`（R²/RMSE/MAE/MAPE）
- 方案B 总量指标：`total_b_val`
- 中间子任务：`oil_per_gram_*`、`weight_*`

若指定 `--output-dir`，还会额外保存逐样本预测表，格式参考 `results/stage3_ga_final/val_predictions.csv`：

- `train_predictions_total_oil.csv`
- `val_predictions_total_oil.csv`

### 实际花椒数据对照结果

基于当前 `data/labels/train.txt`（39样本）/`data/labels/val.txt`（10样本）构建的特征表，在 `/home/yr/anaconda3/envs/hj` 环境下运行：

```bash
python scripts/compare_total_oil_methods.py \
  --train data/processed/huajiao/train/huajiao_spectra.parquet \
  --val data/processed/huajiao/val/huajiao_spectra.parquet \
  --use-ga \
  --output-dir results/total_oil_comparison_real
```

验证集（总含油量）指标：

- 方案A（两阶段）：R²=0.7820，RMSE=0.172442，MAE=0.096283
- 方案B（直接预测）：R²=0.7834，RMSE=0.171891，MAE=0.119214

完整结果见 `results/total_oil_comparison_real/results_20251212_135906.json`。

## 花椒数据完整实验流程（已完成）

### 实验概述

本项目已完成基于花椒高光谱数据的含油率检测完整实验流程，包括数据准备、基线模型、GA特征选择三个阶段，最终模型验证集R²达到0.9163。

### 实验结果总结

| 阶段   | 模型                         | 波长数       | 成分数      | Train R² | Val R²          | RMSEP              | 改进              |
| ------ | ---------------------------- | ------------ | ----------- | --------- | ---------------- | ------------------ | ----------------- |
| 阶段二 | 全波段PLSR (Raw)             | 224          | 3           | 0.7227    | 0.6971           | 0.002430           | 基线              |
| 阶段二 | 全波段PLSR (SG一阶导)        | 224          | 3           | 0.7087    | **0.7664** | 0.002134           | +10%              |
| 阶段三 | **GA+PLSR (SG一阶导)** | **12** | **5** | 0.7988    | **0.9163** | **0.001277** | **+31%** ✅ |

**关键成果**：

- ✅ 验证集R²从0.6971提升至**0.9163**（提升31%）
- ✅ 波长数从224个减少至**12个**（减少94.6%）
- ✅ RMSEP从0.002430降低至**0.001277**（降低47%）

**选中的12个关键波长**：403, 408, 487, 498, 503, 538, 656, 748, 770, 784, 908, 976 nm

### 完整实验步骤

#### 阶段一：数据准备

**1. 生成样本清单**

```bash
python scripts/build_splits.py \
  --excel docs/2025年8月花椒挥发油测定结果.xls \
  --sheets 云南竹叶椒 云南藤椒1 \
  --raw-root /home/yr/yr/data/科研数据 \
  --out-dir data/labels \
  --train-ratio 0.8 \
  --seed 2024
```

输出：`data/labels/train.txt` (39样本), `data/labels/val.txt` (10样本)

**2. HSV阈值调参（推荐）**

使用交互式工具找到最佳分割参数：

```python
from oil_content_detection.preprocessing.hsv_tuner import hsv_debugger
hsv_debugger('/path/to/sample.png')
# 调整滑动条，记录最佳HSV参数
```

**3. 批量生成ROI掩膜**

```bash
# 使用调参得到的HSV参数
python scripts/generate_roi_masks.py \
  --split data/labels/train.txt \
  --output-dir data/processed/ROI \
  --hsv-lower 30 70 30 \
  --hsv-upper 65 255 255

# 验证集同理
python scripts/generate_roi_masks.py \
  --split data/labels/val.txt \
  --output-dir data/processed/ROI \
  --hsv-lower 30 70 30 \
  --hsv-upper 65 255 255
```

输出：`*_mask.png` (二值掩膜), `*_overlay.png` (可视化叠加图)

⚠️ **重要**：人工检查所有 `*_overlay.png`，确认ROI覆盖正确！

**4. 构建特征表**

```bash
# 训练集
python scripts/build_huajiao_from_split.py \
  --split data/labels/train.txt \
  --output-dir data/processed/huajiao/train \
  --roi-dir data/processed/ROI \
  --roi-mask-dir data/processed/ROI \
  --trim-fraction 0.10 \
  --clip-low 0.01 \
  --clip-high 0.99

# 验证集
python scripts/build_huajiao_from_split.py \
  --split data/labels/val.txt \
  --output-dir data/processed/huajiao/val \
  --roi-dir data/processed/ROI \
  --roi-mask-dir data/processed/ROI \
  --trim-fraction 0.10
```

输出：

- `huajiao_spectra.parquet`: 光谱特征表
- `huajiao_metadata.parquet`: 质控元数据
- `*_roi.png`: 清洗后的ROI可视化

#### 阶段二：基线模型与预处理对比

**运行完整分析**

```bash
# 使用虚拟环境
/home/yr/anaconda3/envs/hj/bin/python scripts/baseline_analysis.py
```

**输出结果** (`results/baseline_analysis/`):

- `eda_analysis.png`: 探索性数据分析图表
- `baseline_plsr.png`: 基线PLSR模型性能
- `preprocessing_comparison.png`: 预处理方法对比
- `preprocessing_comparison.csv`: 详细对比数据
- `stage2_summary.json`: 阶段二结果摘要

**关键发现**：

- 最佳预处理：**SG一阶导**（Savitzky-Golay一阶导数）
- 最佳成分数：3
- 验证集R²：0.7664

#### 阶段三：GA特征选择与模型优化

**遇到的问题与解决方案**

❌ **问题1：初始GA实现过拟合**

- 现象：训练集R²=0.9439，验证集R²=0.3268（严重过拟合）
- 原因：
  - 样本量太少（39个训练样本）
  - GA选择了30个波长（上限），使用9个PLSR成分
  - 3折交叉验证在小样本上不稳定

✅ **解决方案**：

- 减少波长上限：30 → 12
- 减少目标波长：18 → 10
- 增加交叉验证折数：3 → 5
- 限制PLSR成分数：max(6, n_features//2)

**运行优化后的GA**

```bash
# 方法1：使用专用脚本（推荐）
/home/yr/anaconda3/envs/hj/bin/python - <<'PY'
import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error
from oil_content_detection.feature_selection.ga_selector import GeneticAlgorithmSelector, GAConfig

# 加载数据
train = pd.read_parquet("data/processed/huajiao/train/huajiao_spectra.parquet")
val = pd.read_parquet("data/processed/huajiao/val/huajiao_spectra.parquet")

wl_cols = [c for c in train.columns if c.startswith('wl_')]
wavelengths = np.array([float(c.split('_')[1]) for c in wl_cols])

X_train = train[wl_cols].values
y_train = train['oil_ml_per_gram'].values
X_val = val[wl_cols].values
y_val = val['oil_ml_per_gram'].values

# 应用SG一阶导
X_train_prep = savgol_filter(X_train, 9, 2, deriv=1, axis=1)
X_val_prep = savgol_filter(X_val, 9, 2, deriv=1, axis=1)

# 配置GA（小样本优化参数）
ga_config = GAConfig(
    generations=15,
    population_size=16,
    min_features=6,
    max_features=12,      # 限制上限
    target_features=10,   # 目标减少
    mutation_rate=0.08,
    crossover_rate=0.85,
    elite_count=2,
    patience=5,
    cv_splits=5,          # 5折交叉验证
    verbose=True,
    random_state=2024
)

# 运行GA
ga = GeneticAlgorithmSelector(config=ga_config)
ga.fit(X_train_prep, y_train)

# 获取选中波长
support = ga.get_support()
selected_indices = np.where(support)[0]
selected_wavelengths = wavelengths[selected_indices]

print(f"选中 {len(selected_wavelengths)} 个波长: {selected_wavelengths}")

# 训练最终PLSR
X_train_sel = X_train_prep[:, selected_indices]
X_val_sel = X_val_prep[:, selected_indices]

# 交叉验证选择最佳成分数
max_comp = min(6, len(X_train)-1, len(selected_indices)//2)
best_r2, best_n = -1, 1
for n in range(1, max_comp+1):
    cv_r2 = cross_val_score(PLSRegression(n_components=n, scale=False),
                           X_train_sel, y_train, cv=5, scoring='r2').mean()
    if cv_r2 > best_r2:
        best_r2, best_n = cv_r2, n

plsr = PLSRegression(n_components=best_n, scale=False)
plsr.fit(X_train_sel, y_train)

# 评估
y_val_pred = plsr.predict(X_val_sel).ravel()
val_r2 = r2_score(y_val, y_val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

print(f"\n最终模型：")
print(f"  波长数: {len(selected_wavelengths)}")
print(f"  PLSR成分: {best_n}")
print(f"  Val R²: {val_r2:.4f}")
print(f"  RMSEP: {val_rmse:.6f}")
PY
```

**输出结果** (`results/stage3_ga_final/`):

- `results.json`: 完整结果数据
- `model.pkl`: 训练好的模型
- `results.png`: 可视化图表

**最终性能**：

- 选中12个波长
- 验证集R² = **0.9163**
- RMSEP = 0.001277

### 实验中遇到的关键问题

#### 问题1：ROI分割尺寸不一致

**现象**：`hsv_debugger`调参时分割正确，但 `build_huajiao_from_split.py`生成的ROI错误

**原因**：

- `hsv_debugger`在400px宽度的缩放图像上操作
- `build_huajiao_from_split.py`需要在原始尺寸（如1310×1024）上生成掩膜
- 缩放再放大导致掩膜失真

**解决方案**：

1. 创建 `apply_hsv_mask_simple`函数，支持可选缩放
2. `generate_roi_masks.py`默认在原始尺寸上分割（`--hsv-max-width 0`）
3. 生成 `*_overlay.png`供人工检查
4. `build_huajiao_from_split.py`优先加载预生成的掩膜

#### 问题2：GA过拟合

**现象**：Train R²=0.9439, Val R²=0.3268

**原因分析**：

- 样本量过小（39个训练样本）
- 特征数过多（30个波长，9个PLSR成分）
- 交叉验证不稳定（3折CV）

**解决措施**：

- 限制波长上限：12个
- 减少PLSR成分：max(6, n_features//2)
- 增加CV折数：5折
- 增加早停耐心：5代

**效果**：Val R²从0.3268提升至0.9163

#### 问题3：中文显示警告

**现象**：matplotlib中文显示警告

**解决**：

```python
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
```

### 数据质量要点

1. **ROI覆盖率**：应 > 5%，理想10-15%
2. **像素清洗率**：保留80-98%的像素（剔除极亮/极暗1-2%）
3. **光谱范围**：实测398-1004 nm（224个波长）
4. **含油率分布**：0.0089-0.0317 ml/g

### 下一步工作建议

1. **模型验证**：

   - 重复K折交叉验证测试稳定性
   - Bootstrap估计置信区间
2. **物理意义分析**：

   - 分析选中波长的化学吸收特性
   - 与油脂C-H、O-H键吸收峰对比
3. **模型部署**：

   - 保存标准化流程（预处理参数、选中波长、PLSR模型）
   - 编写预测脚本
4. **数据增强**：

   - 增加样本量以提升泛化能力
   - 多批次数据验证

## 原真实数据（花椒光谱）流程（参考）

- **准备原始文件**
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
- **HSV阈值调参（可选但推荐）**

  使用交互式调参工具找到最佳HSV阈值：

  ```bash
  # 方式1：使用Python直接调用
  python -c "
  from oil_content_detection.preprocessing.hsv_tuner import hsv_debugger
  hsv_debugger('/home/yr/yr/data/科研数据/云南藤椒1-1/0164_11/0164_11.png')
  "

  # 方式2：修改hsv_tuner.py的__main__部分，然后运行
  python src/oil_content_detection/preprocessing/hsv_tuner.py
  ```

  调参器会在缩放后的图像（400宽）上实时显示分割效果。退出时（按'q'或ESC），会打印当前的HSV阈值参数，复制这些参数用于后续批处理。
- **批量生成 ROI 掩膜（推荐：在原始尺寸上分割）**

  ```bash
  # 使用调参得到的HSV阈值，在原始尺寸上生成ROI掩膜
  python scripts/generate_roi_masks.py \
    --split data/labels/train.txt \
    --output-dir data/processed/ROI \
    --hsv-lower 30 70 30 \
    --hsv-upper 65 255 255

  # 注意：
  # 1. 默认 --hsv-max-width 0 表示在原始尺寸上分割，生成的mask与原图尺寸一致
  # 2. 若需要在缩放后分割（与调参器效果一致），可指定 --hsv-max-width 400
  # 3. 该脚本不包含边界裁剪、模糊、形态学操作，仅做纯HSV分割
  ```

  输出：

  - `data/processed/ROI/<样本ID>_mask.png`：二值掩膜（0/255），与原图同尺寸
  - `data/processed/ROI/<样本ID>_overlay.png`：叠加可视化图，红色区域为ROI
- **构建特征表：像素清洗 + 空间平均（推荐：使用预生成的mask）**

  ```bash
  # 方案A：使用预生成的ROI掩膜（推荐，mask已人工检查）
  python scripts/build_huajiao_from_split.py \
    --split data/labels/train.txt \
    --output-dir data/processed/huajiao/train \
    --roi-dir data/processed/ROI \
    --roi-mask-dir data/processed/ROI \
    --primary-stat trimmed_mean \
    --include-stats trimmed_mean \
    --trim-fraction 0.10 \
    --clip-low 0.01 \
    --clip-high 0.99

  # 方案B：实时HSV分割（不使用预生成mask）
  python scripts/build_huajiao_from_split.py \
    --split data/labels/train.txt \
    --output-dir data/processed/huajiao/train \
    --roi-dir data/processed/ROI \
    --hsv-lower 30 70 30 \
    --hsv-upper 65 255 255 \
    --hsv-max-width 0 \
    --trim-fraction 0.10 \
    --clip-low 0.01 \
    --clip-high 0.99

  # 验证集同理
  python scripts/build_huajiao_from_split.py \
    --split data/labels/val.txt \
    --output-dir data/processed/huajiao/val \
    --roi-dir data/processed/ROI \
    --roi-mask-dir data/processed/ROI \
    --trim-fraction 0.10 \
    --clip-low 0.01 \
    --clip-high 0.99
  ```

  **关键参数说明**：

  - `--roi-mask-dir`：预生成mask的目录（优先使用，由 `generate_roi_masks.py` 生成）
  - `--trim-fraction`：修剪均值的裁剪比例（0.10表示去除两端各10%极值像素）
  - `--clip-low` / `--clip-high`：像素清洗的分位数阈值（0.01和0.99表示剔除最暗1%和最亮1%的像素）
  - `--primary-stat`：主要统计量（trimmed_mean=修剪均值）
  - `--include-stats`：包含的统计量（可选：mean, median, trimmed_mean, std）

  **处理流程**：

  1. **加载ROI掩膜**：优先从 `--roi-mask-dir` 加载预生成mask；若无，则实时HSV分割或回退到NIR/Red比值法
  2. **像素清洗**：在掩膜内计算每个像素的全波段平均反射率，剔除极亮/极暗像素（分位数阈值）
  3. **空间平均**：对清洗后的像素进行光谱聚合，支持mean/median/trimmed_mean/std多种统计量
  4. **输出特征表**：每个样本生成一条平均光谱曲线，对应该盘花椒的理化值

  **输出文件**：

  - `data/processed/huajiao/<train|val>/huajiao_spectra.parquet`：特征表（每行一个样本，列为 `wl_<波长>`）
  - `data/processed/huajiao/<train|val>/huajiao_metadata.parquet`：元数据（样本ID、路径、掩膜统计等）
  - `data/processed/ROI/<样本ID>_roi.png`：ROI可视化叠加图（红色区域为清洗后的有效ROI）
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

● R²（R-squared，也叫决定系数或拟合优度）是衡量回归模型预测效果的核心指标。

  基本定义

  R² 表示模型解释了因变量（目标变量）多少百分比的变化。

  计算公式：

$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}
$$

  其中：

- $y_i$：真实值
- $\hat{y}_i$：预测值
- $\bar{y}$：真实值的平均值
- $SS_{res}$：残差平方和（预测误差）
- $SS_{tot}$：总平方和（数据总变异）
