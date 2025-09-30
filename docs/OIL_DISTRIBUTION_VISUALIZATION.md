# 油茶籽含油分布可视化

本文档详细介绍如何生成每个油茶籽图像的空间含油率分布图，这是论文中的关键可视化技术。

## 功能概述

基于训练好的 PLSR 模型和 GA 选择的波长，对每个油茶籽的高光谱立方体进行**逐像素预测**，生成含油率的二维空间分布图。

### 原理

1. **输入**：高光谱立方体 (高×宽×波长数)
2. **特征选择**：使用 GA 选出的波长子集
3. **逐像素预测**：对 ROI 区域内每个像素应用 PLSR 模型
4. **可视化**：生成伪彩色热力图，展示含油率空间分布

---

## 快速开始

### 1. 准备工作

确保已经完成实验并保存了模型：

```bash
python scripts/run_best_method.py --output-dir results/my_exp
```

需要以下文件：
- `plsr_model.pkl` - 训练好的 PLSR 模型
- `feature_support.npy` - GA 选择的波长掩码
- `simulated_set_II_cube.npz` - 高光谱立方体数据

### 2. 生成可视化

#### 方式 A：生成摘要网格图（推荐）

显示 12 个随机样本的含油分布概览：

```bash
python scripts/visualize_oil_distribution.py results/my_exp --mode summary
```

输出：`results/my_exp/oil_distributions/oil_distribution_summary.png`

#### 方式 B：可视化特定样本

生成详细的三联图（原图 + ROI + 含油分布）：

```bash
python scripts/visualize_oil_distribution.py results/my_exp \
  --mode single \
  --sample-indices 0 1 2 3 4
```

输出：每个样本一张图，如 `sample_000_oil_distribution.png`

#### 方式 C：可视化所有样本

```bash
python scripts/visualize_oil_distribution.py results/my_exp --mode all
```

⚠️ 注意：如果有 102 个样本，将生成 102 张图片！

---

## 命令行参数详解

```bash
python scripts/visualize_oil_distribution.py <results_dir> [options]
```

### 必需参数

- `results_dir`: 包含训练结果的目录（含 model、support 文件）

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cube-data` | `data/processed/set_II/simulated_set_II_cube.npz` | 高光谱立方体数据路径 |
| `--output-dir` | `results_dir/oil_distributions` | 输出目录 |
| `--mode` | `summary` | 可视化模式：`summary`/`all`/`single` |
| `--sample-indices` | None | 指定样本索引（`single` 模式） |
| `--n-samples` | 12 | 摘要网格显示的样本数 |
| `--vmin` | 自动 | 色标最小值 |
| `--vmax` | 自动 | 色标最大值 |
| `--seed` | 42 | 随机种子 |

---

## 可视化图表解读

### 摘要网格图 (Summary Grid)

**布局**：4 列 × N 行的网格，每格显示一个样本

**每格内容**：
- 背景：灰度高光谱图像
- 前景：含油率伪彩色热力图
- 标题：样本 ID、实测含油率、预测平均含油率

**色标**：
- 🔵 蓝色 → 低含油率
- 🟢 青色 → 中低含油率
- 🟡 黄色 → 中等含油率
- 🟠 橙色 → 中高含油率
- 🔴 红色 → 高含油率

**用途**：
- 快速浏览多个样本的含油分布模式
- 识别含油率异常的样本
- 观察样本间的空间分布差异

---

### 单样本详细图 (Three-Panel Plot)

**左图：原始图像**
- 灰度高光谱图像（前 30 个波段平均）
- 黄色轮廓：ROI 区域边界

**中图：ROI 掩膜**
- 半透明灰度背景
- 绿色区域：种子 ROI
- 标题显示 ROI 像素数

**右图：含油分布**
- 伪彩色热力图
- 色标范围：[最小预测值, 最大预测值]
- 统计信息：
  - 实测平均含油率
  - 预测平均含油率 ± 标准差
  - 预测值范围

**用途**：
- 分析单个样本的详细含油分布
- 验证 ROI 提取是否准确
- 研究种子内部含油率的空间变异性

---

## 典型应用场景

### 场景 1：验证模型性能

```bash
# 随机选择 20 个样本生成摘要图
python scripts/visualize_oil_distribution.py results/exp_20250930 \
  --mode summary \
  --n-samples 20
```

观察预测含油率与实测值是否接近，判断模型泛化能力。

### 场景 2：分析异常样本

假设发现样本 15 的预测误差很大：

```bash
# 详细查看该样本
python scripts/visualize_oil_distribution.py results/exp_20250930 \
  --mode single \
  --sample-indices 15
```

检查是否存在：
- ROI 提取错误
- 含油率空间分布异常
- 高光谱数据质量问题

### 场景 3：研究含油率空间分布规律

```bash
# 可视化含油率最高和最低的样本
python scripts/visualize_oil_distribution.py results/exp_20250930 \
  --mode single \
  --sample-indices 0 5 10 50 90 101
```

观察高含油率和低含油率样本的空间分布特征是否不同。

### 场景 4：论文配图

```bash
# 生成统一色标范围的图，便于对比
python scripts/visualize_oil_distribution.py results/exp_20250930 \
  --mode single \
  --sample-indices 0 1 2 3 \
  --vmin 30 \
  --vmax 45
```

固定色标范围 [30%, 45%]，生成可直接用于论文的对比图。

---

## Python API 使用

在 Jupyter Notebook 或脚本中：

```python
from pathlib import Path
from oil_content_detection.visualization import (
    visualize_all_seeds,
    create_summary_grid,
    plot_seed_oil_distribution,
)

# 方法 1: 生成摘要网格
create_summary_grid(
    cube_data_path=Path("data/processed/set_II/simulated_set_II_cube.npz"),
    model_path=Path("results/my_exp/plsr_model.pkl"),
    support_path=Path("results/my_exp/feature_support.npy"),
    output_path=Path("figures/summary.png"),
    n_samples=16,
)

# 方法 2: 批量可视化
visualize_all_seeds(
    cube_data_path=Path("data/processed/set_II/simulated_set_II_cube.npz"),
    model_path=Path("results/my_exp/plsr_model.pkl"),
    support_path=Path("results/my_exp/feature_support.npy"),
    output_dir=Path("figures/individual"),
    sample_indices=[0, 1, 2, 3, 4],  # 或 None 表示全部
    vmin=30,
    vmax=45,
)

# 方法 3: 单样本自定义可视化
import numpy as np
import pickle

# 加载数据
data = np.load("data/processed/set_II/simulated_set_II_cube.npz")
with open("results/my_exp/plsr_model.pkl", "rb") as f:
    model = pickle.load(f)
support = np.load("results/my_exp/feature_support.npy")

# 预测单个样本
from oil_content_detection.visualization import predict_spatial_distribution

oil_map = predict_spatial_distribution(
    cube=data["cubes"][0],
    roi_mask=data["roi_masks"][0],
    model=model,
    support=support,
)

# 自定义可视化
plot_seed_oil_distribution(
    cube=data["cubes"][0],
    roi_mask=data["roi_masks"][0],
    oil_map=oil_map,
    mean_oil_content=data["oil_content"][0],
    sample_id="custom_sample",
    save_path=Path("custom_figure.png"),
)
```

---

## 技术细节

### 逐像素预测算法

```python
def predict_spatial_distribution(cube, roi_mask, model, support):
    """
    输入:
      cube: (H, W, N_wavelengths)
      roi_mask: (H, W) boolean
      support: (N_wavelengths,) boolean
    输出:
      oil_map: (H, W) with NaN for background
    """
    # 1. 提取 ROI 像素
    roi_pixels = cube[roi_mask]  # (n_roi, N_wavelengths)

    # 2. 选择 GA 特征
    roi_pixels_selected = roi_pixels[:, support]  # (n_roi, N_selected)

    # 3. PLSR 预测
    predictions = model.predict(roi_pixels_selected)  # (n_roi,)

    # 4. 回填到空间位置
    oil_map = np.full((H, W), np.nan)
    oil_map[roi_mask] = predictions

    return oil_map
```

### 色标设计

自定义色标从 7 种颜色线性插值生成 256 级：

```python
colors = [
    "#2C3E50",  # 深蓝 - 极低
    "#3498DB",  # 蓝色 - 低
    "#1ABC9C",  # 青色 - 中低
    "#F1C40F",  # 黄色 - 中等
    "#E67E22",  # 橙色 - 中高
    "#E74C3C",  # 红色 - 高
    "#C0392B",  # 深红 - 极高
]
```

### 性能优化

- 单样本预测耗时：~0.5 秒（24×24 像素）
- 102 个样本全部可视化：约 50 秒
- 摘要网格（12 样本）：约 7 秒

如需加速，可考虑并行处理：

```python
from multiprocessing import Pool

def process_sample(idx):
    # 预测 + 保存图片
    pass

with Pool(4) as p:
    p.map(process_sample, range(102))
```

---

## 故障排查

**Q: 报错 "Model file not found"**
A: 确保 results_dir 中包含 `plsr_model.pkl` 和 `feature_support.npy`

**Q: 图像全是灰色，看不到含油分布**
A: 检查 ROI 区域是否有效，或尝试手动设置 `--vmin` 和 `--vmax`

**Q: 预测值全部接近平均值，没有空间变异**
A: 这是正常现象（模拟数据），真实高光谱数据会有更丰富的空间变异

**Q: 色标范围不合理**
A: 使用 `--vmin` 和 `--vmax` 手动设置，例如：
```bash
--vmin 20 --vmax 50
```

**Q: 想要不同的配色方案**
A: 编辑 `spatial_distribution.py` 中的 `create_oil_content_colormap()` 函数

---

## 与论文对比

本实现与论文《基于高光谱成像的油茶籽含油率检测方法》中的可视化方法一致：

| 论文方法 | 本实现 |
|---------|--------|
| 高光谱立方体采集 | ✅ 模拟数据/支持真实数据 |
| ROI 提取 | ✅ 椭圆形 ROI 掩膜 |
| 波长选择（GA） | ✅ 完全复现 |
| PLSR 建模 | ✅ 完全复现 |
| 逐像素预测 | ✅ 实现 |
| 伪彩色可视化 | ✅ 自定义色标 |

---

## 后续改进方向

- [ ] 添加含油率分布的统计直方图
- [ ] 支持 3D 可视化（立体展示）
- [ ] 导出动画（旋转查看）
- [ ] 对比多个模型的预测差异
- [ ] 叠加原始 RGB 图像

---

## 相关文档

- 基础可视化：`VISUALIZATION_GUIDE.md`
- 优化总结：`OPTIMIZATION_SUMMARY.md`
- 主 README：`README.md`