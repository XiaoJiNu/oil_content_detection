# 可视化指南

本文档介绍如何可视化 GA + PLSR 实验结果。

## 快速开始

### 1. 运行实验并保存结果

```bash
python scripts/run_best_method.py \
  --output-dir results/my_experiment \
  --verbose \
  --ga-generations 20
```

### 2. 生成可视化图表

```bash
python scripts/visualize_results.py results/my_experiment
```

这将在 `results/my_experiment/plots/` 目录下生成 3 张图表。

---

## 可视化类型

### 1. GA 训练历史 (`ga_history.png`)

**展示内容：**
- 上图：每代的最佳适应度、平均适应度、适应度范围
- 下图：每代选中的特征数量变化

**用途：**
- 观察 GA 收敛过程
- 判断是否需要更多代数
- 检查早停是否合理

**示例命令：**
```bash
python scripts/visualize_results.py results/my_experiment --plot-type ga
```

---

### 2. 光谱特征选择 (`spectral_selection.png`)

**展示内容：**
- 所有样本的平均光谱曲线（蓝色实线）
- ±1 标准差范围（蓝色阴影）
- GA 选中的波长位置（红色虚线 + 圆点标记）

**用途：**
- 了解选中了哪些波段
- 分析选中波长的物理意义（如吸收峰位置）
- 验证特征选择的合理性

**示例命令：**
```bash
python scripts/visualize_results.py results/my_experiment --plot-type spectral
```

---

### 3. 预测结果对比 (`prediction_results.png`)

**展示内容：**
- 左图：训练集的预测值 vs 实际值散点图
- 右图：测试集的预测值 vs 实际值散点图
- 红色虚线：完美预测线（y=x）
- 标题显示 R² 和 RMSE 指标

**用途：**
- 评估模型拟合效果
- 检查是否过拟合（训练集远好于测试集）
- 识别异常样本点

**示例命令：**
```bash
python scripts/visualize_results.py results/my_experiment --plot-type prediction
```

---

## 高级用法

### 批量生成多个实验的可视化

```bash
#!/bin/bash
for exp in results/exp_*; do
    echo "Processing $exp..."
    python scripts/visualize_results.py "$exp"
done
```

### 自定义输出目录

```bash
# 保存到统一的 figures 目录
python scripts/visualize_results.py results/exp_01 --output-dir figures/exp_01
python scripts/visualize_results.py results/exp_02 --output-dir figures/exp_02
```

### 使用不同的数据集

```bash
# 如果使用了非默认的数据路径
python scripts/visualize_results.py results/custom_exp \
  --data path/to/custom_data.csv
```

---

## 在 Python 中使用

你也可以在 Python 脚本或 Jupyter Notebook 中直接调用可视化函数：

```python
from pathlib import Path
from oil_content_detection.visualization import (
    plot_ga_history,
    plot_spectral_selection,
    plot_prediction_results,
    plot_all_results,
)

# 方法 1: 生成所有图表
plot_all_results(
    results_dir=Path("results/my_experiment"),
    data_path=Path("data/processed/set_II/mean_spectra.csv"),
    output_dir=Path("figures/"),
)

# 方法 2: 单独生成某个图表
plot_ga_history(
    history_path=Path("results/my_experiment/ga_history.json"),
    save_path=Path("figures/ga_history.png"),
)

# 方法 3: 在 Notebook 中交互式显示（不保存文件）
plot_ga_history(
    history_path=Path("results/my_experiment/ga_history.json"),
    save_path=None,  # 不保存，直接显示
)
```

---

## 图表解读技巧

### GA 训练历史图

✅ **好的训练曲线：**
- 最佳适应度逐代上升
- 在某一代后趋于平稳（收敛）
- 早停发生时适应度已接近最优

⚠️ **需要注意的情况：**
- 适应度一直波动不收敛 → 增加代数或调整参数
- 第 1 代就达到最优 → 可能特征数设置不合理
- 特征数量剧烈波动 → 检查 `min_features` 和 `max_features` 设置

### 光谱选择图

✅ **合理的特征选择：**
- 选中的波长分布在多个光谱区域
- 覆盖了主要的吸收/反射峰位置
- 没有集中在某一个狭窄区域

⚠️ **需要注意的情况：**
- 所有波长都集中在某一小段 → 可能数据质量问题
- 完全随机分布无规律 → 增加 GA 代数或调整适应度函数

### 预测结果图

✅ **好的预测效果：**
- 散点紧密分布在 y=x 线附近
- 训练集和测试集 R² 都较高（>0.7）
- 测试集 R² 接近训练集（差距 <0.1）

⚠️ **需要注意的情况：**
- 训练集 R²=0.95，测试集 R²=0.5 → 严重过拟合
- 存在离群点远离主趋势线 → 检查数据质量
- 测试集 RMSE 远大于训练集 → 考虑增加正则化

---

## 图表自定义

如果需要修改图表样式（颜色、字体大小、分辨率等），可以编辑：
```
src/oil_content_detection/visualization/plots.py
```

常用调整：
- 修改 `plt.rcParams["figure.figsize"]` 改变图表大小
- 修改 `dpi=300` 改变分辨率
- 修改颜色代码（如 `#2E86AB`）改变配色方案

---

## 故障排查

**Q: 报错 "No module named 'matplotlib'"**
A: 安装可视化依赖：`pip install matplotlib seaborn`

**Q: 生成的图表是空白的**
A: 检查结果目录中是否包含必要的文件（`ga_history.json` 等）

**Q: 中文显示为方块**
A: 在 `plots.py` 开头添加：
```python
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或其他中文字体
```

**Q: 图表太小看不清**
A: 使用图片查看器的缩放功能，或修改 `figsize` 参数

---

## 完整工作流示例

```bash
# 1. 运行实验
python scripts/run_best_method.py \
  --output-dir results/exp_$(date +%Y%m%d_%H%M%S) \
  --verbose \
  --ga-generations 30 \
  --seed 42

# 2. 生成可视化
python scripts/visualize_results.py results/exp_20250930_133000

# 3. 查看图表
xdg-open results/exp_20250930_133000/plots/ga_history.png
```

---

## 更多信息

- 查看可视化脚本帮助：`python scripts/visualize_results.py --help`
- 源代码：`src/oil_content_detection/visualization/plots.py`
- 问题反馈：提交 issue 到项目仓库