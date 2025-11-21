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
