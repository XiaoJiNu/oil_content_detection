# 油茶籽含油率检测复现项目

本仓库用于复现论文《基于高光谱成像的油茶籽含油率检测方法》中的最佳方案（光谱集Ⅱ + 遗传算法 + PLSR），当前提供模拟数据与完整代码骨架，便于后续替换为真实高光谱/含油率测定数据。

## 项目目标
- 构建油茶籽高光谱数据的预处理、波段筛选与回归建模流程。
- 通过遗传算法筛选特征波长并训练 PLS 回归模型，预测含油率。
- 为未来接入真实设备与实验数据提供可扩展的工程框架与文档。

## 快速开始
1. **准备环境**（示例命令）
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install numpy pandas scikit-learn
   ```
2. **运行复现流程（使用模拟数据）**
   ```bash
   PYTHONPATH=src MKL_THREADING_LAYER=SEQUENTIAL \
   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
   python scripts/run_best_method.py
   ```
   输出包括：遗传算法筛选出的波长列表、训练/测试集 R² 与 RMSE 指标。
3. **更换为真实数据**
   - 将实测光谱整理为与 `data/processed/simulated_spectral_set_II.csv` 相同的列格式（`sample_id`、`wl_<波长>`、`oil_content`）。
   - 覆盖该文件或调整 `RunConfig.data_path` 指向新数据后重新运行脚本。

## 仓库结构
- `src/oil_content_detection/`：核心源码
  - `feature_selection/ga_selector.py`：遗传算法波段筛选
  - `models/plsr_best.py`：GA + PLSR 管线
- `scripts/run_best_method.py`：一键执行最佳方案
- `data/processed/simulated_spectral_set_II.csv`：模拟光谱数据集
- `docs/reference_docs/`：需求文档、复现计划与实验记录
- `AGENTS.md`：贡献者指南

## 文档与记录
- `docs/reference_docs/功能需求文档/复现计划.md`：整体复现计划与架构说明。
- `docs/reference_docs/实验记录/模拟复现记录.md`：模拟数据实验过程与指标。

## 下一步建议
- 接入真实高光谱采集系统，补充数据预处理与含油率标定脚本。
- 将模型训练、可视化等步骤拓展为 CLI/Notebook 形式，便于批量实验。
- 引入自动化测试覆盖关键模块（特征选择、PLSR 拟合）。
