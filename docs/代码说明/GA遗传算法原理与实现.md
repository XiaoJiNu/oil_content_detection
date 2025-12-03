# GA 遗传算法原理与实现说明

## 目录
1. [遗传算法基本原理](#1-遗传算法基本原理)
2. [本项目中的 GA 应用场景](#2-本项目中的-ga-应用场景)
3. [算法流程详解](#3-算法流程详解)
4. [核心参数说明](#4-核心参数说明)
5. [代码实现细节](#5-代码实现细节)
6. [使用示例](#6-使用示例)

---

## 1. 遗传算法基本原理

遗传算法 (Genetic Algorithm, GA) 是一种模拟自然界生物进化过程的优化算法,属于启发式搜索算法。其核心思想来源于达尔文的"适者生存"理论。

### 1.1 生物学类比

| 生物学概念 | GA 概念 | 本项目实现 |
|----------|--------|----------|
| 个体 | 解的编码 | 布尔掩码数组 (选中/未选中波长) |
| 种群 | 解的集合 | 多个波长选择方案 (population_size=12) |
| 基因 | 解的组成部分 | 单个波长是否被选择 |
| 适应度 | 解的质量评价 | PLSR 交叉验证 R² 分数 |
| 繁殖 | 产生新解 | 交叉、变异操作 |
| 代 (Generation) | 迭代过程 | 每次种群更新 |

### 1.2 算法优势

- **全局搜索能力强**: 不容易陷入局部最优
- **无需梯度信息**: 适用于黑盒优化问题
- **并行性强**: 种群中的个体可以并行评估
- **适合组合优化**: 特别适合特征选择这类离散优化问题

---

## 2. 本项目中的 GA 应用场景

### 2.1 问题定义

**任务**: 从高光谱数据的 601 个波长中选择 ~18 个最有代表性的波长,用于 PLSR 模型训练

**为什么需要 GA?**
- 搜索空间巨大: C(601, 18) ≈ 10^27 种组合
- 暴力搜索不可行
- 贪心策略易陷入局部最优
- GA 可以在合理时间内找到高质量的特征子集

### 2.2 编码方式

**个体表示**: 布尔掩码 (Boolean Mask)
```python
# 示例: 601 个波长,True 表示该波长被选中
individual = np.array([False, True, False, ..., True, False])  # 长度 601
selected_count = individual.sum()  # 约 18 个
```

### 2.3 适应度函数

**目标**: 使 PLSR 模型的交叉验证 R² 分数最大化,同时倾向于选择目标数量的特征

**公式**:
```
fitness = mean_cv_r2 - penalty
penalty = 0.004 × |selected_count - target_features|
```

**设计理念**:
- 主项 `mean_cv_r2`: 评估预测性能 (3-fold CV)
- 惩罚项 `penalty`: 软约束,引导特征数量接近目标 (18 个)
- 系数 0.004: 经验值,平衡性能与特征数量

**实现代码** (`ga_selector.py:62-73`):
```python
def _evaluate(self, X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    selected = np.where(mask)[0]
    if selected.size == 0:
        return float("-inf")
    n_components = self._pls_components(selected.size)
    model = PLSRegression(n_components=n_components, scale=False)
    cv = KFold(self.config.cv_splits, shuffle=True, random_state=self.config.random_state)
    scores = cross_val_score(model, X[:, selected], y, scoring="r2", cv=cv, n_jobs=None)
    mean_score = scores.mean()
    # 软惩罚,偏向目标特征数量
    penalty = 0.004 * abs(selected.size - self.config.target_features)
    return mean_score - penalty
```

---

## 3. 算法流程详解

### 3.1 整体流程图

```
初始化
  ├─ 生成初始种群 (12 个随机个体)
  └─ 评估初始适应度

迭代进化 (最多 10 代)
  ├─ 选择操作 (锦标赛选择)
  │   └─ 从种群中挑选优秀个体作为父代
  ├─ 交叉操作 (单点交叉, 概率 0.85)
  │   └─ 父代基因重组产生子代
  ├─ 变异操作 (比特翻转, 概率 0.04)
  │   └─ 随机改变某些基因增加多样性
  ├─ 边界修正
  │   └─ 确保特征数量在 [10, 22] 范围内
  ├─ 评估新种群适应度
  └─ 精英保留 (保留最佳个体)

终止条件
  ├─ 达到最大代数 (10)
  └─ 早停 (4 代无改善)
```

### 3.2 详细步骤

#### Step 1: 初始化种群 (`_initial_population`, ga_selector.py:75-81)

```python
def _initial_population(self, n_features: int) -> np.ndarray:
    pop = np.zeros((self.config.population_size, n_features), dtype=bool)
    for i in range(self.config.population_size):
        # 每个个体随机选择 [min_features, max_features] 个特征
        active = self._rng.integers(self.config.min_features, self.config.max_features + 1)
        idx = self._rng.choice(n_features, size=active, replace=False)
        pop[i, idx] = True
    return pop
```

**策略**: 随机生成,确保初始多样性

#### Step 2: 选择操作 - 锦标赛选择 (`_select_parents`, ga_selector.py:83-90)

```python
def _select_parents(self, population: np.ndarray, fitness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    def pick_one() -> np.ndarray:
        contenders = self._rng.choice(population.shape[0], size=3, replace=False)
        best_idx = contenders[np.argmax(fitness[contenders])]
        return population[best_idx]
    return pick_one(), pick_one()
```

**机制**:
- 随机选取 3 个候选个体
- 选择其中适应度最高的作为父代
- 重复两次得到两个父代

**优势**: 比轮盘赌选择更稳定,避免优势个体过早支配种群

#### Step 3: 交叉操作 - 单点交叉 (`_crossover`, ga_selector.py:92-98)

```python
def _crossover(self, parent_a: np.ndarray, parent_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if self._rng.random() > self.config.crossover_rate:  # 0.85
        return parent_a.copy(), parent_b.copy()
    cut = self._rng.integers(1, parent_a.size - 1)
    child_a = np.concatenate([parent_a[:cut], parent_b[cut:]])
    child_b = np.concatenate([parent_b[:cut], parent_a[cut:]])
    return child_a, child_b
```

**图示**:
```
父代 A: [1, 1, 0, 0, 1, 0, 1]
                 ↑ 切割点
父代 B: [0, 1, 1, 1, 0, 1, 0]

子代 A: [1, 1, 0, | 1, 0, 1, 0]  # 前半来自 A,后半来自 B
子代 B: [0, 1, 1, | 0, 1, 0, 1]  # 前半来自 B,后半来自 A
```

#### Step 4: 变异操作 - 比特翻转 (`_mutate`, ga_selector.py:100-103)

```python
def _mutate(self, mask: np.ndarray) -> np.ndarray:
    mutation_flags = self._rng.random(mask.size) < self.config.mutation_rate  # 0.04
    mask ^= mutation_flags  # XOR 操作翻转选中的比特
    return mask
```

**作用**: 以 4% 的概率翻转每个基因,避免早熟收敛

**示例**:
```
原始: [1, 0, 1, 1, 0]
变异: [1, 1, 1, 1, 0]  # 第2位从0变为1
```

#### Step 5: 边界修正 (`_ensure_bounds`, ga_selector.py:45-53)

```python
@staticmethod
def _ensure_bounds(mask: np.ndarray, cfg: GAConfig, rng: np.random.Generator) -> np.ndarray:
    idx = np.where(mask)[0]
    if idx.size < cfg.min_features:
        # 特征太少,随机添加
        add_idx = rng.choice(np.where(~mask)[0], size=cfg.min_features - idx.size, replace=False)
        mask[add_idx] = True
    elif idx.size > cfg.max_features:
        # 特征太多,随机删除
        drop_idx = rng.choice(idx, size=idx.size - cfg.max_features, replace=False)
        mask[drop_idx] = False
    return mask
```

**必要性**: 交叉和变异后可能产生不合法的个体 (特征数过多或过少)

#### Step 6: 精英保留 (ga_selector.py:120-121)

```python
elite_indices = np.argsort(fitness)[-self.config.elite_count:]
elites = population[elite_indices]
```

**策略**: 直接将当前代最优的 1 个个体保留到下一代,防止最优解丢失

#### Step 7: 早停机制 (ga_selector.py:150-153)

```python
if self.config.patience and no_improve >= self.config.patience:
    if self.config.verbose:
        print(f"Early stopping at generation {gen + 1} (no improvement for {self.config.patience} generations)")
    break
```

**条件**: 连续 4 代没有改善,提前终止

---

## 4. 核心参数说明

### 4.1 GAConfig 参数表

| 参数 | 默认值 | 作用 | 调优建议 |
|-----|-------|------|---------|
| `population_size` | 12 | 种群大小 | 太小易早熟,太大计算慢 |
| `generations` | 10 | 最大迭代代数 | 观察收敛曲线决定 |
| `crossover_rate` | 0.85 | 交叉概率 | 通常 0.7-0.9 |
| `mutation_rate` | 0.04 | 变异概率 | 太高破坏好解,太低多样性不足 |
| `elite_count` | 1 | 精英个体数 | 小规模种群建议 1-2 个 |
| `min_features` | 10 | 最少特征数 | 下界约束 |
| `max_features` | 22 | 最多特征数 | 上界约束 |
| `target_features` | 18 | 目标特征数 | 软约束,影响惩罚项 |
| `cv_splits` | 3 | 交叉验证折数 | 影响评估可靠性 |
| `patience` | 4 | 早停耐心值 | 防止过度迭代 |
| `verbose` | False | 是否打印训练进度 | 调试时建议 True |

### 4.2 参数设置原则

**种群大小与迭代次数的权衡**:
- 小种群 + 多代数: 收敛慢但计算快
- 大种群 + 少代数: 收敛快但单代计算慢
- 本项目: 12 × 10 = 120 次适应度评估 (每次评估需 3-fold CV)

**变异率设置**:
- 经验公式: `mutation_rate ≈ 1 / gene_count`
- 本项目: 1/601 ≈ 0.0017,但使用 0.04 (偏大)
- 原因: 高光谱特征高度相关,需要更强探索能力

---

## 5. 代码实现细节

### 5.1 模块结构

```
src/oil_content_detection/feature_selection/ga_selector.py
├─ GAConfig (dataclass)              # 配置参数
├─ GeneticAlgorithmSelector (class)  # 主类
│   ├─ __init__                      # 初始化
│   ├─ fit                           # 训练入口
│   ├─ _initial_population           # 生成初始种群
│   ├─ _evaluate                     # 适应度评估 (PLSR CV)
│   ├─ _select_parents               # 锦标赛选择
│   ├─ _crossover                    # 单点交叉
│   ├─ _mutate                       # 比特翻转变异
│   ├─ _ensure_bounds                # 边界修正
│   ├─ _record_generation            # 记录历史
│   ├─ get_support                   # 获取最佳掩码
│   └─ selected_indices              # 获取选中索引
└─ select_wavelengths (function)     # 便捷接口
```

### 5.2 关键设计点

#### 5.2.1 PLSR 成分数自适应 (ga_selector.py:56-60)

```python
@staticmethod
def _pls_components(feature_count: int) -> int:
    if feature_count <= 2:
        return 1
    return min(10, feature_count // 2)
```

**原理**:
- PLSR 成分数不能超过特征数
- 经验规则: 成分数 = 特征数的一半,上限 10
- 确保数值稳定性

#### 5.2.2 线程安全问题 (plsr_best.py:21)

```python
setup_single_thread()  # 模块导入时强制单线程
```

**原因**:
- scikit-learn 的 `cross_val_score` 内部使用 `joblib` 多线程
- 种群中多个个体并行评估时,会产生嵌套并行
- NumPy/OpenBLAS/MKL 等数值库可能冲突
- 解决方案: 环境变量强制单线程 (`OMP_NUM_THREADS=1`)

#### 5.2.3 训练历史记录 (ga_selector.py:157-174)

```python
def _record_generation(self, gen: int, fitness: np.ndarray, best_score: float, best_features: int) -> None:
    history_entry = {
        "generation": gen,
        "best_score": best_score,
        "mean_score": fitness.mean(),
        "max_score": fitness.max(),
        "min_score": fitness.min(),
        "best_features": best_features,
    }
    self._history.append(history_entry)
```

**用途**:
- 可视化收敛曲线
- 诊断过拟合或早熟
- 保存在 `results/*/ga_history.json`

---

## 6. 使用示例

### 6.1 基本用法

```python
from oil_content_detection.feature_selection.ga_selector import GAConfig, GeneticAlgorithmSelector
import numpy as np

# 准备数据
X_train = np.random.randn(68, 601)  # 68 样本, 601 个波长
y_train = np.random.randn(68)        # 含油率

# 配置 GA
config = GAConfig(
    population_size=12,
    generations=10,
    target_features=18,
    verbose=True  # 打印训练进度
)

# 训练
selector = GeneticAlgorithmSelector(config)
selector.fit(X_train, y_train)

# 获取结果
best_mask = selector.get_support()          # 布尔掩码
selected_idx = selector.selected_indices()  # 选中的索引
best_score = selector.best_score()          # 最佳适应度

print(f"Selected {selected_idx.size} features with CV R² = {best_score:.4f}")
print(f"Indices: {selected_idx}")
```

### 6.2 完整流程 (与 PLSR 结合)

```python
from oil_content_detection.models.plsr_best import RunConfig, train_plsr_best
from pathlib import Path

# 配置
config = RunConfig(
    data_path=Path("data/processed/set_II/mean_spectra.csv"),
    ga_generations=15,       # 增加迭代次数
    ga_population=20,        # 增加种群规模
    target_features=20,      # 调整目标特征数
    output_dir=Path("results/custom_run"),
    verbose=True
)

# 运行
result = train_plsr_best(config)

# 结果
print(f"Selected wavelengths: {result.selected_wavelengths}")
print(f"Test R²: {result.test_r2:.4f}")
```

### 6.3 可视化训练过程

```python
import matplotlib.pyplot as plt

# 读取训练历史
history = selector.history()
generations = [h["generation"] for h in history]
best_scores = [h["best_score"] for h in history]
mean_scores = [h["mean_score"] for h in history]

# 绘制收敛曲线
plt.figure(figsize=(10, 6))
plt.plot(generations, best_scores, label="Best Score", marker="o")
plt.plot(generations, mean_scores, label="Mean Score", marker="s")
plt.xlabel("Generation")
plt.ylabel("Fitness (CV R²)")
plt.title("GA Training History")
plt.legend()
plt.grid(True)
plt.show()
```

---

## 附录: 常见问题

### Q1: 为什么适应度是 R² 而不是 RMSE?
**A**: R² 无量纲,取值 [0, 1],更适合作为优化目标。RMSE 依赖于目标变量的尺度。

### Q2: 如何判断 GA 是否收敛?
**A**: 观察 `ga_history.json` 中的 `best_score`,如果连续多代不再上升,说明已收敛。

### Q3: 参数如何调优?
**A**:
1. 先用小规模参数 (pop=6, gen=5) 快速测试
2. 观察收敛曲线,如果未收敛则增加 `generations`
3. 如果早熟 (过早收敛到次优解),增加 `mutation_rate` 或 `population_size`

### Q4: GA 能保证找到全局最优吗?
**A**: 不能。GA 是启发式算法,但通过多次运行 (不同 `random_state`) 可以提高找到全局最优的概率。

### Q5: 为什么不使用多线程加速?
**A**: 交叉验证已经是并行操作,嵌套并行会导致线程冲突。建议在外层 (多次实验) 并行,而非内层。

---

**相关文件**:
- 核心实现: `src/oil_content_detection/feature_selection/ga_selector.py`
- 集成流程: `src/oil_content_detection/models/plsr_best.py`
- 运行脚本: `scripts/run_best_method.py`
- 测试用例: `tests/test_ga_selector.py`
