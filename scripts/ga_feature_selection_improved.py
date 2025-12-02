#!/usr/bin/env python
"""
阶段三改进版：使用更保守的GA参数
针对小样本量数据的优化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error
from scipy.signal import savgol_filter
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def apply_sg_derivative(X, window=9, polyorder=2, deriv=1):
    """应用Savitzky-Golay一阶导数"""
    return savgol_filter(X, window, polyorder, deriv=deriv, axis=1)


def load_data(train_path, val_path):
    """加载训练集和验证集"""
    train = pd.read_parquet(train_path)
    val = pd.read_parquet(val_path)

    wl_cols = [c for c in train.columns if c.startswith('wl_')]
    wavelengths = np.array([float(c.split('_')[1]) for c in wl_cols])

    X_train = train[wl_cols].values
    y_train = train['oil_ml_per_gram'].values
    X_val = val[wl_cols].values
    y_val = val['oil_ml_per_gram'].values

    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")
    print(f"波长范围: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")

    return X_train, y_train, X_val, y_val, wavelengths, wl_cols


class GeneticAlgorithmSelector:
    """遗传算法特征选择器 - 小样本优化版"""

    def __init__(self, generations=30, population_size=20, min_features=5,
                 max_features=15, target_features=10, mutation_rate=0.10,
                 crossover_rate=0.80, tournament_size=3, patience=8,
                 n_components_ratio=0.3, random_state=2024):
        self.generations = generations
        self.population_size = population_size
        self.min_features = min_features
        self.max_features = max_features
        self.target_features = target_features
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.patience = patience
        self.n_components_ratio = n_components_ratio  # PLSR成分数 = 特征数 * ratio
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.best_individual_ = None
        self.best_score_ = -np.inf
        self.support_ = None
        self.history_ = []

    def _initialize_population(self, n_features):
        """初始化种群"""
        population = []
        for _ in range(self.population_size):
            n_selected = self.rng.randint(self.min_features, self.max_features + 1)
            individual = np.zeros(n_features, dtype=bool)
            selected_idx = self.rng.choice(n_features, n_selected, replace=False)
            individual[selected_idx] = True
            population.append(individual)
        return population

    def _evaluate_fitness(self, individual, X, y):
        """评估个体适应度 - 使用留一交叉验证"""
        n_selected = individual.sum()
        if n_selected < 3:
            return -1.0

        X_selected = X[:, individual]

        # 根据特征数动态设置PLSR成分数，避免过拟合
        n_components = max(2, min(5, int(n_selected * self.n_components_ratio)))

        try:
            plsr = PLSRegression(n_components=n_components, scale=False)

            # 小样本使用留一交叉验证
            loo = LeaveOneOut()
            cv_r2 = cross_val_score(plsr, X_selected, y, cv=loo, scoring='r2').mean()

            # 软约束：偏离目标特征数的惩罚（减小惩罚系数）
            feature_penalty = abs(n_selected - self.target_features) / self.target_features
            fitness = cv_r2 - 0.02 * feature_penalty  # 减小惩罚系数从0.05到0.02

            return fitness
        except:
            return -1.0

    def _tournament_selection(self, population, fitness_scores):
        """锦标赛选择"""
        idx = self.rng.choice(len(population), self.tournament_size, replace=False)
        best_idx = idx[np.argmax([fitness_scores[i] for i in idx])]
        return population[best_idx].copy()

    def _crossover(self, parent1, parent2):
        """单点交叉"""
        if self.rng.rand() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        point = self.rng.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2

    def _mutate(self, individual):
        """变异"""
        for i in range(len(individual)):
            if self.rng.rand() < self.mutation_rate:
                individual[i] = not individual[i]

        # 确保特征数在范围内
        n_selected = individual.sum()
        if n_selected < self.min_features:
            n_add = self.min_features - n_selected
            false_idx = np.where(~individual)[0]
            if len(false_idx) >= n_add:
                add_idx = self.rng.choice(false_idx, n_add, replace=False)
                individual[add_idx] = True
        elif n_selected > self.max_features:
            n_remove = n_selected - self.max_features
            true_idx = np.where(individual)[0]
            remove_idx = self.rng.choice(true_idx, n_remove, replace=False)
            individual[remove_idx] = False

        return individual

    def fit(self, X, y, verbose=True):
        """运行遗传算法"""
        n_features = X.shape[1]

        # 初始化种群
        population = self._initialize_population(n_features)

        best_gen_score = -np.inf
        no_improve_count = 0

        for gen in range(self.generations):
            # 评估适应度
            fitness_scores = [self._evaluate_fitness(ind, X, y) for ind in population]

            # 更新最佳个体
            gen_best_idx = np.argmax(fitness_scores)
            gen_best_score = fitness_scores[gen_best_idx]

            if gen_best_score > self.best_score_:
                self.best_score_ = gen_best_score
                self.best_individual_ = population[gen_best_idx].copy()
                self.support_ = self.best_individual_

            # 记录历史
            self.history_.append({
                'generation': gen,
                'best_fitness': gen_best_score,
                'mean_fitness': np.mean(fitness_scores),
                'n_features': population[gen_best_idx].sum(),
            })

            if verbose:
                n_feat = population[gen_best_idx].sum()
                print(f"  Gen {gen+1:2d}: Best Fitness={gen_best_score:.4f}, "
                      f"Mean={np.mean(fitness_scores):.4f}, Features={n_feat}")

            # 早停检查
            if gen_best_score > best_gen_score + 1e-6:  # 增加容差
                best_gen_score = gen_best_score
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= self.patience:
                if verbose:
                    print(f"  早停: {self.patience}代无明显改进")
                break

            # 生成新一代
            new_population = []

            # 精英保留（保留更多精英）
            elite_count = max(2, self.population_size // 10)
            elite_idx = np.argsort(fitness_scores)[-elite_count:]
            for idx in elite_idx:
                new_population.append(population[idx].copy())

            # 生成剩余个体
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                new_population.extend([child1, child2])

            population = new_population[:self.population_size]

        return self


def run_ga_selection(X_train, y_train, wavelengths, output_dir, verbose=True):
    """步骤12: 运行GA特征选择 - 改进版"""
    print("\n" + "="*60)
    print("步骤12: GA特征波长选择 (小样本优化版)")
    print("="*60)

    output_dir = Path(output_dir)

    # 配置GA - 更保守的参数
    ga = GeneticAlgorithmSelector(
        generations=30,
        population_size=20,
        min_features=5,
        max_features=15,  # 减少上限，避免过拟合
        target_features=10,  # 目标10个波长
        mutation_rate=0.10,  # 增加变异率，增加多样性
        crossover_rate=0.80,
        tournament_size=3,
        patience=8,
        n_components_ratio=0.3,  # PLSR成分数 = 特征数 * 0.3
        random_state=2024
    )

    print(f"\nGA配置 (小样本优化):")
    print(f"  代数: {ga.generations}, 种群: {ga.population_size}")
    print(f"  特征范围: {ga.min_features}-{ga.max_features}, 目标: {ga.target_features}")
    print(f"  变异率: {ga.mutation_rate}, 交叉率: {ga.crossover_rate}")
    print(f"  PLSR成分比例: {ga.n_components_ratio}")
    print(f"  交叉验证: 留一法 (LOO)")

    print(f"\n开始GA搜索...")
    ga.fit(X_train, y_train, verbose=verbose)

    # 获取选中的波长
    selected_indices = np.where(ga.support_)[0]
    selected_wavelengths = wavelengths[selected_indices]

    print(f"\n✓ GA完成!")
    print(f"  选中 {len(selected_wavelengths)} 个波长")
    print(f"  最佳适应度 (LOO R²): {ga.best_score_:.4f}")
    print(f"\n选中的波长:")
    for i, wl in enumerate(selected_wavelengths):
        if (i + 1) % 10 == 0:
            print(f"  {wl:.1f} nm")
        else:
            print(f"  {wl:.1f} nm", end="")
    print()

    # 可视化GA历史
    history_df = pd.DataFrame(ga.history_)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 适应度曲线
    ax = axes[0]
    ax.plot(history_df['generation'], history_df['best_fitness'], 'o-', label='最佳适应度', linewidth=2)
    ax.plot(history_df['generation'], history_df['mean_fitness'], 's-', label='平均适应度', alpha=0.7)
    ax.set_xlabel('代数')
    ax.set_ylabel('适应度 (LOO R²)')
    ax.set_title('GA训练历史')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 特征数变化
    ax = axes[1]
    ax.plot(history_df['generation'], history_df['n_features'], 'o-', color='green', linewidth=2)
    ax.axhline(ga.target_features, color='r', linestyle='--', alpha=0.5, label=f'目标: {ga.target_features}')
    ax.set_xlabel('代数')
    ax.set_ylabel('特征数')
    ax.set_title('选中特征数变化')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 波长分布
    ax = axes[2]
    ax.scatter(wavelengths, np.zeros_like(wavelengths), alpha=0.3, s=10, label='所有波长')
    ax.scatter(selected_wavelengths, np.zeros_like(selected_wavelengths), color='red', s=50,
               marker='v', label=f'选中波长 (n={len(selected_wavelengths)})')
    ax.set_xlabel('波长 (nm)')
    ax.set_yticks([])
    ax.set_title('选中波长分布')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    fig_path = output_dir / 'ga_selection_improved.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ GA图表已保存: {fig_path}")
    plt.close()

    # 保存结果
    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj

    ga_results = {
        'selected_wavelengths': selected_wavelengths.tolist(),
        'selected_indices': selected_indices.tolist(),
        'best_fitness': float(ga.best_score_),
        'n_selected': int(len(selected_wavelengths)),
        'history': convert_numpy(ga.history_),
    }

    json_path = output_dir / 'ga_results_improved.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(ga_results, f, indent=2, ensure_ascii=False)
    print(f"✓ GA结果已保存: {json_path}")

    return ga, selected_indices, selected_wavelengths


def train_final_model(X_train, y_train, X_val, y_val, selected_indices,
                     selected_wavelengths, output_dir):
    """训练最终的GA+PLSR模型 - 改进版"""
    print("\n" + "="*60)
    print("使用选中波长训练最终PLSR模型 (保守策略)")
    print("="*60)

    output_dir = Path(output_dir)

    # 提取选中的波长
    X_train_selected = X_train[:, selected_indices]
    X_val_selected = X_val[:, selected_indices]

    # 最佳成分数搜索 - 限制上限
    n_features = len(selected_indices)
    max_comp = min(7, len(X_train) - 1, n_features // 2)  # 更保守的上限
    cv_scores = []

    print(f"\n寻找最佳成分数 (最多{max_comp}个):")
    for n_comp in range(1, max_comp + 1):
        plsr = PLSRegression(n_components=n_comp, scale=False)
        kf = KFold(n_splits=5, shuffle=True, random_state=2024)
        cv_r2 = cross_val_score(plsr, X_train_selected, y_train, cv=kf, scoring='r2').mean()
        cv_scores.append(cv_r2)
        print(f"  n_components={n_comp:2d}: CV R²={cv_r2:.4f}")

    best_n = np.argmax(cv_scores) + 1
    print(f"\n最佳成分数: {best_n} (CV R²={cv_scores[best_n-1]:.4f})")

    # 训练最终模型
    plsr_final = PLSRegression(n_components=best_n, scale=False)
    plsr_final.fit(X_train_selected, y_train)

    # 评估
    y_train_pred = plsr_final.predict(X_train_selected).ravel()
    y_val_pred = plsr_final.predict(X_val_selected).ravel()

    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

    print(f"\nGA+PLSR最终模型性能:")
    print(f"  特征数: {len(selected_wavelengths)} / 224 (减少 {(1 - len(selected_wavelengths)/224)*100:.1f}%)")
    print(f"  PLSR成分: {best_n}")
    print(f"  Train R²: {train_r2:.4f}, RMSE: {train_rmse:.6f}")
    print(f"  Val   R²: {val_r2:.4f}, RMSE: {val_rmse:.6f}")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 训练集预测
    ax = axes[0]
    ax.scatter(y_train, y_train_pred, alpha=0.6, edgecolor='k')
    ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    ax.set_xlabel('实际含油率 (ml/g)')
    ax.set_ylabel('预测含油率 (ml/g)')
    ax.set_title(f'训练集 (R²={train_r2:.4f}, RMSE={train_rmse:.6f})')
    ax.grid(True, alpha=0.3)

    # 验证集预测
    ax = axes[1]
    ax.scatter(y_val, y_val_pred, alpha=0.6, edgecolor='k', color='orange')
    ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    ax.set_xlabel('实际含油率 (ml/g)')
    ax.set_ylabel('预测含油率 (ml/g)')
    ax.set_title(f'验证集 (R²={val_r2:.4f}, RMSE={val_rmse:.6f})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / 'ga_plsr_performance_improved.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 性能图表已保存: {fig_path}")
    plt.close()

    # 保存模型
    model_data = {
        'model': plsr_final,
        'selected_wavelengths': selected_wavelengths,
        'selected_indices': selected_indices,
        'n_components': best_n,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
    }

    model_path = output_dir / 'ga_plsr_model_improved.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"✓ 模型已保存: {model_path}")

    return model_data


def main():
    """主函数"""
    print("\n" + "="*60)
    print("阶段三改进版: GA特征波长选择 (小样本优化)")
    print("="*60)

    # 路径设置
    train_path = "data/processed/huajiao/train/huajiao_spectra.parquet"
    val_path = "data/processed/huajiao/val/huajiao_spectra.parquet"
    output_dir = Path("results/ga_selection_improved")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    X_train, y_train, X_val, y_val, wavelengths, wl_cols = load_data(train_path, val_path)

    # 应用最佳预处理（SG一阶导）
    print("\n应用预处理: SG一阶导...")
    X_train_prep = apply_sg_derivative(X_train)
    X_val_prep = apply_sg_derivative(X_val)
    print("✓ 预处理完成")

    # 步骤12: GA选择 - 改进版
    ga, selected_indices, selected_wavelengths = run_ga_selection(
        X_train_prep, y_train, wavelengths, output_dir, verbose=True
    )

    # 训练最终模型 - 改进版
    model_data = train_final_model(
        X_train_prep, y_train, X_val_prep, y_val,
        selected_indices, selected_wavelengths, output_dir
    )

    # 对比分析
    print("\n" + "="*60)
    print("步骤13: 结果分析与对比")
    print("="*60)

    # 加载基线结果
    baseline_path = Path("results/baseline_analysis/stage2_summary.json")
    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            baseline_results = json.load(f)

        baseline_val_r2 = baseline_results['baseline']['val_r2']
        best_prep_val_r2 = baseline_results['best_preprocessing']['val_r2']

        print(f"\n模型性能对比:")
        print(f"{'方法':<30} {'波长数':<10} {'成分数':<10} {'Val R²':<12} {'RMSEP':<12}")
        print("-" * 80)
        print(f"{'全波段PLSR (Raw)':<30} {224:<10} {baseline_results['baseline']['n_components']:<10} {baseline_val_r2:<12.4f} {baseline_results['baseline']['val_rmse']:<12.6f}")
        print(f"{'全波段PLSR (SG一阶导)':<30} {224:<10} {baseline_results['best_preprocessing']['n_components']:<10} {best_prep_val_r2:<12.4f} {baseline_results['best_preprocessing']['rmsep']:<12.6f}")
        print(f"{'GA+PLSR (SG一阶导,改进版)':<30} {len(selected_wavelengths):<10} {model_data['n_components']:<10} {model_data['val_r2']:<12.4f} {model_data['val_rmse']:<12.6f}  ✅")

        improvement = model_data['val_r2'] - baseline_val_r2
        improvement_vs_best = model_data['val_r2'] - best_prep_val_r2
        print(f"\n相比Raw基线改进: {improvement:+.4f} ({improvement/baseline_val_r2*100:+.1f}%)")
        print(f"相比SG一阶导改进: {improvement_vs_best:+.4f} ({improvement_vs_best/best_prep_val_r2*100:+.1f}%)")

    print(f"\n{'='*60}")
    print("阶段三完成!")
    print(f"{'='*60}")
    print(f"✓ 所有结果已保存至: {output_dir}")
    print(f"\n关键输出:")
    print(f"  - GA选择的波长: {output_dir}/ga_results_improved.json")
    print(f"  - 最终模型: {output_dir}/ga_plsr_model_improved.pkl")
    print(f"  - 可视化图表: {output_dir}/*.png")


if __name__ == '__main__':
    main()
