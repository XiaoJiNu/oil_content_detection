#!/usr/bin/env python
"""
阶段三：GA特征波长选择
使用遗传算法筛选关键波长，并训练优化后的PLSR模型
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, KFold
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
    """遗传算法特征选择器"""

    def __init__(self, generations=20, population_size=12, min_features=8,
                 max_features=30, target_features=18, mutation_rate=0.04,
                 crossover_rate=0.85, tournament_size=3, patience=4, random_state=2024):
        self.generations = generations
        self.population_size = population_size
        self.min_features = min_features
        self.max_features = max_features
        self.target_features = target_features
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.patience = patience
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
        """评估个体适应度"""
        if individual.sum() < 3:
            return -1.0

        X_selected = X[:, individual]
        n_components = min(10, individual.sum() // 2)

        try:
            plsr = PLSRegression(n_components=n_components, scale=False)
            kf = KFold(n_splits=3, shuffle=True, random_state=self.random_state)
            cv_r2 = cross_val_score(plsr, X_selected, y, cv=kf, scoring='r2').mean()

            # 软约束：偏离目标特征数的惩罚
            n_selected = individual.sum()
            feature_penalty = abs(n_selected - self.target_features) / self.target_features
            fitness = cv_r2 - 0.05 * feature_penalty

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
            if gen_best_score > best_gen_score:
                best_gen_score = gen_best_score
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= self.patience:
                if verbose:
                    print(f"  早停: {self.patience}代无改进")
                break

            # 生成新一代
            new_population = []

            # 精英保留
            elite_idx = np.argsort(fitness_scores)[-2:]
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
    """步骤12: 运行GA特征选择"""
    print("\n" + "="*60)
    print("步骤12: GA特征波长选择")
    print("="*60)

    output_dir = Path(output_dir)

    # 配置GA
    ga = GeneticAlgorithmSelector(
        generations=20,
        population_size=12,
        min_features=8,
        max_features=30,
        target_features=18,
        mutation_rate=0.04,
        crossover_rate=0.85,
        tournament_size=3,
        patience=4,
        random_state=2024
    )

    print(f"\nGA配置:")
    print(f"  代数: {ga.generations}, 种群: {ga.population_size}")
    print(f"  特征范围: {ga.min_features}-{ga.max_features}, 目标: {ga.target_features}")
    print(f"  变异率: {ga.mutation_rate}, 交叉率: {ga.crossover_rate}")

    print(f"\n开始GA搜索...")
    ga.fit(X_train, y_train, verbose=verbose)

    # 获取选中的波长
    selected_indices = np.where(ga.support_)[0]
    selected_wavelengths = wavelengths[selected_indices]

    print(f"\n✓ GA完成!")
    print(f"  选中 {len(selected_wavelengths)} 个波长")
    print(f"  最佳适应度: {ga.best_score_:.4f}")
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
    ax.set_ylabel('适应度 (CV R²)')
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
    fig_path = output_dir / 'ga_selection.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ GA图表已保存: {fig_path}")
    plt.close()

    # 保存结果（转换numpy类型）
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

    json_path = output_dir / 'ga_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(ga_results, f, indent=2, ensure_ascii=False)
    print(f"✓ GA结果已保存: {json_path}")

    return ga, selected_indices, selected_wavelengths


def train_final_model(X_train, y_train, X_val, y_val, selected_indices,
                     selected_wavelengths, output_dir):
    """训练最终的GA+PLSR模型"""
    print("\n" + "="*60)
    print("使用选中波长训练最终PLSR模型")
    print("="*60)

    output_dir = Path(output_dir)

    # 提取选中的波长
    X_train_selected = X_train[:, selected_indices]
    X_val_selected = X_val[:, selected_indices]

    # 最佳成分数搜索
    max_comp = min(15, len(X_train) - 1, len(selected_indices) // 2)
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
    fig_path = output_dir / 'ga_plsr_performance.png'
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

    model_path = output_dir / 'ga_plsr_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"✓ 模型已保存: {model_path}")

    return model_data


def main():
    """主函数"""
    print("\n" + "="*60)
    print("阶段三: GA特征波长选择与模型优化")
    print("="*60)

    # 路径设置
    train_path = "data/processed/huajiao/train/huajiao_spectra.parquet"
    val_path = "data/processed/huajiao/val/huajiao_spectra.parquet"
    output_dir = Path("results/ga_selection")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    X_train, y_train, X_val, y_val, wavelengths, wl_cols = load_data(train_path, val_path)

    # 应用最佳预处理（SG一阶导）
    print("\n应用预处理: SG一阶导...")
    X_train_prep = apply_sg_derivative(X_train)
    X_val_prep = apply_sg_derivative(X_val)
    print("✓ 预处理完成")

    # 步骤12: GA选择
    ga, selected_indices, selected_wavelengths = run_ga_selection(
        X_train_prep, y_train, wavelengths, output_dir, verbose=True
    )

    # 训练最终模型
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
        print(f"{'方法':<20} {'波长数':<10} {'Val R²':<12} {'RMSEP':<12}")
        print("-" * 60)
        print(f"{'全波段PLSR (Raw)':<20} {224:<10} {baseline_val_r2:<12.4f} {baseline_results['baseline']['val_rmse']:<12.6f}")
        print(f"{'全波段PLSR (SG一阶导)':<20} {224:<10} {best_prep_val_r2:<12.4f} {baseline_results['best_preprocessing']['rmsep']:<12.6f}")
        print(f"{'GA+PLSR (SG一阶导)':<20} {len(selected_wavelengths):<10} {model_data['val_r2']:<12.4f} {model_data['val_rmse']:<12.6f}  ✅")

        improvement = model_data['val_r2'] - baseline_val_r2
        print(f"\n相比基线改进: {improvement:+.4f} ({improvement/baseline_val_r2*100:+.1f}%)")

    print(f"\n{'='*60}")
    print("阶段三完成!")
    print(f"{'='*60}")
    print(f"✓ 所有结果已保存至: {output_dir}")
    print(f"\n关键输出:")
    print(f"  - GA选择的波长: {output_dir}/ga_results.json")
    print(f"  - 最终模型: {output_dir}/ga_plsr_model.pkl")
    print(f"  - 可视化图表: {output_dir}/*.png")
    print(f"\n下一步: 进入阶段四(模型验证与部署)")


if __name__ == '__main__':
    main()
