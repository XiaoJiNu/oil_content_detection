#!/usr/bin/env python
"""
基线模型分析脚本
完成阶段二：EDA、基线PLSR、预处理对比
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data(train_path, val_path):
    """加载训练集和验证集"""
    train = pd.read_parquet(train_path)
    val = pd.read_parquet(val_path)

    # 提取波长列
    wl_cols = [c for c in train.columns if c.startswith('wl_')]
    wavelengths = np.array([float(c.split('_')[1]) for c in wl_cols])

    X_train = train[wl_cols].values
    y_train = train['oil_ml_per_gram'].values
    X_val = val[wl_cols].values
    y_val = val['oil_ml_per_gram'].values

    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")
    print(f"波长范围: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")
    print(f"含油率范围: {y_train.min():.4f} - {y_train.max():.4f}")

    return X_train, y_train, X_val, y_val, wavelengths, wl_cols


def eda_analysis(X_train, y_train, wavelengths, output_dir):
    """步骤9: 探索性数据分析"""
    print("\n" + "="*60)
    print("步骤9: 光谱数据探索性分析")
    print("="*60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 光谱曲线可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1.1 所有样本光谱
    ax = axes[0, 0]
    for i in range(len(X_train)):
        ax.plot(wavelengths, X_train[i], alpha=0.3, linewidth=0.5)
    ax.plot(wavelengths, X_train.mean(axis=0), 'r-', linewidth=2, label='平均光谱')
    ax.set_xlabel('波长 (nm)')
    ax.set_ylabel('反射率')
    ax.set_title(f'所有样本光谱曲线 (n={len(X_train)})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 1.2 含油率分布
    ax = axes[0, 1]
    ax.hist(y_train, bins=15, edgecolor='black', alpha=0.7)
    ax.axvline(y_train.mean(), color='r', linestyle='--', linewidth=2, label=f'均值={y_train.mean():.4f}')
    ax.axvline(np.median(y_train), color='g', linestyle='--', linewidth=2, label=f'中位数={np.median(y_train):.4f}')
    ax.set_xlabel('含油率 (ml/g)')
    ax.set_ylabel('频数')
    ax.set_title('含油率分布直方图')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 1.3 波长-含油率相关性
    ax = axes[1, 0]
    correlations = np.array([np.corrcoef(X_train[:, i], y_train)[0, 1] for i in range(X_train.shape[1])])
    ax.plot(wavelengths, correlations, linewidth=1.5)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('波长 (nm)')
    ax.set_ylabel('Pearson相关系数')
    ax.set_title('各波长与含油率的相关性')
    ax.grid(True, alpha=0.3)

    # 找出高相关波长
    high_corr_idx = np.argsort(np.abs(correlations))[-10:]
    print(f"\n相关性最高的10个波长:")
    for idx in high_corr_idx[::-1]:
        print(f"  {wavelengths[idx]:.1f} nm: r={correlations[idx]:.3f}")

    # 1.4 PCA降维可视化
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train)

    ax = axes[1, 1]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='viridis', s=100, edgecolor='k')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('PCA降维可视化')
    plt.colorbar(scatter, ax=ax, label='含油率 (ml/g)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / 'eda_analysis.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ EDA图表已保存: {fig_path}")
    plt.close()

    # 统计摘要
    print(f"\n数据统计摘要:")
    print(f"  样本数: {len(X_train)}")
    print(f"  波长数: {len(wavelengths)}")
    print(f"  含油率均值±标准差: {y_train.mean():.4f} ± {y_train.std():.4f}")
    print(f"  PCA前2个成分解释方差: {pca.explained_variance_ratio_[:2].sum():.1%}")


def train_baseline_plsr(X_train, y_train, X_val, y_val, output_dir):
    """步骤10: 基线PLSR模型"""
    print("\n" + "="*60)
    print("步骤10: 基线模型训练 (全波段PLSR)")
    print("="*60)

    output_dir = Path(output_dir)

    # 网格搜索最佳成分数
    max_components = min(20, len(X_train) - 1, X_train.shape[1])
    cv_scores = []
    train_scores = []

    print(f"\n寻找最佳PLSR成分数 (最多{max_components}个):")

    for n_comp in range(1, max_components + 1):
        plsr = PLSRegression(n_components=n_comp, scale=False)

        # 5折交叉验证
        kf = KFold(n_splits=5, shuffle=True, random_state=2024)
        cv_r2 = cross_val_score(plsr, X_train, y_train, cv=kf, scoring='r2')
        cv_scores.append(cv_r2.mean())

        # 训练集拟合
        plsr.fit(X_train, y_train)
        train_r2 = plsr.score(X_train, y_train)
        train_scores.append(train_r2)

        print(f"  n_components={n_comp:2d}: CV R²={cv_r2.mean():.4f}±{cv_r2.std():.4f}, Train R²={train_r2:.4f}")

    # 选择最佳成分数
    best_n = np.argmax(cv_scores) + 1
    print(f"\n最佳成分数: {best_n} (CV R²={cv_scores[best_n-1]:.4f})")

    # 训练最终模型
    plsr_best = PLSRegression(n_components=best_n, scale=False)
    plsr_best.fit(X_train, y_train)

    # 评估
    y_train_pred = plsr_best.predict(X_train).ravel()
    y_val_pred = plsr_best.predict(X_val).ravel()

    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

    print(f"\n基线模型性能 (全波段PLSR, n_components={best_n}):")
    print(f"  Train R²: {train_r2:.4f}, RMSE: {train_rmse:.6f}")
    print(f"  Val   R²: {val_r2:.4f}, RMSE: {val_rmse:.6f}")

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 成分数选择曲线
    ax = axes[0]
    ax.plot(range(1, max_components+1), train_scores, 'o-', label='Train R²')
    ax.plot(range(1, max_components+1), cv_scores, 's-', label='CV R²')
    ax.axvline(best_n, color='r', linestyle='--', alpha=0.5, label=f'最佳: {best_n}')
    ax.set_xlabel('PLSR成分数')
    ax.set_ylabel('R²')
    ax.set_title('PLSR成分数选择')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 训练集预测
    ax = axes[1]
    ax.scatter(y_train, y_train_pred, alpha=0.6, edgecolor='k')
    ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    ax.set_xlabel('实际含油率')
    ax.set_ylabel('预测含油率')
    ax.set_title(f'训练集 (R²={train_r2:.4f}, RMSE={train_rmse:.6f})')
    ax.grid(True, alpha=0.3)

    # 验证集预测
    ax = axes[2]
    ax.scatter(y_val, y_val_pred, alpha=0.6, edgecolor='k', color='orange')
    ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    ax.set_xlabel('实际含油率')
    ax.set_ylabel('预测含油率')
    ax.set_title(f'验证集 (R²={val_r2:.4f}, RMSE={val_rmse:.6f})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / 'baseline_plsr.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 基线模型图表已保存: {fig_path}")
    plt.close()

    return {
        'model': plsr_best,
        'n_components': best_n,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
    }


def apply_preprocessing(X, method, **kwargs):
    """应用预处理方法"""
    X = X.copy()

    if method == 'raw':
        return X

    elif method == 'snv':
        # 标准正态变量变换
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True)
        std[std == 0] = 1  # 避免除零
        return (X - mean) / std

    elif method == 'msc':
        # 多元散射校正
        mean_spectrum = X.mean(axis=0)
        X_corrected = np.zeros_like(X)
        for i in range(len(X)):
            # 线性拟合
            coef = np.polyfit(mean_spectrum, X[i], 1)
            # 校正
            X_corrected[i] = (X[i] - coef[1]) / coef[0]
        return X_corrected

    elif method == 'sg_1st':
        # Savitzky-Golay一阶导数
        from scipy.signal import savgol_filter
        window = kwargs.get('window', 9)
        polyorder = kwargs.get('polyorder', 2)
        return savgol_filter(X, window, polyorder, deriv=1, axis=1)

    elif method == 'detrend':
        # 去趋势
        from scipy.signal import detrend as scipy_detrend
        return scipy_detrend(X, axis=1)

    elif method == 'snv_1st':
        # SNV + 一阶导数
        X_snv = apply_preprocessing(X, 'snv')
        return apply_preprocessing(X_snv, 'sg_1st', **kwargs)

    else:
        raise ValueError(f"未知的预处理方法: {method}")


def compare_preprocessing(X_train, y_train, X_val, y_val, output_dir):
    """步骤11: 预处理方法对比"""
    print("\n" + "="*60)
    print("步骤11: 光谱预处理对比实验")
    print("="*60)

    output_dir = Path(output_dir)

    methods = {
        'raw': '原始数据',
        'snv': 'SNV',
        'msc': 'MSC',
        'sg_1st': 'SG一阶导',
        'detrend': '去趋势',
        'snv_1st': 'SNV+一阶导',
    }

    results = []

    for method, name in methods.items():
        print(f"\n测试: {name}")

        # 应用预处理
        try:
            X_train_prep = apply_preprocessing(X_train, method)
            X_val_prep = apply_preprocessing(X_val, method)
        except Exception as e:
            print(f"  ❌ 预处理失败: {e}")
            continue

        # 检查NaN
        if np.isnan(X_train_prep).any() or np.isnan(X_val_prep).any():
            print(f"  ❌ 预处理产生NaN值")
            continue

        # 最佳成分数搜索
        max_comp = min(15, len(X_train) - 1, X_train_prep.shape[1])
        best_cv_r2 = -np.inf
        best_n = 1

        for n_comp in range(1, max_comp + 1):
            plsr = PLSRegression(n_components=n_comp, scale=False)
            kf = KFold(n_splits=5, shuffle=True, random_state=2024)
            cv_r2 = cross_val_score(plsr, X_train_prep, y_train, cv=kf, scoring='r2').mean()
            if cv_r2 > best_cv_r2:
                best_cv_r2 = cv_r2
                best_n = n_comp

        # 训练最终模型
        plsr = PLSRegression(n_components=best_n, scale=False)
        plsr.fit(X_train_prep, y_train)

        y_train_pred = plsr.predict(X_train_prep).ravel()
        y_val_pred = plsr.predict(X_val_prep).ravel()

        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        rmsecv = np.sqrt(mean_squared_error(y_train, y_train_pred))  # 近似
        rmsep = np.sqrt(mean_squared_error(y_val, y_val_pred))

        results.append({
            'method': method,
            'name': name,
            'n_components': best_n,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'rmsecv': rmsecv,
            'rmsep': rmsep,
        })

        print(f"  ✓ n_comp={best_n}, Train R²={train_r2:.4f}, Val R²={val_r2:.4f}, RMSEP={rmsep:.6f}")

    # 结果汇总
    df_results = pd.DataFrame(results)
    print(f"\n预处理方法对比结果:")
    print(df_results.to_string(index=False))

    # 找出最佳方法
    best_idx = df_results['val_r2'].idxmax()
    best_method = df_results.loc[best_idx]
    print(f"\n✅ 最佳方法: {best_method['name']}")
    print(f"   Val R² = {best_method['val_r2']:.4f}, RMSEP = {best_method['rmsep']:.6f}")

    # 保存结果
    csv_path = output_dir / 'preprocessing_comparison.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"\n✓ 对比结果已保存: {csv_path}")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # R²对比
    ax = axes[0]
    x = np.arange(len(df_results))
    width = 0.35
    ax.bar(x - width/2, df_results['train_r2'], width, label='Train R²', alpha=0.8)
    ax.bar(x + width/2, df_results['val_r2'], width, label='Val R²', alpha=0.8)
    ax.set_xlabel('预处理方法')
    ax.set_ylabel('R²')
    ax.set_title('不同预处理方法的R²对比')
    ax.set_xticks(x)
    ax.set_xticklabels(df_results['name'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # RMSE对比
    ax = axes[1]
    ax.bar(x - width/2, df_results['rmsecv'], width, label='RMSECV', alpha=0.8)
    ax.bar(x + width/2, df_results['rmsep'], width, label='RMSEP', alpha=0.8)
    ax.set_xlabel('预处理方法')
    ax.set_ylabel('RMSE')
    ax.set_title('不同预处理方法的RMSE对比')
    ax.set_xticks(x)
    ax.set_xticklabels(df_results['name'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_path = output_dir / 'preprocessing_comparison.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ 对比图表已保存: {fig_path}")
    plt.close()

    return df_results, best_method


def main():
    """主函数"""
    print("\n" + "="*60)
    print("阶段二: 基线模型与预处理对比")
    print("="*60)

    # 路径设置
    train_path = "data/processed/huajiao/train/huajiao_spectra.parquet"
    val_path = "data/processed/huajiao/val/huajiao_spectra.parquet"
    output_dir = Path("results/baseline_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    X_train, y_train, X_val, y_val, wavelengths, wl_cols = load_data(train_path, val_path)

    # 步骤9: EDA
    eda_analysis(X_train, y_train, wavelengths, output_dir)

    # 步骤10: 基线PLSR
    baseline_result = train_baseline_plsr(X_train, y_train, X_val, y_val, output_dir)

    # 步骤11: 预处理对比
    preprocess_results, best_method = compare_preprocessing(X_train, y_train, X_val, y_val, output_dir)

    # 保存摘要（排除模型对象）
    baseline_summary = {k: v for k, v in baseline_result.items() if k != 'model'}

    summary = {
        'baseline': baseline_summary,
        'best_preprocessing': best_method.to_dict(),
    }

    import json
    summary_path = output_dir / 'stage2_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        # 转换numpy类型为Python原生类型
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        summary_clean = {k: {kk: convert(vv) for kk, vv in v.items()} if isinstance(v, dict) else convert(v)
                        for k, v in summary.items()}
        json.dump(summary_clean, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("阶段二完成!")
    print(f"{'='*60}")
    print(f"✓ 所有结果已保存至: {output_dir}")
    print(f"\n下一步:")
    print(f"  - 检查结果图表: {output_dir}/*.png")
    print(f"  - 如果基线模型Val R² > 0.70, 可以进入阶段三(GA特征选择)")
    print(f"  - 如果Val R² < 0.70, 建议检查数据质量或增加样本量")


if __name__ == '__main__':
    main()
