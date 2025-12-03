#!/usr/bin/env python
"""
样本级预测脚本
用于加载已训练的模型,生成每个样本的详细预测结果

使用方法:
  python scripts/predict_samples.py
  python scripts/predict_samples.py --output results/stage3_ga_final/val_predictions.csv --plot
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示 - 使用AR PL UMing中文字体
font_path = '/usr/share/fonts/truetype/arphic/uming.ttc'
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['axes.unicode_minus'] = False


def load_model(model_path):
    """加载模型文件"""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    print(f"✓ 模型已加载: {model_path}")

    # 兼容不同的模型保存格式
    if 'wavelengths' in model_data:
        # 格式1: plsr, support, wavelengths
        selected_wavelengths = model_data['wavelengths']
        model = model_data['plsr']
        support = model_data['support']
        n_components = model.n_components
    elif 'selected_wavelengths' in model_data:
        # 格式2: model, selected_wavelengths, selected_indices
        selected_wavelengths = model_data['selected_wavelengths']
        model = model_data['model']
        support = model_data['selected_indices']
        n_components = model_data.get('n_components', model.n_components)
    else:
        raise ValueError("不支持的模型文件格式")

    print(f"  - 选中波长数: {len(selected_wavelengths)}")
    print(f"  - PLSR成分数: {n_components}")

    # 重新打包为统一格式
    normalized_data = {
        'model': model,
        'support': support,
        'selected_wavelengths': selected_wavelengths,
        'n_components': n_components
    }

    return normalized_data


def load_data(data_path):
    """加载验证集数据"""
    df = pd.read_parquet(data_path)

    # 提取样本ID和波长列
    sample_ids = df['sample_id'].values
    wl_cols = [c for c in df.columns if c.startswith('wl_')]
    X = df[wl_cols].values
    y = df['oil_ml_per_gram'].values

    wavelengths = np.array([float(c.split('_')[1]) for c in wl_cols])

    print(f"\n✓ 数据已加载: {data_path}")
    print(f"  - 样本数: {len(sample_ids)}")
    print(f"  - 波长数: {len(wl_cols)}")
    print(f"  - 含油率范围: {y.min():.6f} - {y.max():.6f} ml/g")

    return sample_ids, X, y, wavelengths


def apply_preprocessing(X):
    """应用SG一阶导预处理"""
    X_prep = savgol_filter(X, 9, 2, deriv=1, axis=1)
    print(f"\n✓ 预处理完成: SG一阶导 (window=9, polyorder=2)")
    return X_prep


def predict_samples(model_data, X, y, sample_ids):
    """生成样本级预测"""
    model = model_data['model']
    support = model_data['support']
    selected_wavelengths = model_data['selected_wavelengths']

    # 提取选中波长
    X_selected = X[:, support]

    # 预测
    y_pred = model.predict(X_selected).ravel()

    # 计算误差
    abs_error = np.abs(y - y_pred)
    rel_error = (y - y_pred) / y * 100

    # 构建结果DataFrame
    results_df = pd.DataFrame({
        'sample_id': sample_ids,
        'actual_oil_ml_per_g': y,
        'predicted_oil_ml_per_g': y_pred,
        'absolute_error': abs_error,
        'relative_error_percent': rel_error
    })

    # 计算总体指标
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = np.mean(abs_error)
    mape = np.mean(np.abs(rel_error))

    metrics = {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'n_samples': len(y)
    }

    print(f"\n✓ 预测完成")
    print(f"  - R²: {r2:.4f}")
    print(f"  - RMSE: {rmse:.6f}")
    print(f"  - MAE: {mae:.6f}")
    print(f"  - MAPE: {mape:.2f}%")

    return results_df, metrics, selected_wavelengths


def print_results_table(results_df, metrics):
    """打印结果表格"""
    print("\n" + "="*70)
    print("样本级预测详情")
    print("="*70)

    # 打印表头
    print(f"{'样本ID':<12} {'真实值':>12} {'预测值':>12} {'绝对误差':>12} {'相对误差%':>12}")
    print("-" * 70)

    # 打印每个样本
    for _, row in results_df.iterrows():
        print(f"{row['sample_id']:<12} "
              f"{row['actual_oil_ml_per_g']:>12.6f} "
              f"{row['predicted_oil_ml_per_g']:>12.6f} "
              f"{row['absolute_error']:>12.6f} "
              f"{row['relative_error_percent']:>12.2f}")

    print("-" * 70)
    r2_str = f"R²={metrics['r2']:.4f}"
    rmse_str = f"RMSE={metrics['rmse']:.6f}"
    mae_str = f"MAE={metrics['mae']:.6f}"
    mape_str = f"MAPE={metrics['mape']:.2f}%"
    print(f"{'总体指标':<12} {r2_str:>12} {rmse_str:>12} {mae_str:>12} {mape_str:>12}")
    print("="*70)


def save_results(results_df, output_path):
    """保存结果到CSV"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_path, index=False, float_format='%.6f')
    print(f"\n✓ 结果已保存: {output_path}")


def plot_results(results_df, metrics, selected_wavelengths, output_path):
    """生成可视化图表"""
    fig = plt.figure(figsize=(15, 5))

    # 子图1: 预测vs实际散点图
    ax1 = plt.subplot(1, 3, 1)
    y_actual = results_df['actual_oil_ml_per_g'].values
    y_pred = results_df['predicted_oil_ml_per_g'].values

    ax1.scatter(y_actual, y_pred, alpha=0.6, edgecolor='k', s=80)

    # 添加样本ID标注
    for i, row in results_df.iterrows():
        ax1.annotate(row['sample_id'],
                    (row['actual_oil_ml_per_g'], row['predicted_oil_ml_per_g']),
                    fontsize=7, alpha=0.7,
                    xytext=(3, 3), textcoords='offset points')

    # 添加y=x线
    lim_min = min(y_actual.min(), y_pred.min())
    lim_max = max(y_actual.max(), y_pred.max())
    ax1.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', lw=2, alpha=0.7)

    ax1.set_xlabel('实际含油率 (ml/g)', fontsize=11)
    ax1.set_ylabel('预测含油率 (ml/g)', fontsize=11)
    ax1.set_title(f'预测 vs 实际\nR²={metrics["r2"]:.4f}, RMSE={metrics["rmse"]:.6f}',
                 fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 子图2: 误差条形图
    ax2 = plt.subplot(1, 3, 2)
    errors = results_df['absolute_error'].values
    sample_ids = results_df['sample_id'].values
    colors = ['red' if e > errors.mean() else 'green' for e in errors]

    bars = ax2.barh(range(len(sample_ids)), errors, color=colors, alpha=0.6, edgecolor='k')
    ax2.set_yticks(range(len(sample_ids)))
    ax2.set_yticklabels(sample_ids, fontsize=9)
    ax2.set_xlabel('绝对误差 (ml/g)', fontsize=11)
    ax2.set_title(f'各样本误差分布\nMAE={metrics["mae"]:.6f}', fontsize=12)
    ax2.axvline(errors.mean(), color='blue', linestyle='--', lw=2, alpha=0.7,
               label=f'平均误差={errors.mean():.6f}')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='x')

    # 子图3: 相对误差分布
    ax3 = plt.subplot(1, 3, 3)
    rel_errors = results_df['relative_error_percent'].values

    ax3.hist(rel_errors, bins=10, alpha=0.6, edgecolor='k', color='orange')
    ax3.axvline(0, color='red', linestyle='--', lw=2, alpha=0.7, label='零误差')
    ax3.axvline(rel_errors.mean(), color='blue', linestyle='--', lw=2, alpha=0.7,
               label=f'平均={rel_errors.mean():.2f}%')
    ax3.set_xlabel('相对误差 (%)', fontsize=11)
    ax3.set_ylabel('样本数', fontsize=11)
    ax3.set_title(f'相对误差分布\nMAPE={metrics["mape"]:.2f}%', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='生成样本级预测结果')
    parser.add_argument('--model', type=str,
                       default='results/stage3_ga_final/model.pkl',
                       help='模型文件路径 (默认: results/stage3_ga_final/model.pkl)')
    parser.add_argument('--data', type=str,
                       default='data/processed/huajiao/val/huajiao_spectra.parquet',
                       help='数据文件路径 (默认: 验证集)')
    parser.add_argument('--output', type=str,
                       default='results/stage3_ga_final/val_predictions.csv',
                       help='输出CSV文件路径')
    parser.add_argument('--plot', type=str, nargs='?', const='auto',
                       help='生成可视化图表 (可选指定路径,默认与CSV同目录)')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("阶段三GA+PLSR验证集预测")
    print("="*70)

    # 1. 加载模型
    model_data = load_model(args.model)

    # 2. 加载数据
    sample_ids, X, y, wavelengths = load_data(args.data)

    # 3. 预处理
    X_prep = apply_preprocessing(X)

    # 4. 预测
    results_df, metrics, selected_wavelengths = predict_samples(
        model_data, X_prep, y, sample_ids
    )

    # 5. 打印结果表格
    print_results_table(results_df, metrics)

    # 6. 保存CSV
    if args.output:
        save_results(results_df, args.output)

    # 7. 生成可视化 (如果指定)
    if args.plot:
        if args.plot == 'auto':
            plot_path = Path(args.output).parent / 'val_predictions_plot.png'
        else:
            plot_path = args.plot
        plot_results(results_df, metrics, selected_wavelengths, plot_path)

    print("\n✓ 全部完成!")


if __name__ == '__main__':
    main()
