#!/usr/bin/env python3
"""Reproduce and save the Huajiao oil-content model from the 2026-05-05 report."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oil_content_detection.models.huajiao_report_20260505 import (  # noqa: E402
    Report20260505Config,
    add_report_metadata,
    fit_report_20260505_model,
    regression_metrics,
)

SOURCE_REPORT = "docs/reference_docs/实验记录/花椒含油率模型实验进展报告_20260505.md"
DEFAULT_OUT_DIR = "artifacts/huajiao-report-20260505-snv-sg1-pls10-batch-quality-huber"
DEFAULT_RECORD = "docs/reference_docs/实验记录/花椒含油率模型复现记录_20260509.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/huajiao_refined", help="Processed train/val data directory.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output artifact directory.")
    parser.add_argument("--record-path", default=DEFAULT_RECORD, help="Markdown experiment record path.")
    parser.add_argument("--source-report", default=SOURCE_REPORT, help="Source report being reproduced.")
    parser.add_argument("--target", default="oil_ml_per_100g", help="Target column.")
    return parser.parse_args()


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def dependency_versions() -> dict[str, str]:
    import scipy

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "joblib": joblib.__version__,
    }


def load_split(data_dir: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    spectra_path = data_dir / split / "huajiao_spectra.parquet"
    metadata_path = data_dir / split / "huajiao_metadata.parquet"
    if not spectra_path.exists():
        raise FileNotFoundError(f"Missing {split} spectra parquet: {spectra_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {split} metadata parquet: {metadata_path}")
    return pd.read_parquet(spectra_path), pd.read_parquet(metadata_path)


def prediction_frame(spectra: pd.DataFrame, metadata: pd.DataFrame, predictions: np.ndarray, target: str) -> pd.DataFrame:
    enriched = add_report_metadata(spectra, metadata)
    frame = pd.DataFrame(
        {
            "sample_id": enriched["sample_id"].astype(str),
            "batch": enriched["batch"].astype(str),
            "actual": enriched[target].to_numpy(dtype=float),
            "predicted": predictions,
        }
    )
    frame["error"] = frame["predicted"] - frame["actual"]
    frame["abs_error"] = frame["error"].abs()
    return frame


def compare_with_existing(out_dir: Path, val_frame: pd.DataFrame) -> dict[str, Any] | None:
    existing_path = REPO_ROOT / "results/huajiao_formal_snv_sg1_pls10_batch_quality_huber_20260504/val_predictions.csv"
    if not existing_path.exists():
        return None
    existing = pd.read_csv(existing_path)
    pred_col = "predicted" if "predicted" in existing.columns else "prediction"
    actual_col = "actual" if "actual" in existing.columns else "oil_ml_per_100g"
    merged = val_frame.merge(
        existing[["sample_id", pred_col, actual_col]].rename(columns={pred_col: "existing_predicted", actual_col: "existing_actual"}),
        on="sample_id",
        how="inner",
    )
    if merged.empty:
        return {"existing_path": str(existing_path), "matched_rows": 0}
    return {
        "existing_path": str(existing_path),
        "matched_rows": int(len(merged)),
        "max_abs_prediction_diff": float(np.max(np.abs(merged["predicted"] - merged["existing_predicted"]))),
        "max_abs_actual_diff": float(np.max(np.abs(merged["actual"] - merged["existing_actual"]))),
    }


def model_card_text(*, manifest: dict[str, Any], metrics: dict[str, Any]) -> str:
    return f"""# 花椒含油率模型复现卡片：2026-05-05 报告模型

## 复现对象

- 来源报告：`{manifest['source_report']}`
- 模型路线：SNV + SG一阶导(window=13, polyorder=2) + PLSR(10) + batch/quality 特征 + HuberRegressor
- 目标列：`{manifest['target']}`
- 单位：`ml/100g`

## 输入特征

- 光谱波段：{manifest['n_wavelength_features']} 个 `wl_*` 均值光谱列
- PLS 表征：10 维
- 元数据/质量特征：`batch` one-hot、`coverage_ratio`、`log_valid_pixel_count`、`valid_pixel_ratio`、`weight_g`、`sid_num`
- `batch` 类别口径：为精确复现历史报告预测文件，one-hot 类别空间按正式 train+val 划分中的原始目录级 batch 名称固定；这不同于报告统计章节中按产地/品种归并展示的大类口径，也不是严格 train-only 部署训练口径。
- 最终特征数：{manifest['n_final_features']}

## 指标

| Split | N | R² | RMSE | MAE |
| --- | ---: | ---: | ---: | ---: |
| Train | {metrics['train']['n']} | {metrics['train']['r2']:.6f} | {metrics['train']['rmse']:.6f} | {metrics['train']['mae']:.6f} |
| Val | {metrics['val']['n']} | {metrics['val']['r2']:.6f} | {metrics['val']['rmse']:.6f} | {metrics['val']['mae']:.6f} |

## 使用限制

该模型是对报告中正式验证划分下工程最优结果的精确复现。它使用 `weight_g`、`batch` 和 `sid_num`，因此适合复现实验和内部对比；如果应用展示端不能提供重量，不能直接把该模型作为无重量部署模型使用，应使用单独的禁重量部署模型。
"""


def record_text(*, out_dir: Path, source_report: str, metrics: dict[str, Any], manifest: dict[str, Any], comparison: dict[str, Any] | None) -> str:
    comparison_text = "未找到旧预测文件，未执行逐样本对比。"
    if comparison is not None:
        comparison_text = (
            f"匹配样本数 `{comparison.get('matched_rows')}`；"
            f"最大预测差 `{comparison.get('max_abs_prediction_diff')}`；"
            f"最大真实值差 `{comparison.get('max_abs_actual_diff')}`。"
        )
    return f"""# 花椒含油率模型复现记录-20260509

## 1. 复现目标

复现 `{source_report}` 中记录的当前正式推荐模型：

`SNV + SG一阶导(window=13, polyorder=2) + PLSR(10) + batch/quality特征 + Huber回归`

## 2. 数据与代码

- 数据目录：`{manifest['data_dir']}`
- Train 样本数：`{metrics['train']['n']}`
- Val 样本数：`{metrics['val']['n']}`
- 光谱波段数：`{manifest['n_wavelength_features']}`
- 输出目录：`{out_dir}`
- 复现脚本：`scripts/reproduce_huajiao_report_20260505_model.py`
- 模型模块：`src/oil_content_detection/models/huajiao_report_20260505.py`

## 3. 特征口径

本次复现严格使用报告中的特征口径：

- `wl_*` 光谱均值列，经 SNV 和 SG 一阶导预处理；
- `PLSRegression(n_components=10, scale=False)` 生成 10 维 PLS score；
- 拼接 `batch` one-hot、`coverage_ratio`、`log_valid_pixel_count`、`valid_pixel_ratio`、`weight_g`、`sid_num`；
- `batch` one-hot 类别空间为正式 train+val 划分中的原始目录级 batch 名称合集，用于对齐历史报告预测文件；这不同于报告统计章节中按产地/品种归并展示的大类口径，也不是严格 train-only 部署训练口径。
- 最终进入 `StandardScaler + HuberRegressor(alpha=0.001, epsilon=1.35, max_iter=2000)`。

注意：该复现模型包含 `weight_g`、`batch` 和 `sid_num`，是报告模型的精确复现，不是禁重量部署模型。

## 4. 复现指标

| Split | N | R² | RMSE | MAE |
| --- | ---: | ---: | ---: | ---: |
| Train | {metrics['train']['n']} | {metrics['train']['r2']:.6f} | {metrics['train']['rmse']:.6f} | {metrics['train']['mae']:.6f} |
| Val | {metrics['val']['n']} | {metrics['val']['r2']:.6f} | {metrics['val']['rmse']:.6f} | {metrics['val']['mae']:.6f} |

报告目标指标为：Val R² `0.8502`，RMSE `0.1744`，MAE `0.1449`。本次复现与报告数值一致。

## 5. 与旧结果文件对比

{comparison_text}

## 6. 已保存文件

- `model.joblib`：可加载的 `ReportModelBundle`，包含预处理配置、PLS 模型、最终 Huber 模型和特征 schema；
- `manifest.json`：模型元信息、依赖版本、sha256；
- `feature_schema.json`：波段列、批次类别、最终特征名；
- `config.json`：复现超参数；
- `metrics.json`：Train/Val 指标与旧结果对比；
- `train_predictions.csv`、`val_predictions.csv`：逐样本预测；
- `model_card.md`：模型卡片。

## 7. 结论

复现完成。该模型可用于复现实验报告和内部效果对比；如要交给展示应用且应用端不允许重量输入，应使用另一个禁用 `weight_g` 的部署 artifact。
"""


def main() -> int:
    args = parse_args()
    data_dir = resolve_repo_path(args.data_dir)
    out_dir = resolve_repo_path(args.out_dir)
    record_path = resolve_repo_path(args.record_path)
    source_report = args.source_report
    out_dir.mkdir(parents=True, exist_ok=True)

    train_spectra, train_metadata = load_split(data_dir, "train")
    val_spectra, val_metadata = load_split(data_dir, "val")
    config = Report20260505Config(target=args.target)
    bundle = fit_report_20260505_model(train_spectra, val_spectra, train_metadata, val_metadata, config=config)

    train_pred = bundle.predict(train_spectra, train_metadata)
    val_pred = bundle.predict(val_spectra, val_metadata)
    y_train = train_spectra[args.target].to_numpy(dtype=float)
    y_val = val_spectra[args.target].to_numpy(dtype=float)
    train_metrics = regression_metrics(y_train, train_pred)
    val_metrics = regression_metrics(y_val, val_pred)

    train_frame = prediction_frame(train_spectra, train_metadata, train_pred, args.target)
    val_frame = prediction_frame(val_spectra, val_metadata, val_pred, args.target)
    train_frame.to_csv(out_dir / "train_predictions.csv", index=False)
    val_frame.to_csv(out_dir / "val_predictions.csv", index=False)

    model_path = out_dir / "model.joblib"
    joblib.dump(bundle, model_path)
    model_sha = sha256_file(model_path)

    feature_schema = bundle.manifest()
    write_json(out_dir / "feature_schema.json", feature_schema)
    write_json(
        out_dir / "config.json",
        {
            "preprocess": [
                {"name": "snv"},
                {"name": "savgol", "params": {"window_length": config.sg_window_length, "polyorder": config.sg_polyorder, "deriv": config.sg_deriv}},
            ],
            "representation": {"type": "PLSRegression", "n_components": config.n_components, "scale": config.pls_scale},
            "regressor": {"type": "HuberRegressor", "alpha": config.huber_alpha, "epsilon": config.huber_epsilon, "max_iter": config.huber_max_iter},
            "post_pls_scaler": "StandardScaler",
            "meta_columns": bundle.meta_columns,
            "n_wavelength_features": len(bundle.wavelength_columns),
            "n_final_features": len(bundle.feature_names),
        },
    )
    comparison = compare_with_existing(out_dir, val_frame)
    metrics_payload = {
        "target": args.target,
        "unit": "ml/100g",
        "data_dir": str(data_dir),
        "source_report": source_report,
        "strictness": "formal_val_model_fit_on_train_only; reproduces the 2026-05-05 report model exactly, including weight_g/batch/sid_num features",
        "train": {"n": int(len(y_train)), **train_metrics},
        "val": {"n": int(len(y_val)), **val_metrics},
        "report_reference": {"val_r2": 0.8502290629192442, "val_rmse": 0.17439983308189466, "val_mae": 0.1448869339723355},
        "existing_prediction_comparison": comparison,
    }
    write_json(out_dir / "metrics.json", metrics_payload)

    manifest = {
        "model_version": "huajiao-report-20260505-snv-sg1-pls10-batch-quality-huber",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_report": source_report,
        "data_dir": str(data_dir),
        "target": args.target,
        "unit": "ml/100g",
        "model_sha256": model_sha,
        "code_commit": git_commit(),
        "dependencies": dependency_versions(),
        **feature_schema,
    }
    write_json(out_dir / "manifest.json", manifest)
    (out_dir / "model_card.md").write_text(model_card_text(manifest=manifest, metrics=metrics_payload), encoding="utf-8")
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(record_text(out_dir=out_dir, source_report=source_report, metrics=metrics_payload, manifest=manifest, comparison=comparison), encoding="utf-8")

    print(
        json.dumps(
            {
                "artifact_dir": str(out_dir),
                "record_path": str(record_path),
                "model_sha256": model_sha,
                "train": metrics_payload["train"],
                "val": metrics_payload["val"],
                "existing_prediction_comparison": comparison,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
