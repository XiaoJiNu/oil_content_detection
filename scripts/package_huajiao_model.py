#!/usr/bin/env python3
"""Package the deployable Huajiao oil-content model artifact.

This script packages the 2026-05-05 report-derived serving model family:
SNV + SG first derivative + PLS scores + ROI quality features + a compact
linear regressor. It intentionally excludes weight, batch, sample-id, and
target-derived fields.

For the default mode-B artifact, features are re-extracted from each ENVI cube
with the same traditional ROI flow used at inference time. This avoids training
on manual-ROI spectra while serving on automatic-ROI spectra.
"""
from __future__ import annotations

import argparse
import json
import os
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

from oil_content_detection.acquisition.envi_reader import load_envi_cube  # noqa: E402
from oil_content_detection.inference.artifact import sha256_file  # noqa: E402
from oil_content_detection.inference.features import (  # noqa: E402
    extract_roi_spectrum_features,
)
from oil_content_detection.inference.roi import (  # noqa: E402
    load_rgb_image_bgr,
    refine_mask_in_roi_if_configured,
    segment_image_roi,
)
from oil_content_detection.inference.training import (  # noqa: E402
    build_deployment_pipeline,
    prepare_deployment_training_frame,
    regression_metrics,
    wavelength_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and package Huajiao oil-content serving artifact."
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed/huajiao_refined",
        help="Processed data directory with train/val parquet files.",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/huajiao-oil-content-v0.1.0",
        help="Artifact output directory.",
    )
    parser.add_argument(
        "--model-version",
        default="huajiao-oil-content-v0.1.0",
        help="Model version string.",
    )
    parser.add_argument(
        "--target", default="oil_ml_per_100g", help="Target column name."
    )
    parser.add_argument(
        "--n-components", type=int, default=10, help="PLS component count."
    )
    parser.add_argument(
        "--regressor",
        choices=["huber", "ridge"],
        default="huber",
        help="Final compact linear regressor type.",
    )
    parser.add_argument(
        "--huber-alpha", type=float, default=0.001, help="HuberRegressor alpha."
    )
    parser.add_argument(
        "--huber-max-iter", type=int, default=2000, help="HuberRegressor max_iter."
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=10.0,
        help="Ridge alpha when --regressor=ridge.",
    )
    parser.add_argument(
        "--feature-source",
        choices=["auto-roi-cube", "processed"],
        default="auto-roi-cube",
        help="Use mode-B automatic ROI cube features (default) or existing processed spectra.",
    )
    parser.add_argument(
        "--roi-backend",
        choices=["traditional", "dinov3"],
        default="traditional",
        help="ROI backend for auto-roi-cube feature extraction and serving.",
    )
    parser.add_argument(
        "--dinov3-roi-artifact",
        default="artifacts/huajiao-dinov3-roi-seg-v0.1.0",
        help="DINOv3 ROI segmentation artifact directory; relative paths are resolved from repo root.",
    )
    parser.add_argument(
        "--roi-device",
        default="auto",
        help="Device for DINOv3 ROI inference during packaging/serving config, e.g. auto, cpu, cuda.",
    )
    parser.add_argument(
        "--dinov3-post-refine-mode",
        choices=["none", "gray_otsu", "color_texture"],
        default="none",
        help="Optional deterministic in-polygon refinement after DINOv3 coarse ROI.",
    )
    parser.add_argument(
        "--dinov3-refine-open-size",
        type=int,
        default=5,
        help="Morphological open kernel for DINOv3 post-refinement.",
    )
    parser.add_argument(
        "--dinov3-refine-close-size",
        type=int,
        default=5,
        help="Morphological close kernel for DINOv3 post-refinement.",
    )
    parser.add_argument(
        "--dinov3-refine-hole-fill-max-area",
        type=int,
        default=300,
        help="Maximum hole area filled during DINOv3 post-refinement.",
    )
    parser.add_argument(
        "--roi-segmentation-mode",
        choices=["color_texture", "gray_otsu", "hybrid"],
        default="hybrid",
        help="Traditional image-processing ROI strategy used for auto-roi-cube extraction.",
    )
    parser.add_argument(
        "--roi-target-coverage",
        type=float,
        default=0.17,
        help="Hybrid ROI target full-image coverage ratio.",
    )
    parser.add_argument(
        "--roi-min-coverage",
        type=float,
        default=0.05,
        help="Hybrid ROI minimum full-image coverage ratio.",
    )
    parser.add_argument(
        "--roi-max-coverage",
        type=float,
        default=0.35,
        help="Hybrid ROI maximum full-image coverage ratio.",
    )
    parser.add_argument(
        "--disable-roi-texture",
        action="store_true",
        help="Disable texture OR branch in color/hybrid ROI extraction.",
    )
    parser.add_argument(
        "--source-report",
        default="docs/reference_docs/实验记录/花椒含油率模型实验进展报告_20260505.md",
        help="Experiment report this artifact is derived from.",
    )
    return parser.parse_args()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return None


def _read_split(data_dir: Path, split: str) -> pd.DataFrame:
    path = data_dir / split / "huajiao_spectra.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {split} spectra parquet: {path}")
    return pd.read_parquet(path)


def _read_metadata(data_dir: Path, split: str) -> pd.DataFrame:
    path = data_dir / split / "huajiao_metadata.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {split} metadata parquet: {path}")
    return pd.read_parquet(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _artifact_path_for_config(path: str | Path, *, out_dir: Path) -> str:
    resolved = _resolve_repo_path(path)
    rel_path = os.path.relpath(resolved, out_dir.resolve())
    return Path(rel_path).as_posix()


def _feature_source_label(feature_source: str, roi_backend: str) -> str:
    if feature_source == "auto-roi-cube" and roi_backend == "dinov3":
        return "dinov3-roi-cube"
    return feature_source


def _build_roi_config(args: argparse.Namespace, *, out_dir: Path) -> dict[str, Any]:
    roi_config: dict[str, Any] = {
        "backend": args.roi_backend,
        "require_same_shape": True,
        "min_pepper_pixels": 100,
        "min_coverage_ratio": 0.0001,
        "max_coverage_ratio": 0.75,
        "segmentation": {
            "segmentation_mode": args.roi_segmentation_mode,
            "use_texture": not args.disable_roi_texture,
            "hybrid_target_coverage_ratio": args.roi_target_coverage,
            "hybrid_min_coverage_ratio": args.roi_min_coverage,
            "hybrid_max_coverage_ratio": args.roi_max_coverage,
        },
    }
    if args.roi_backend == "dinov3":
        roi_config["dinov3"] = {
            "artifact_dir": _artifact_path_for_config(
                args.dinov3_roi_artifact, out_dir=out_dir
            ),
            "device": args.roi_device,
            "mask_source": "dinov3_roi_segmentation",
        }
        post_refine_mode = getattr(args, "dinov3_post_refine_mode", "none")
        if post_refine_mode != "none":
            roi_config["dinov3"]["post_refinement"] = {
                "enabled": True,
                "segmentation_mode": post_refine_mode,
                "pepper_open_size": int(getattr(args, "dinov3_refine_open_size", 5)),
                "pepper_close_size": int(getattr(args, "dinov3_refine_close_size", 5)),
                "hole_fill_max_area": int(
                    getattr(args, "dinov3_refine_hole_fill_max_area", 300)
                ),
                "label_fallback_enabled": False,
                "label_max_area_ratio": 0.0,
            }
    return roi_config


def _load_dinov3_roi_predictor(artifact_dir: str | Path, device: str | None):
    from oil_content_detection.segmentation.predictor import DinoV3ROIPredictor

    resolved_artifact_dir = _resolve_repo_path(artifact_dir)
    resolved_device = None if device in {None, "", "auto"} else device
    return DinoV3ROIPredictor.from_artifact(
        resolved_artifact_dir, device=resolved_device
    )


def _dependency_versions() -> dict[str, str]:
    import cv2
    import scipy

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "opencv": cv2.__version__,
        "joblib": joblib.__version__,
    }


def _image_path_from_metadata(row: pd.Series) -> Path:
    sample_dir = Path(str(row["sample_dir"]))
    candidates = [
        sample_dir / f"{row['sample_id']}.png",
        sample_dir / f"{row.get('sample_id_raw', row['sample_id'])}.png",
        sample_dir / f"{row['sample_id']}.jpg",
        sample_dir / f"{row.get('sample_id_raw', row['sample_id'])}.jpg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No RGB image found for sample {row['sample_id']} under {sample_dir}"
    )


def _extract_auto_roi_features(
    *,
    spectra_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    split: str,
    wl_cols: list[str],
    feature_columns: list[str],
    target: str,
    roi_config: dict[str, Any],
    cache_path: Path,
    roi_backend: str,
    dinov3_predictor: Any | None = None,
) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)

    metadata_by_id = metadata_df.set_index("sample_id", drop=False)
    rows: list[dict[str, Any]] = []
    for idx, sample in spectra_df.reset_index(drop=True).iterrows():
        sample_id = str(sample["sample_id"])
        if sample_id not in metadata_by_id.index:
            raise KeyError(f"Metadata missing for sample_id={sample_id}")
        meta = metadata_by_id.loc[sample_id]
        if isinstance(meta, pd.DataFrame):
            meta = meta.iloc[0]
        image_path = _image_path_from_metadata(meta)
        hdr_path = Path(str(meta["hdr_path"]))
        image_bgr = load_rgb_image_bgr(image_path)
        cube, _ = load_envi_cube(hdr_path, memmap=True)
        if roi_backend == "dinov3":
            if dinov3_predictor is None:
                raise ValueError(
                    "dinov3_predictor is required when roi_backend='dinov3'"
                )
            roi_prediction = dinov3_predictor.predict_image(
                image_path=image_path, sample_id=sample_id
            )
            base_mask = np.asarray(roi_prediction.mask, dtype=bool)
            roi_info = {
                "backend": "dinov3",
                "segmentation_strategy": "dinov3_roi_segmentation",
                "dinov3_threshold": float(getattr(roi_prediction, "threshold", np.nan)),
                "dinov3_coverage_ratio": float(
                    getattr(roi_prediction, "coverage_ratio", np.nan)
                ),
            }
            mask, refinement_info, mask_source = refine_mask_in_roi_if_configured(
                image_bgr=image_bgr,
                base_mask=base_mask,
                roi_config=roi_config,
                base_mask_source="dinov3_roi_segmentation",
            )
            roi_info.update(refinement_info)
        else:
            roi_result, _, _ = segment_image_roi(
                image_bgr=image_bgr,
                image_stem=sample_id,
                roi_config=roi_config,
                debug_dir=None,
            )
            mask = roi_result.pepper_mask
            roi_info = roi_result.info
            mask_source = "traditional_roi_segmentation"
        feature_row, roi_metrics = extract_roi_spectrum_features(
            cube=np.asarray(cube),
            mask=mask,
            wavelength_columns=wl_cols,
            feature_columns=feature_columns,
            mask_source=mask_source,
            roi_info=roi_info,
        )
        out = {
            "sample_id": sample_id,
            "sample_id_raw": sample.get("sample_id_raw", sample_id),
            target: float(sample[target]),
            "pixel_count": int(roi_metrics.pixel_count),
            "valid_pixel_count": int(roi_metrics.valid_pixel_count),
            "coverage_ratio": float(roi_metrics.coverage_ratio),
            "auto_roi_strategy": str(
                roi_info.get("segmentation_strategy", roi_backend)
            ),
        }
        for info_key, info_value in roi_info.items():
            if isinstance(
                info_value, (int, float, np.integer, np.floating)
            ) and np.isfinite(float(info_value)):
                out[f"roi_{info_key}"] = float(info_value)
        for col in wl_cols:
            out[col] = float(feature_row[col])
        rows.append(out)
        print(
            f"[{split}] extracted {idx + 1}/{len(spectra_df)} {sample_id}", flush=True
        )

    frame = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    return frame


def _model_card_text(
    *,
    model_version: str,
    source_report: str,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    feature_schema: dict[str, Any],
    data_dir: str,
    feature_source: str,
    roi_backend: str,
) -> str:
    roi_description = (
        "DINOv3 ROI 分割模型" if roi_backend == "dinov3" else "传统图像处理 ROI"
    )
    return f"""# {model_version} Model Card

## 用途

花椒含油率展示/工程调用模型。输入为同视角、同尺寸 RGB 图像与 ENVI 高光谱 cube，系统自动用{roi_description}提取花椒区域，聚合 ROI 内平均光谱，并输出 `{feature_schema.get('target_name', 'oil_ml_per_100g')}` 预测值。

## 来源实验

- 来源报告：`{source_report}`
- 报告推荐路线：`SNV + SG一阶导(window=13, polyorder=2) + PLSR(10) + HuberRegressor`
- 本 artifact 的部署约束：不使用 `weight_g`、`batch`、`sid_num`；只使用上传文件可直接得到的波段均值和 ROI 质量特征。
- 本 artifact 的特征来源：`{feature_source}`；ROI 后端：`{roi_backend}`。默认发布版使用自动 ROI 从 cube 重新提取训练/验证特征，使训练口径与展示推理口径一致。

## 数据与特征

- 训练/验证数据：`{data_dir}`
- 波段特征数：{len(feature_schema['wavelength_columns'])}
- ROI 质量特征：{', '.join(feature_schema['quality_columns'])}
- 禁止字段：{', '.join(feature_schema['forbidden_columns'])}

## 指标

| Split | R² | RMSE | MAE |
| --- | ---: | ---: | ---: |
| Train | {train_metrics['r2']:.6f} | {train_metrics['rmse']:.6f} | {train_metrics['mae']:.6f} |
| Val | {val_metrics['r2']:.6f} | {val_metrics['rmse']:.6f} | {val_metrics['mae']:.6f} |

注意：该指标是当前正式 Train/Val 划分下的工程指标。本 artifact 为了满足部署约束，去掉了报告最优模型中的重量、批次和样本编号相关特征，因此指标可能低于报告中的 `Val R²=0.8502` 最优记录。

## 输入契约

1. `image_path`: RGB/BGR 图像文件，OpenCV 可读取。
2. `hdr_path`: ENVI `.hdr` 文件，旁边必须有同名 `.dat`。
3. RGB 图像高宽必须与 cube 的 `(lines, samples)` 完全一致；不做自动配准和 resize。

## 输出

- `prediction`: 含油率预测值。
- `unit`: `ml/100g`。
- `roi`: ROI 像素数、覆盖率、有效光谱像素数、可选 mask/overlay 路径。
- `qc`: 状态和 warning。

## 风险与限制

- ROI 分割依赖当前拍摄背景/红纸/标签风格；换背景或光照后必须先做 ROI overlay 验收。
- 当前数据量较小，不代表严格跨产地、跨批次泛化能力。
- 不应把该模型用于超出当前花椒样本、设备格式和拍摄流程的生产决策，除非补充验证。
"""


def main() -> int:
    args = parse_args()
    data_dir = (
        (REPO_ROOT / args.data_dir).resolve()
        if not Path(args.data_dir).is_absolute()
        else Path(args.data_dir)
    )
    out_dir = (
        (REPO_ROOT / args.out_dir).resolve()
        if not Path(args.out_dir).is_absolute()
        else Path(args.out_dir)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df_raw = _read_split(data_dir, "train")
    val_df_raw = _read_split(data_dir, "val")
    wl_cols = wavelength_columns(train_df_raw)
    feature_columns = wl_cols + [
        "coverage_ratio",
        "log_valid_pixel_count",
        "valid_pixel_ratio",
    ]
    roi_config = _build_roi_config(args, out_dir=out_dir)
    feature_source_label = _feature_source_label(args.feature_source, args.roi_backend)
    dinov3_predictor = None
    if args.feature_source == "auto-roi-cube" and args.roi_backend == "dinov3":
        dinov3_predictor = _load_dinov3_roi_predictor(
            args.dinov3_roi_artifact, args.roi_device
        )

    if args.feature_source == "auto-roi-cube":
        train_md = _read_metadata(data_dir, "train")
        val_md = _read_metadata(data_dir, "val")
        feature_cache_dir = out_dir / "training_features"
        train_df = _extract_auto_roi_features(
            spectra_df=train_df_raw,
            metadata_df=train_md,
            split="train",
            wl_cols=wl_cols,
            feature_columns=feature_columns,
            target=args.target,
            roi_config=roi_config,
            cache_path=feature_cache_dir / f"train_{args.roi_backend}_roi_features.csv",
            roi_backend=args.roi_backend,
            dinov3_predictor=dinov3_predictor,
        )
        val_df = _extract_auto_roi_features(
            spectra_df=val_df_raw,
            metadata_df=val_md,
            split="val",
            wl_cols=wl_cols,
            feature_columns=feature_columns,
            target=args.target,
            roi_config=roi_config,
            cache_path=feature_cache_dir / f"val_{args.roi_backend}_roi_features.csv",
            roi_backend=args.roi_backend,
            dinov3_predictor=dinov3_predictor,
        )
    else:
        train_df = train_df_raw
        val_df = val_df_raw

    X_train, y_train, feature_schema = prepare_deployment_training_frame(
        train_df, target=args.target
    )
    X_val, y_val, val_schema = prepare_deployment_training_frame(
        val_df, target=args.target
    )
    if feature_schema["feature_columns"] != val_schema["feature_columns"]:
        raise ValueError("Train and val feature schemas differ")

    model = build_deployment_pipeline(
        wavelength_columns=feature_schema["wavelength_columns"],
        quality_columns=feature_schema["quality_columns"],
        n_components=args.n_components,
        regressor_type=args.regressor,
        huber_alpha=args.huber_alpha,
        huber_max_iter=args.huber_max_iter,
        ridge_alpha=args.ridge_alpha,
    )
    model.fit(X_train, y_train)
    train_pred = np.asarray(model.predict(X_train), dtype=float).reshape(-1)
    val_pred = np.asarray(model.predict(X_val), dtype=float).reshape(-1)
    train_metrics = regression_metrics(y_train, train_pred)
    val_metrics = regression_metrics(y_val, val_pred)

    model_path = out_dir / "model.joblib"
    joblib.dump(model, model_path)
    model_sha = sha256_file(model_path)

    feature_schema = dict(feature_schema)
    feature_schema.update(
        {
            "target_name": args.target,
            "description": "Deployable mode-B schema: wl_* mean spectrum + ROI quality features only; no weight, batch, or sid fields.",
        }
    )
    _write_json(out_dir / "feature_schema.json", feature_schema)
    feature_sha = sha256_file(out_dir / "feature_schema.json")

    preprocess_config = {
        "preprocess": [
            {"name": "snv"},
            {
                "name": "savgol",
                "params": {"window_length": 13, "polyorder": 2, "deriv": 1},
            },
        ],
        "representation": {
            "type": "PLSScoreTransformer",
            "n_components": args.n_components,
            "scale": False,
        },
        "regressor": (
            {
                "type": "HuberRegressor",
                "alpha": args.huber_alpha,
                "max_iter": args.huber_max_iter,
            }
            if args.regressor == "huber"
            else {"type": "Ridge", "alpha": args.ridge_alpha}
        ),
    }
    _write_json(out_dir / "preprocess_config.json", preprocess_config)
    _write_json(out_dir / "roi_config.json", roi_config)

    metrics_payload = {
        "target": args.target,
        "unit": "ml/100g",
        "source_report": args.source_report,
        "data_dir": str(data_dir),
        "feature_source": feature_source_label,
        "roi_backend": args.roi_backend,
        "strictness": "formal train/val split; model fit on train only; report-derived family repackaged without weight/batch/sid for mode-B serving",
        "deployment_constraints": {
            "input_mode": "envi_cube_with_rgb_image",
            "require_same_shape": True,
            "forbidden_features": feature_schema["forbidden_columns"],
        },
        "train": {"n": int(len(y_train)), **train_metrics},
        "val": {"n": int(len(y_val)), **val_metrics},
        "reference_report_best": {
            "val_r2": 0.8502290629192442,
            "val_rmse": 0.17439983308189466,
            "val_mae": 0.1448869339723355,
        },
    }
    _write_json(out_dir / "metrics.json", metrics_payload)

    manifest = {
        "model_version": args.model_version,
        "target_name": args.target,
        "unit": "ml/100g",
        "input_mode": "envi_cube_with_rgb_image",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_data_id": str(data_dir),
        "feature_source": feature_source_label,
        "roi_backend": args.roi_backend,
        "source_report": args.source_report,
        "code_commit": _git_commit(),
        "model_sha256": model_sha,
        "feature_schema_sha256": feature_sha,
        "required_python": ">=3.10",
        "dependencies": _dependency_versions(),
    }
    _write_json(out_dir / "manifest.json", manifest)

    train_out = train_df[["sample_id", args.target]].copy()
    train_out["prediction"] = train_pred
    train_out["error"] = train_out["prediction"] - train_out[args.target]
    train_out.to_csv(out_dir / "train_predictions.csv", index=False)
    val_out = val_df[["sample_id", args.target]].copy()
    val_out["prediction"] = val_pred
    val_out["error"] = val_out["prediction"] - val_out[args.target]
    val_out.to_csv(out_dir / "val_predictions.csv", index=False)

    (out_dir / "model_card.md").write_text(
        _model_card_text(
            model_version=args.model_version,
            source_report=args.source_report,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            feature_schema=feature_schema,
            data_dir=str(data_dir),
            feature_source=feature_source_label,
            roi_backend=args.roi_backend,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "artifact_dir": str(out_dir),
                "feature_source": feature_source_label,
                "roi_backend": args.roi_backend,
                "train": train_metrics,
                "val": val_metrics,
                "model_sha256": model_sha,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
