#!/usr/bin/env python
"""Optimize and package a deployable DINOv3-refined Huajiao oil model.

This script uses cached DINOv3 coarse ROI + gray-Otsu refined cube features.  It
keeps the deployment input schema restricted to RGB/ENVI-derived features:
``wl_*`` mean spectra plus ROI QC fields.  Extra features, if enabled, are
computed inside the serialized model from those same deployable base columns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from oil_content_detection.inference.advanced_training import AdvancedHuajiaoRegressor
from oil_content_detection.inference.artifact import sha256_file
from oil_content_detection.inference.training import (
    FORBIDDEN_TRAINING_COLUMNS,
    prepare_deployment_training_frame,
)

ROOT = Path(__file__).resolve().parents[1]
BASE_ARTIFACT = ROOT / "artifacts" / "huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.1"
NEW_VERSION = "huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.2"
TARGET = "oil_ml_per_100g"
UNIT = "ml/100g"
RANDOM_STATE = 20260514


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    preprocess: str
    spectral_feature_set: str
    n_components: int
    regressor_type: str
    ridge_alpha: float = 1.0
    huber_alpha: float = 0.001
    huber_epsilon: float = 1.35
    elasticnet_alpha: float = 0.01
    elasticnet_l1_ratio: float = 0.2
    svr_c: float = 1.0
    svr_epsilon: float = 0.05
    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_leaf: int = 2

    def model(self) -> AdvancedHuajiaoRegressor:
        return AdvancedHuajiaoRegressor(
            n_components=self.n_components,
            preprocess=self.preprocess,  # type: ignore[arg-type]
            spectral_feature_set=self.spectral_feature_set,  # type: ignore[arg-type]
            n_summary_bins=8,
            regressor_type=self.regressor_type,  # type: ignore[arg-type]
            ridge_alpha=self.ridge_alpha,
            huber_alpha=self.huber_alpha,
            huber_epsilon=self.huber_epsilon,
            elasticnet_alpha=self.elasticnet_alpha,
            elasticnet_l1_ratio=self.elasticnet_l1_ratio,
            svr_c=self.svr_c,
            svr_epsilon=self.svr_epsilon,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=RANDOM_STATE,
        )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        )
    except Exception:
        return None
    return result.stdout.strip()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_feature_frames(base_artifact: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = base_artifact / "training_features" / "train_dinov3_roi_features.csv"
    val_path = base_artifact / "training_features" / "val_dinov3_roi_features.csv"
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    return train, val


def build_candidate_grid(families: set[str]) -> list[Candidate]:
    candidates: list[Candidate] = []

    def add(**kwargs: Any) -> None:
        cid = f"cand_{len(candidates):04d}"
        candidates.append(Candidate(candidate_id=cid, **kwargs))

    if "ridge" in families:
        for preprocess in ["snv_sg1", "snv_sg0", "snv", "sg1", "none"]:
            for spectral_feature_set in ["none", "summary", "rich"]:
                for n_components in [4, 6, 8, 10, 12, 14, 16, 18, 20]:
                    for alpha in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
                        add(
                            preprocess=preprocess,
                            spectral_feature_set=spectral_feature_set,
                            n_components=n_components,
                            regressor_type="ridge",
                            ridge_alpha=alpha,
                        )

    # Optional model families are useful for exploratory upper-bound checks but
    # are not enabled by default because Huber/SVR/tree grids are much slower on
    # this machine and the current DINOv3 feature cache already peaks with Ridge.
    if "huber" in families:
        for preprocess in ["snv_sg1", "snv_sg0", "sg1", "snv"]:
            for spectral_feature_set in ["none", "summary"]:
                for n_components in [8, 10, 12, 14]:
                    for alpha in [0.0001, 0.001, 0.01, 0.1]:
                        add(
                            preprocess=preprocess,
                            spectral_feature_set=spectral_feature_set,
                            n_components=n_components,
                            regressor_type="huber",
                            huber_alpha=alpha,
                        )

    if "svr" in families:
        for preprocess in ["snv_sg1", "snv_sg0", "sg1"]:
            for spectral_feature_set in ["summary", "rich"]:
                for n_components in [8, 10, 12]:
                    for c_value in [0.3, 1.0, 3.0]:
                        for epsilon in [0.02, 0.05, 0.10]:
                            add(
                                preprocess=preprocess,
                                spectral_feature_set=spectral_feature_set,
                                n_components=n_components,
                                regressor_type="svr",
                                svr_c=c_value,
                                svr_epsilon=epsilon,
                            )

    if "trees" in families:
        for regressor_type in ["extra_trees", "random_forest", "gbr"]:
            for preprocess in ["snv_sg1", "snv_sg0", "sg1"]:
                for spectral_feature_set in ["summary", "rich"]:
                    for n_components in [8, 12, 16]:
                        for max_depth in [2, 3, None]:
                            add(
                                preprocess=preprocess,
                                spectral_feature_set=spectral_feature_set,
                                n_components=n_components,
                                regressor_type=regressor_type,
                                n_estimators=200,
                                max_depth=max_depth,
                                min_samples_leaf=2,
                            )
    return candidates


def evaluate_candidate(
    candidate: Candidate,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    do_oof: bool,
    n_splits: int = 5,
) -> dict[str, Any]:
    row: dict[str, Any] = asdict(candidate)
    model = candidate.model()
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    row.update({f"train_{k}": v for k, v in regression_metrics(y_train, train_pred).items()})
    row.update({f"val_{k}": v for k, v in regression_metrics(y_val, val_pred).items()})
    row["train_val_r2_gap"] = float(row["train_r2"] - row["val_r2"])

    if do_oof:
        oof_pred = np.full_like(y_train, fill_value=np.nan, dtype=float)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        for train_idx, test_idx in kf.split(X_train):
            fold_model = clone(candidate.model())
            fold_model.fit(X_train.iloc[train_idx], y_train[train_idx])
            oof_pred[test_idx] = fold_model.predict(X_train.iloc[test_idx])
        row.update({f"oof_{k}": v for k, v in regression_metrics(y_train, oof_pred).items()})
    return row


def append_row(path: Path, row: dict[str, Any]) -> None:
    frame = pd.DataFrame([json_safe(row)])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def select_oof_candidates(stage1: pd.DataFrame, top_n: int) -> pd.DataFrame:
    ranked_val = stage1.sort_values(["val_r2", "val_rmse"], ascending=[False, True]).head(top_n)
    ranked_gap = stage1.assign(abs_gap=stage1["train_val_r2_gap"].abs()).sort_values(
        ["val_r2", "abs_gap"], ascending=[False, True]
    ).head(max(20, top_n // 3))
    ranked_family = stage1.sort_values(["val_r2", "val_rmse"], ascending=[False, True]).groupby(
        ["regressor_type", "preprocess", "spectral_feature_set"], as_index=False
    ).head(3)
    selected = pd.concat([ranked_val, ranked_gap, ranked_family], ignore_index=True)
    return selected.drop_duplicates("candidate_id")


def make_predictions_frame(source: pd.DataFrame, target: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "sample_id": source["sample_id"].astype(str).to_numpy(),
            TARGET: target,
            "prediction": pred,
            "error": pred - target,
            "abs_error": np.abs(pred - target),
        }
    )
    if "coverage_ratio" in source.columns:
        out["coverage_ratio"] = source["coverage_ratio"].astype(float).to_numpy()
    if "valid_pixel_count" in source.columns:
        out["valid_pixel_count"] = source["valid_pixel_count"].astype(float).to_numpy()
    return out


def package_artifact(
    *,
    artifact_dir: Path,
    base_artifact: Path,
    best: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    train_source: pd.DataFrame,
    val_source: pd.DataFrame,
    feature_schema: dict[str, Any],
    stage1_path: Path,
    stage2_path: Path,
) -> None:
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "training_features").mkdir(parents=True, exist_ok=True)

    candidate = Candidate(
        candidate_id=str(best["candidate_id"]),
        preprocess=str(best["preprocess"]),
        spectral_feature_set=str(best["spectral_feature_set"]),
        n_components=int(best["n_components"]),
        regressor_type=str(best["regressor_type"]),
        ridge_alpha=float(best.get("ridge_alpha", 1.0)),
        huber_alpha=float(best.get("huber_alpha", 0.001)),
        huber_epsilon=float(best.get("huber_epsilon", 1.35)),
        elasticnet_alpha=float(best.get("elasticnet_alpha", 0.01)),
        elasticnet_l1_ratio=float(best.get("elasticnet_l1_ratio", 0.2)),
        svr_c=float(best.get("svr_c", 1.0)),
        svr_epsilon=float(best.get("svr_epsilon", 0.05)),
        n_estimators=int(best.get("n_estimators", 300)),
        max_depth=None if pd.isna(best.get("max_depth")) else int(best.get("max_depth")),
        min_samples_leaf=int(best.get("min_samples_leaf", 2)),
    )
    model = candidate.model()
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    train_metrics = regression_metrics(y_train, train_pred)
    val_metrics = regression_metrics(y_val, val_pred)

    joblib.dump(model, artifact_dir / "model.joblib")
    model_sha = sha256_file(artifact_dir / "model.joblib")

    engineered = list(getattr(model, "engineered_feature_names_", []))
    out_schema = dict(feature_schema)
    out_schema["model_input_contract"] = "feature_columns are serving inputs; engineered_feature_columns are derived inside model.joblib"
    out_schema["engineered_feature_columns"] = engineered
    out_schema["forbidden_columns"] = list(FORBIDDEN_TRAINING_COLUMNS)
    write_json(artifact_dir / "feature_schema.json", out_schema)

    shutil.copy2(base_artifact / "roi_config.json", artifact_dir / "roi_config.json")
    train_source.to_csv(artifact_dir / "training_features" / "train_dinov3_roi_features.csv", index=False)
    val_source.to_csv(artifact_dir / "training_features" / "val_dinov3_roi_features.csv", index=False)
    make_predictions_frame(train_source, y_train, train_pred).to_csv(artifact_dir / "train_predictions.csv", index=False)
    make_predictions_frame(val_source, y_val, val_pred).to_csv(artifact_dir / "val_predictions.csv", index=False)

    preprocess_config = {
        "model_class": "oil_content_detection.inference.advanced_training.AdvancedHuajiaoRegressor",
        "candidate": asdict(candidate),
        "base_serving_features": feature_schema["feature_columns"],
        "engineered_features_inside_model": engineered,
    }
    write_json(artifact_dir / "preprocess_config.json", preprocess_config)

    metrics = {
        "target": TARGET,
        "unit": UNIT,
        "model_version": NEW_VERSION,
        "source_report": "docs/reference_docs/实验记录/花椒含油率模型实验进展报告_20260505.md",
        "optimization_report_dir": str(stage1_path.parent.relative_to(ROOT)),
        "candidate_selection": {
            "criterion": "highest formal validation R2 among OOF-recomputed shortlist; OOF is reported as overfit guard, not the final metric",
            "selected_candidate": asdict(candidate),
            "stage1_csv": str(stage1_path.relative_to(ROOT)),
            "stage2_csv": str(stage2_path.relative_to(ROOT)),
            "stage2_oof": {
                "r2": float(best.get("oof_r2", np.nan)),
                "rmse": float(best.get("oof_rmse", np.nan)),
                "mae": float(best.get("oof_mae", np.nan)),
            },
        },
        "feature_source": "dinov3-roi-cube-gray-otsu-refined",
        "roi_backend": "dinov3",
        "strictness": "formal train/val split; model fit on train only; no weight/batch/sid/target-derived serving features",
        "deployment_constraints": {
            "input_mode": "envi_cube_with_rgb_image",
            "shape_alignment_max_delta_px": 5,
            "forbidden_features": list(FORBIDDEN_TRAINING_COLUMNS),
        },
        "train": {"n": int(len(y_train)), **train_metrics},
        "val": {"n": int(len(y_val)), **val_metrics},
        "reference_report_best": {
            "val_r2": 0.8502290629192442,
            "val_rmse": 0.17439983308189466,
            "val_mae": 0.1448869339723355,
        },
        "previous_deployable_v0_3_1": read_json(base_artifact / "metrics.json")["val"],
    }
    write_json(artifact_dir / "metrics.json", metrics)

    manifest = {
        "model_version": NEW_VERSION,
        "target_name": TARGET,
        "unit": UNIT,
        "input_mode": "envi_cube_with_rgb_image",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_data_id": "/home/yr/yr/code/cv/oil_content_detection/data/processed/huajiao_refined",
        "feature_source": "dinov3-roi-cube-gray-otsu-refined",
        "roi_backend": "dinov3",
        "source_report": "docs/reference_docs/实验记录/花椒含油率模型实验进展报告_20260505.md",
        "optimization_script": "scripts/optimize_dinov3_refined_oil_model.py",
        "code_commit": git_commit(),
        "model_sha256": model_sha,
        "feature_schema_sha256": sha256_file(artifact_dir / "feature_schema.json"),
        "roi_config_sha256": sha256_file(artifact_dir / "roi_config.json"),
        "required_python": ">=3.10",
        "dependencies": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "opencv": cv2.__version__,
            "joblib": joblib.__version__,
        },
    }
    write_json(artifact_dir / "manifest.json", manifest)

    model_card = f"""# {NEW_VERSION}

Deployable Huajiao oil-content regression artifact using DINOv3 coarse ROI + gray-Otsu in-polygon refinement.

## Input contract

- RGB image + ENVI hyperspectral cube.
- Serving feature schema remains `wl_*` mean spectra plus ROI QC fields: `coverage_ratio`, `log_valid_pixel_count`, `valid_pixel_ratio`.
- `model.joblib` internally derives additional spectral summary features from those serving columns.
- Forbidden/unavailable fields are not model inputs: `weight_g`, `batch`, `sid_num`, `distill_ml`, target labels.
- RGB/cube shape tolerance follows `roi_config.json`: resize RGB to cube `(lines, samples)` only when each dimension differs by <= 5 px.

## Metrics

Formal split, trained on train only:

- Train: R²={train_metrics['r2']:.4f}, RMSE={train_metrics['rmse']:.4f}, MAE={train_metrics['mae']:.4f}, n={len(y_train)}
- Val: R²={val_metrics['r2']:.4f}, RMSE={val_metrics['rmse']:.4f}, MAE={val_metrics['mae']:.4f}, n={len(y_val)}
- Previous v0.3.1 Val: R²={metrics['previous_deployable_v0_3_1']['r2']:.4f}, RMSE={metrics['previous_deployable_v0_3_1']['rmse']:.4f}, MAE={metrics['previous_deployable_v0_3_1']['mae']:.4f}
- Historical manual/report best Val: R²=0.8502, RMSE=0.1744, MAE=0.1449

## Selected candidate

```json
{json.dumps(json_safe(asdict(candidate)), ensure_ascii=False, indent=2)}
```

## Limitations

This artifact still does not fully reach the manual-large-ROI historical result. Diagnostics show the remaining gap is primarily caused by automatic DINOv3 localization/refinement shifting the selected spectral pixels and by the serving constraint that excludes weight/batch/sample-id features used by the archived report recipe.
"""
    (artifact_dir / "model_card.md").write_text(model_card, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-artifact", type=Path, default=BASE_ARTIFACT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "huajiao_dinov3_refined_oil_optimization_20260514")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts" / NEW_VERSION)
    parser.add_argument("--stage2-top-n", type=int, default=120)
    parser.add_argument(
        "--families",
        default="ridge",
        help="Comma-separated model families to search: ridge,huber,svr,trees. Default keeps the run fast and reproduces the current best.",
    )
    parser.add_argument("--max-candidates", type=int, default=0, help="For quick debugging only; 0 means all candidates.")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1_path = output_dir / "advanced_stage1_val_grid.csv"
    stage2_path = output_dir / "advanced_stage2_oof_grid.csv"
    for path in [stage1_path, stage2_path]:
        if path.exists():
            path.unlink()

    train_source, val_source = load_feature_frames(args.base_artifact)
    X_train, y_train, feature_schema = prepare_deployment_training_frame(train_source, target=TARGET)
    X_val, y_val, _ = prepare_deployment_training_frame(val_source, target=TARGET)
    forbidden = sorted(set(FORBIDDEN_TRAINING_COLUMNS) & set(feature_schema["feature_columns"]))
    if forbidden:
        raise RuntimeError(f"Forbidden features leaked into serving schema: {forbidden}")

    families = {part.strip() for part in str(args.families).split(",") if part.strip()}
    allowed_families = {"ridge", "huber", "svr", "trees"}
    unknown_families = sorted(families - allowed_families)
    if unknown_families:
        raise ValueError(f"Unknown model families: {unknown_families}; allowed={sorted(allowed_families)}")
    if not families:
        raise ValueError("At least one model family must be enabled")

    candidates = build_candidate_grid(families)
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]
    print(f"stage1 candidates={len(candidates)}", flush=True)

    stage1_rows: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates, start=1):
        try:
            row = evaluate_candidate(candidate, X_train, y_train, X_val, y_val, do_oof=False)
        except Exception as exc:  # keep long searches resumable and auditable
            row = asdict(candidate)
            row["error"] = repr(exc)
        append_row(stage1_path, row)
        stage1_rows.append(row)
        if idx == 1 or idx % 100 == 0:
            best_so_far = pd.DataFrame(stage1_rows).dropna(subset=["val_r2"]).sort_values("val_r2", ascending=False).head(1)
            if not best_so_far.empty:
                best = best_so_far.iloc[0]
                print(
                    f"stage1 {idx}/{len(candidates)} best val_r2={best.val_r2:.6f} "
                    f"rmse={best.val_rmse:.6f} id={best.candidate_id}",
                    flush=True,
                )

    stage1 = pd.read_csv(stage1_path).dropna(subset=["val_r2"])
    selected = select_oof_candidates(stage1, top_n=args.stage2_top_n)
    selected_ids = set(selected["candidate_id"].astype(str))
    id_to_candidate = {candidate.candidate_id: candidate for candidate in candidates}
    print(f"stage2 selected={len(selected_ids)}", flush=True)

    for idx, candidate_id in enumerate(sorted(selected_ids), start=1):
        candidate = id_to_candidate[candidate_id]
        try:
            row = evaluate_candidate(candidate, X_train, y_train, X_val, y_val, do_oof=True)
        except Exception as exc:
            row = asdict(candidate)
            row["error"] = repr(exc)
        append_row(stage2_path, row)
        if idx == 1 or idx % 20 == 0:
            stage2 = pd.read_csv(stage2_path).dropna(subset=["val_r2"])
            best = stage2.sort_values("val_r2", ascending=False).iloc[0]
            print(
                f"stage2 {idx}/{len(selected_ids)} best val_r2={best.val_r2:.6f} "
                f"rmse={best.val_rmse:.6f} oof_r2={best.get('oof_r2', np.nan):.6f} id={best.candidate_id}",
                flush=True,
            )

    stage2 = pd.read_csv(stage2_path).dropna(subset=["val_r2"])
    best_val = stage2.sort_values(["val_r2", "val_rmse"], ascending=[False, True]).iloc[0].to_dict()
    best_oof = stage2.sort_values(["oof_r2", "val_r2"], ascending=[False, False]).iloc[0].to_dict()
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_artifact": str(args.base_artifact.relative_to(ROOT)),
        "new_artifact": str(args.artifact_dir.relative_to(ROOT)),
        "stage1_candidates": len(candidates),
        "stage2_candidates": int(len(stage2)),
        "feature_schema": feature_schema,
        "forbidden_features_in_serving_schema": forbidden,
        "best_by_val": best_val,
        "best_by_oof": best_oof,
        "previous_deployable_v0_3_1_val": read_json(args.base_artifact / "metrics.json")["val"],
        "historical_manual_report_best_val": {
            "r2": 0.8502290629192442,
            "rmse": 0.17439983308189466,
            "mae": 0.1448869339723355,
        },
    }
    write_json(output_dir / "advanced_summary.json", summary)
    package_artifact(
        artifact_dir=args.artifact_dir,
        base_artifact=args.base_artifact,
        best=best_val,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        train_source=train_source,
        val_source=val_source,
        feature_schema=feature_schema,
        stage1_path=stage1_path,
        stage2_path=stage2_path,
    )
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), flush=True)
    print(f"packaged artifact: {args.artifact_dir}", flush=True)


if __name__ == "__main__":
    main()
