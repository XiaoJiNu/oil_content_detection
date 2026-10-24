#!/usr/bin/env python3
"""Quick exploratory model search for huajiao oil-content formal split."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, HuberRegressor, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))

from oil_content_detection.preprocessing import PreprocessStep, apply_preprocessing_pipeline  # noqa: E402

warnings.filterwarnings("ignore")


def metric_tuple(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    return (
        float(r2_score(y_true, y_pred)),
        float(np.sqrt(mean_squared_error(y_true, y_pred))),
        float(mean_absolute_error(y_true, y_pred)),
    )


def wavelength_cols(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if c.startswith("wl_") and c[3:].isdigit()], key=lambda c: int(c.split("_")[1]))


def add_meta(df: pd.DataFrame, md: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["batch"] = md["sample_dir"].astype(str).apply(lambda s: Path(s).parent.name).values
    out["sid_num"] = out["sample_id"].astype(str).str.extract(r"(\d+)").astype(int)
    out["log_valid_pixel_count"] = np.log1p(out["valid_pixel_count"])
    out["valid_pixel_ratio"] = out["valid_pixel_count"] / out["pixel_count"]
    out["weight_per_pixel"] = out["weight_g"] / out["valid_pixel_count"]
    out["log_weight"] = np.log(out["weight_g"])
    return out


def append_meta(A: np.ndarray, B: np.ndarray, tr: pd.DataFrame, va: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if not cols:
        return A, B
    dtr = pd.DataFrame(A, index=tr.index)
    dva = pd.DataFrame(B, index=va.index)
    for c in cols:
        if tr[c].dtype == "object":
            cats = sorted(set(tr[c].astype(str)) | set(va[c].astype(str)))
            for cat in cats:
                dtr[f"{c}_{cat}"] = (tr[c].astype(str) == cat).astype(float).values
                dva[f"{c}_{cat}"] = (va[c].astype(str) == cat).astype(float).values
        else:
            dtr[c] = tr[c].to_numpy(dtype=float)
            dva[c] = va[c].to_numpy(dtype=float)
    return dtr.to_numpy(dtype=float), dva.to_numpy(dtype=float)


def main() -> None:
    data_dir = repo_root / "data/processed/huajiao_refined"
    tr = pd.read_parquet(data_dir / "train/huajiao_spectra.parquet")
    va = pd.read_parquet(data_dir / "val/huajiao_spectra.parquet")
    mdtr = pd.read_parquet(data_dir / "train/huajiao_metadata.parquet")
    mdva = pd.read_parquet(data_dir / "val/huajiao_metadata.parquet")
    tr = add_meta(tr, mdtr)
    va = add_meta(va, mdva)
    ytr = tr["oil_ml_per_100g"].to_numpy(dtype=float)
    yva = va["oil_ml_per_100g"].to_numpy(dtype=float)
    fc = wavelength_cols(tr)

    means = tr.groupby("batch")["oil_ml_per_100g"].mean()
    meds = tr.groupby("batch")["oil_ml_per_100g"].median()
    counts = tr.groupby("batch")["oil_ml_per_100g"].count()
    global_mean = float(ytr.mean())
    for df in (tr, va):
        df["batch_mean_te"] = df["batch"].map(means).fillna(global_mean)
        df["batch_median_te"] = df["batch"].map(meds).fillna(global_mean)
        df["batch_count_te"] = df["batch"].map(counts).fillna(0)

    preps = {
        "raw": [],
        "sg": [PreprocessStep("sg", {"window_length": 13, "polyorder": 2, "deriv": 0})],
        "sg1": [PreprocessStep("sg", {"window_length": 13, "polyorder": 2, "deriv": 1})],
        "snv_sg1": [PreprocessStep("snv"), PreprocessStep("sg", {"window_length": 13, "polyorder": 2, "deriv": 1})],
    }
    meta_options = {
        "none": [],
        "te": ["batch_mean_te", "batch_median_te", "batch_count_te"],
        "te_quality": ["batch_mean_te", "batch_median_te", "batch_count_te", "coverage_ratio", "log_valid_pixel_count", "valid_pixel_ratio"],
        "te_quality_weight": ["batch_mean_te", "batch_median_te", "batch_count_te", "coverage_ratio", "log_valid_pixel_count", "valid_pixel_ratio", "weight_g", "log_weight", "weight_per_pixel"],
        "batch_quality": ["batch", "coverage_ratio", "log_valid_pixel_count", "valid_pixel_ratio", "weight_g", "sid_num"],
    }
    rows: list[tuple] = []
    for pname, steps in preps.items():
        Xtr = apply_preprocessing_pipeline(tr[fc].to_numpy(dtype=float), steps) if steps else tr[fc].to_numpy(dtype=float)
        Xva = apply_preprocessing_pipeline(va[fc].to_numpy(dtype=float), steps) if steps else va[fc].to_numpy(dtype=float)
        reps: list[tuple[str, np.ndarray, np.ndarray]] = [("spec", Xtr, Xva)]
        for n in [5, 8, 10, 15, 20, 30, 40]:
            if n < min(Xtr.shape):
                pls = PLSRegression(n_components=n, scale=False).fit(Xtr, ytr)
                reps.append((f"pls{n}", pls.transform(Xtr), pls.transform(Xva)))
                pca = PCA(n_components=n, random_state=0).fit(Xtr)
                reps.append((f"pca{n}", pca.transform(Xtr), pca.transform(Xva)))
        for rname, A0, B0 in reps:
            for mname, mcols in meta_options.items():
                A, B = append_meta(A0, B0, tr, va, mcols)
                alphas = np.logspace(-4, 4, 60)
                models = [
                    ("ridge", make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))),
                    ("bayes", make_pipeline(StandardScaler(), BayesianRidge())),
                    ("huber", make_pipeline(StandardScaler(), HuberRegressor(max_iter=1000, alpha=0.001))),
                ]
                if A.shape[1] <= 90:
                    models.extend(
                        [
                            ("svr_rbf3", make_pipeline(StandardScaler(), SVR(C=3, epsilon=0.05, gamma="scale"))),
                            ("svr_rbf10", make_pipeline(StandardScaler(), SVR(C=10, epsilon=0.05, gamma="scale"))),
                            ("svr_rbf30", make_pipeline(StandardScaler(), SVR(C=30, epsilon=0.05, gamma="scale"))),
                            ("krr", make_pipeline(StandardScaler(), KernelRidge(alpha=0.05, kernel="rbf", gamma=0.05))),
                            ("extra", ExtraTreesRegressor(n_estimators=400, random_state=2, min_samples_leaf=1, max_features=None)),
                            ("gbr", GradientBoostingRegressor(random_state=2, learning_rate=0.03, n_estimators=300, max_depth=2)),
                        ]
                    )
                for model_name, model in models:
                    try:
                        model.fit(A, ytr)
                        pred = np.asarray(model.predict(B)).ravel()
                        r2, rmse, mae = metric_tuple(yva, pred)
                        rows.append((r2, rmse, mae, pname, rname, mname, model_name, A.shape[1]))
                    except Exception as exc:
                        rows.append((-999.0, 999.0, 999.0, pname, rname, mname, model_name + f":ERR:{type(exc).__name__}", A.shape[1]))
    rows = sorted(rows, key=lambda x: x[0], reverse=True)
    out_dir = repo_root / "results/huajiao_formal_quick_search"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["val_r2", "val_rmse", "val_mae", "preprocess", "representation", "meta", "model", "n_features"]).to_csv(out_dir / "summary.csv", index=False)
    for row in rows[:100]:
        print(row)


if __name__ == "__main__":
    main()
