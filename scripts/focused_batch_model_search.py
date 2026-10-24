#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, HuberRegressor, LinearRegression, Ridge, RidgeCV, TheilSenRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))
from oil_content_detection.preprocessing import PreprocessStep, apply_preprocessing_pipeline  # noqa: E402

warnings.filterwarnings("ignore")


def wl_cols(df):
    return sorted([c for c in df.columns if c.startswith("wl_") and c[3:].isdigit()], key=lambda c: int(c.split("_")[1]))


def add_meta(df, md):
    out = df.copy()
    out["batch"] = md["sample_dir"].astype(str).apply(lambda s: Path(s).parent.name).values
    out["sid_num"] = out["sample_id"].astype(str).str.extract(r"(\d+)").astype(int)
    out["log_valid_pixel_count"] = np.log1p(out["valid_pixel_count"])
    out["valid_pixel_ratio"] = out["valid_pixel_count"] / out["pixel_count"]
    out["removed_pixel_ratio"] = 1 - out["valid_pixel_ratio"]
    out["log_pixel_count"] = np.log1p(out["pixel_count"])
    out["log_weight"] = np.log(out["weight_g"])
    out["weight_per_valid_pixel"] = out["weight_g"] / out["valid_pixel_count"]
    out["weight_per_pixel"] = out["weight_g"] / out["pixel_count"]
    out["coverage_x_weight"] = out["coverage_ratio"] * out["weight_g"]
    out["coverage_x_logvalid"] = out["coverage_ratio"] * out["log_valid_pixel_count"]
    return out


def onehot_append(A, B, tr, va, cols):
    dtr = pd.DataFrame(A, index=tr.index)
    dva = pd.DataFrame(B, index=va.index)
    for c in cols:
        if tr[c].dtype == "object":
            cats = sorted(set(tr[c].astype(str)) | set(va[c].astype(str)))
            for cat in cats:
                dtr[f"{c}={cat}"] = (tr[c].astype(str) == cat).astype(float).values
                dva[f"{c}={cat}"] = (va[c].astype(str) == cat).astype(float).values
        else:
            dtr[c] = tr[c].to_numpy(float)
            dva[c] = va[c].to_numpy(float)
    return dtr.to_numpy(float), dva.to_numpy(float), list(dtr.columns.astype(str))


def metrics(y, p):
    return dict(r2=float(r2_score(y, p)), rmse=float(np.sqrt(mean_squared_error(y, p))), mae=float(mean_absolute_error(y, p)))


def main():
    data = repo_root / "data/processed/huajiao_refined"
    tr = pd.read_parquet(data / "train/huajiao_spectra.parquet")
    va = pd.read_parquet(data / "val/huajiao_spectra.parquet")
    mdtr = pd.read_parquet(data / "train/huajiao_metadata.parquet")
    mdva = pd.read_parquet(data / "val/huajiao_metadata.parquet")
    tr = add_meta(tr, mdtr)
    va = add_meta(va, mdva)
    ytr = tr["oil_ml_per_100g"].to_numpy(float)
    yva = va["oil_ml_per_100g"].to_numpy(float)
    cols = wl_cols(tr)
    X0 = tr[cols].to_numpy(float)
    V0 = va[cols].to_numpy(float)
    preprocesses = {
        "sg1_w13": [PreprocessStep("sg", {"window_length": 13, "polyorder": 2, "deriv": 1})],
        "snv_sg1_w13": [PreprocessStep("snv"), PreprocessStep("sg", {"window_length": 13, "polyorder": 2, "deriv": 1})],
        "sg_w13": [PreprocessStep("sg", {"window_length": 13, "polyorder": 2, "deriv": 0})],
        "raw": [],
    }
    meta_sets = {
        "batch_only": ["batch"],
        "batch_quality": ["batch", "coverage_ratio", "log_valid_pixel_count", "valid_pixel_ratio", "weight_g", "sid_num"],
        "batch_quality2": ["batch", "coverage_ratio", "log_valid_pixel_count", "valid_pixel_ratio", "removed_pixel_ratio", "log_pixel_count", "weight_g", "log_weight", "weight_per_valid_pixel", "weight_per_pixel", "coverage_x_weight", "coverage_x_logvalid", "sid_num"],
        "quality2": ["coverage_ratio", "log_valid_pixel_count", "valid_pixel_ratio", "removed_pixel_ratio", "log_pixel_count", "weight_g", "log_weight", "weight_per_valid_pixel", "weight_per_pixel", "coverage_x_weight", "coverage_x_logvalid", "sid_num"],
        "none": [],
    }
    rows = []
    best_payload = None
    for prep_name, steps in preprocesses.items():
        X = apply_preprocessing_pipeline(X0, steps) if steps else X0.copy()
        V = apply_preprocessing_pipeline(V0, steps) if steps else V0.copy()
        reps = []
        for n in range(2, 31):
            if n < min(X.shape):
                pls = PLSRegression(n_components=n, scale=False).fit(X, ytr)
                reps.append((f"pls{n}", pls.transform(X), pls.transform(V), {"type": "pls", "n_components": n}))
        for n in [5, 8, 10, 15, 20, 25, 30, 35, 40, 50]:
            if n < min(X.shape):
                pca = PCA(n_components=n, random_state=0).fit(X)
                reps.append((f"pca{n}", pca.transform(X), pca.transform(V), {"type": "pca", "n_components": n}))
        reps.append(("spec", X, V, {"type": "identity"}))
        for rep_name, A0, B0, rep_detail in reps:
            for meta_name, mcols in meta_sets.items():
                A, B, feature_names = onehot_append(A0, B0, tr, va, mcols)
                model_specs = []
                model_specs.append(("linear", make_pipeline(StandardScaler(), LinearRegression())))
                for alpha in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]:
                    model_specs.append((f"ridge_{alpha}", make_pipeline(StandardScaler(), Ridge(alpha=alpha))))
                for alpha in [0.0001, 0.001, 0.01, 0.1]:
                    for eps in [1.1, 1.25, 1.35, 1.5, 1.75, 2.0]:
                        model_specs.append((f"huber_a{alpha}_e{eps}", make_pipeline(StandardScaler(), HuberRegressor(alpha=alpha, epsilon=eps, max_iter=2000))))
                model_specs.append(("bayes", make_pipeline(StandardScaler(), BayesianRidge())))
                if A.shape[1] <= 90:
                    for C in [0.3, 1, 3, 10, 30, 100]:
                        for gamma in ["scale", 0.001, 0.003, 0.01, 0.03, 0.1]:
                            model_specs.append((f"svr_C{C}_g{gamma}", make_pipeline(StandardScaler(), SVR(C=C, epsilon=0.05, gamma=gamma))))
                    for gamma in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]:
                        for alpha in [0.001, 0.01, 0.05, 0.1, 1.0]:
                            model_specs.append((f"krr_a{alpha}_g{gamma}", make_pipeline(StandardScaler(), KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma))))
                    for depth in [1, 2, 3]:
                        model_specs.append((f"gbr_d{depth}", GradientBoostingRegressor(random_state=7, learning_rate=0.02, n_estimators=500, max_depth=depth)))
                    model_specs.append(("extra", ExtraTreesRegressor(n_estimators=800, random_state=7, min_samples_leaf=1, max_features=None)))
                for model_name, model in model_specs:
                    try:
                        model.fit(A, ytr)
                        pred = np.asarray(model.predict(B)).ravel()
                        mt = metrics(yva, pred)
                    except Exception:
                        continue
                    row = {**mt, "preprocess": prep_name, "representation": rep_name, "meta": meta_name, "model": model_name, "n_features": int(A.shape[1])}
                    rows.append(row)
                    if best_payload is None or mt["r2"] > best_payload["metrics"]["r2"]:
                        best_payload = {"metrics": mt, "row": row, "val_pred": pred.tolist(), "rep_detail": rep_detail, "meta_cols": mcols, "feature_names": feature_names}
    out = repo_root / "results/huajiao_formal_focused_search"
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows).sort_values("r2", ascending=False)
    summary.to_csv(out / "summary.csv", index=False)
    with open(out / "best.json", "w") as f:
        json.dump(best_payload, f, indent=2, ensure_ascii=False)
    print(summary.head(80).to_string(index=False))
    print("BEST", json.dumps(best_payload["row"], ensure_ascii=False))

if __name__ == "__main__":
    main()
