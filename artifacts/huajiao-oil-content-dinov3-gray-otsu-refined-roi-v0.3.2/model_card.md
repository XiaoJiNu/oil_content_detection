# huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.2

Deployable Huajiao oil-content regression artifact using DINOv3 coarse ROI + gray-Otsu in-polygon refinement.

## Input contract

- RGB image + ENVI hyperspectral cube.
- Serving feature schema remains `wl_*` mean spectra plus ROI QC fields: `coverage_ratio`, `log_valid_pixel_count`, `valid_pixel_ratio`.
- `model.joblib` internally derives additional spectral summary features from those serving columns.
- Forbidden/unavailable fields are not model inputs: `weight_g`, `batch`, `sid_num`, `distill_ml`, target labels.
- RGB/cube shape tolerance follows `roi_config.json`: resize RGB to cube `(lines, samples)` only when each dimension differs by <= 5 px.

## Metrics

Formal split, trained on train only:

- Train: R²=0.6451, RMSE=0.2638, MAE=0.2037, n=157
- Val: R²=0.8010, RMSE=0.2010, MAE=0.1623, n=40
- Previous v0.3.1 Val: R²=0.7780, RMSE=0.2123, MAE=0.1707
- Historical manual/report best Val: R²=0.8502, RMSE=0.1744, MAE=0.1449

## Selected candidate

```json
{
  "candidate_id": "cand_0284",
  "preprocess": "snv_sg0",
  "spectral_feature_set": "summary",
  "n_components": 12,
  "regressor_type": "ridge",
  "ridge_alpha": 1.0,
  "huber_alpha": 0.001,
  "huber_epsilon": 1.35,
  "elasticnet_alpha": 0.01,
  "elasticnet_l1_ratio": 0.2,
  "svr_c": 1.0,
  "svr_epsilon": 0.05,
  "n_estimators": 300,
  "max_depth": null,
  "min_samples_leaf": 2
}
```

## Limitations

This artifact still does not fully reach the manual-large-ROI historical result. Diagnostics show the remaining gap is primarily caused by automatic DINOv3 localization/refinement shifting the selected spectral pixels and by the serving constraint that excludes weight/batch/sample-id features used by the archived report recipe.
