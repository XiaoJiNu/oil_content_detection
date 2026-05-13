from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from oil_content_detection.inference.training import (
    build_deployment_pipeline,
    prepare_deployment_training_frame,
)
from scripts import package_huajiao_model as packager


def _tiny_training_df(n_rows: int = 8, n_bands: int = 2) -> pd.DataFrame:
    payload = {
        "sample_id": [f"S{i:02d}" for i in range(n_rows)],
        "pixel_count": np.arange(10, 10 + n_rows),
        "valid_pixel_count": np.arange(9, 9 + n_rows),
        "coverage_ratio": np.linspace(0.1, 0.2, n_rows),
        "weight_g": np.linspace(40.0, 50.0, n_rows),
        "batch": ["A", "B"] * (n_rows // 2) + (["A"] if n_rows % 2 else []),
        "sid_num": np.arange(n_rows),
        "oil_ml_per_100g": np.linspace(0.8, 1.5, n_rows),
    }
    for band_idx in range(n_bands):
        wavelength = 400 + band_idx * 100
        payload[f"wl_{wavelength}"] = np.linspace(
            1.0 + band_idx, 2.0 + band_idx, n_rows
        )
    return pd.DataFrame(payload)


def test_prepare_deployment_training_frame_excludes_non_deployable_features():
    X, y, feature_schema = prepare_deployment_training_frame(_tiny_training_df())

    assert list(y) == list(np.linspace(0.8, 1.5, 8))
    assert "weight_g" not in X.columns
    assert "batch" not in X.columns
    assert "sid_num" not in X.columns
    assert "oil_ml_per_100g" not in X.columns
    assert feature_schema["wavelength_columns"] == ["wl_400", "wl_500"]
    assert feature_schema["quality_columns"] == [
        "coverage_ratio",
        "log_valid_pixel_count",
        "valid_pixel_ratio",
    ]
    assert feature_schema["feature_columns"] == [
        "wl_400",
        "wl_500",
        "coverage_ratio",
        "log_valid_pixel_count",
        "valid_pixel_ratio",
    ]
    assert X["valid_pixel_ratio"].iloc[0] == 0.9


def test_build_deployment_pipeline_fits_report_derived_model_family():
    X, y, feature_schema = prepare_deployment_training_frame(_tiny_training_df())
    model = build_deployment_pipeline(
        wavelength_columns=feature_schema["wavelength_columns"],
        quality_columns=feature_schema["quality_columns"],
        n_components=2,
    )

    model.fit(X, y)
    pred = model.predict(X.iloc[:2])

    assert pred.shape == (2,)
    assert np.isfinite(pred).all()


def test_build_dinov3_roi_config_records_backend_and_relative_artifact_path(tmp_path):
    out_dir = tmp_path / "artifacts" / "huajiao-oil-content-dinov3-roi-v0.3.0"
    roi_artifact = tmp_path / "artifacts" / "huajiao-dinov3-roi-seg-v0.1.0"
    args = SimpleNamespace(
        roi_backend="dinov3",
        dinov3_roi_artifact=str(roi_artifact),
        roi_device="cpu",
        roi_segmentation_mode="hybrid",
        roi_target_coverage=0.17,
        roi_min_coverage=0.05,
        roi_max_coverage=0.35,
        disable_roi_texture=True,
    )

    roi_config = packager._build_roi_config(args, out_dir=out_dir)

    assert roi_config["backend"] == "dinov3"
    assert roi_config["dinov3"]["artifact_dir"] == "../huajiao-dinov3-roi-seg-v0.1.0"
    assert roi_config["dinov3"]["device"] == "cpu"
    assert roi_config["segmentation"]["segmentation_mode"] == "hybrid"
    assert "post_refinement" not in roi_config["dinov3"]


def test_build_dinov3_roi_config_records_optional_post_refinement(tmp_path):
    out_dir = tmp_path / "artifacts" / "huajiao-oil-content-dinov3-refined-roi-v0.3.1"
    roi_artifact = tmp_path / "artifacts" / "huajiao-dinov3-roi-seg-v0.1.0"
    args = SimpleNamespace(
        roi_backend="dinov3",
        dinov3_roi_artifact=str(roi_artifact),
        roi_device="cpu",
        roi_segmentation_mode="hybrid",
        roi_target_coverage=0.17,
        roi_min_coverage=0.05,
        roi_max_coverage=0.35,
        disable_roi_texture=False,
        dinov3_post_refine_mode="gray_otsu",
        dinov3_refine_open_size=5,
        dinov3_refine_close_size=5,
        dinov3_refine_hole_fill_max_area=300,
    )

    roi_config = packager._build_roi_config(args, out_dir=out_dir)

    assert roi_config["backend"] == "dinov3"
    post_refinement = roi_config["dinov3"]["post_refinement"]
    assert post_refinement == {
        "enabled": True,
        "segmentation_mode": "gray_otsu",
        "pepper_open_size": 5,
        "pepper_close_size": 5,
        "hole_fill_max_area": 300,
        "label_fallback_enabled": False,
        "label_max_area_ratio": 0.0,
    }


def test_feature_source_label_distinguishes_dinov3_auto_roi():
    assert (
        packager._feature_source_label("auto-roi-cube", "traditional")
        == "auto-roi-cube"
    )
    assert (
        packager._feature_source_label("auto-roi-cube", "dinov3") == "dinov3-roi-cube"
    )
    assert packager._feature_source_label("processed", "dinov3") == "processed"


def test_extract_dinov3_auto_roi_features_applies_configured_post_refinement(
    tmp_path, monkeypatch
):
    spectra_df = pd.DataFrame(
        {
            "sample_id": ["S01"],
            "sample_id_raw": ["0346_11"],
            "oil_ml_per_100g": [1.23],
            "wl_400": [0.0],
            "wl_500": [0.0],
        }
    )
    metadata_df = pd.DataFrame(
        {
            "sample_id": ["S01"],
            "sample_dir": [str(tmp_path)],
            "hdr_path": [str(tmp_path / "S01.hdr")],
        }
    )
    cube = np.arange(4 * 4 * 2, dtype=np.float32).reshape(4, 4, 2)
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    class FakeDinoV3Predictor:
        def predict_image(self, *, image_path, sample_id):
            return SimpleNamespace(
                mask=np.ones((4, 4), dtype=bool), threshold=0.5, coverage_ratio=1.0
            )

    calls: dict[str, object] = {}

    def fake_refine_mask_in_roi_if_configured(
        *, image_bgr, base_mask, roi_config, base_mask_source
    ):
        calls["base_mask_pixels"] = int(base_mask.sum())
        calls["mode"] = roi_config["dinov3"]["post_refinement"]["segmentation_mode"]
        refined = np.zeros_like(base_mask, dtype=bool)
        refined[0, 0] = True
        refined[1, 1] = True
        return (
            refined,
            {"segmentation_strategy": "gray_otsu"},
            "dinov3_roi_gray_otsu_refined",
        )

    monkeypatch.setattr(
        packager, "_image_path_from_metadata", lambda row: tmp_path / "S01.png"
    )
    monkeypatch.setattr(packager, "load_rgb_image_bgr", lambda path: image)
    monkeypatch.setattr(
        packager, "load_envi_cube", lambda path, memmap=True: (cube, None)
    )
    monkeypatch.setattr(
        packager,
        "refine_mask_in_roi_if_configured",
        fake_refine_mask_in_roi_if_configured,
    )

    out = packager._extract_auto_roi_features(
        spectra_df=spectra_df,
        metadata_df=metadata_df,
        split="train",
        wl_cols=["wl_400", "wl_500"],
        feature_columns=["wl_400", "wl_500", "coverage_ratio"],
        target="oil_ml_per_100g",
        roi_config={
            "backend": "dinov3",
            "dinov3": {
                "post_refinement": {"enabled": True, "segmentation_mode": "gray_otsu"}
            },
        },
        cache_path=tmp_path / "cache" / "train.csv",
        roi_backend="dinov3",
        dinov3_predictor=FakeDinoV3Predictor(),
    )

    assert calls == {"base_mask_pixels": 16, "mode": "gray_otsu"}
    assert out.loc[0, "auto_roi_strategy"] == "gray_otsu"
    assert out.loc[0, "coverage_ratio"] == 2 / 16
    assert out.loc[0, "wl_400"] == np.mean([0.0, 10.0])
    assert out.loc[0, "wl_500"] == np.mean([1.0, 11.0])


def test_main_records_roi_backend_in_processed_artifact_outputs(
    tmp_path, monkeypatch, capsys
):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "artifact"
    (data_dir / "train").mkdir(parents=True)
    (data_dir / "val").mkdir(parents=True)
    _tiny_training_df(n_rows=16, n_bands=20).to_parquet(
        data_dir / "train" / "huajiao_spectra.parquet"
    )
    _tiny_training_df(n_rows=8, n_bands=20).to_parquet(
        data_dir / "val" / "huajiao_spectra.parquet"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "package_huajiao_model.py",
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(out_dir),
            "--model-version",
            "test-dinov3-processed",
            "--feature-source",
            "processed",
            "--roi-backend",
            "dinov3",
            "--dinov3-roi-artifact",
            str(tmp_path / "artifacts" / "huajiao-dinov3-roi-seg-v0.1.0"),
            "--roi-device",
            "cpu",
            "--n-components",
            "2",
            "--regressor",
            "ridge",
        ],
    )

    assert packager.main() == 0

    stdout_payload = json.loads(capsys.readouterr().out)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    model_card = (out_dir / "model_card.md").read_text(encoding="utf-8")
    assert stdout_payload["feature_source"] == "processed"
    assert stdout_payload["roi_backend"] == "dinov3"
    assert manifest["feature_source"] == "processed"
    assert manifest["roi_backend"] == "dinov3"
    assert metrics["feature_source"] == "processed"
    assert metrics["roi_backend"] == "dinov3"
    assert "ROI 后端：`dinov3`" in model_card
