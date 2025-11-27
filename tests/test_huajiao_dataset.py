from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from oil_content_detection.pipelines.huajiao_dataset import (
    AggregationConfig,
    HuajiaoROIConfig,
    build_huajiao_dataset,
)


def _write_sample_cube(tmp_path: Path) -> Path:
    sample_dir = tmp_path / "0164_11"
    sample_dir.mkdir()
    hdr_path = sample_dir / "REFLECTANCE_0164_11.hdr"
    dat_path = sample_dir / "REFLECTANCE_0164_11.dat"
    hdr_path.write_text(
        "\n".join(
            [
                "ENVI",
                "samples = 2",
                "lines = 2",
                "bands = 2",
                "interleave = bsq",
                "data type = 4",
                "byte order = 0",
                "wavelength = {650, 800}",
            ]
        )
    )
    cube = np.array(
        [
            [[0.2, 0.4], [0.2, 0.4]],
            [[0.2, 0.4], [0.2, 0.4]],
        ],
        dtype=np.float32,
    )
    cube_bsq = np.transpose(cube, (2, 0, 1))
    cube_bsq.tofile(dat_path)
    return hdr_path


def _fake_read_excel(df: pd.DataFrame) -> Callable[..., pd.DataFrame]:
    def _reader(*args: Any, **kwargs: Any) -> pd.DataFrame:
        return df.copy()

    return _reader


def test_build_huajiao_dataset(monkeypatch, tmp_path: Path) -> None:
    _write_sample_cube(tmp_path)

    labels_df = pd.DataFrame(
        {"高光谱图件编号": ["0164-11"], "蒸馏量（初）ml": [0.5], "重量": [5.0]}
    )
    monkeypatch.setattr(pd, "read_excel", _fake_read_excel(labels_df))

    roi_cfg = HuajiaoROIConfig(
        ratio_floor=1.0,
        ratio_quantile=0.1,
        intensity_quantile=0.0,
        closing_size=1,
        opening_size=1,
        min_area=1,
        clip_low=0.0,
        clip_high=1.0,
    )
    agg_cfg = AggregationConfig(trim_fraction=0.0, primary_stat="mean", include_stats=("mean", "median"))

    spectra_df, meta_df = build_huajiao_dataset(
        raw_root=tmp_path,
        excel_path=tmp_path / "labels.xls",
        output_dir=tmp_path / "out",
        roi_config=roi_cfg,
        agg_config=agg_cfg,
        save=False,
    )

    assert len(spectra_df) == 1
    assert len(meta_df) == 1

    row = spectra_df.iloc[0]
    assert row["sample_id"] == "0164_11"
    assert np.isclose(row["oil_ml_per_gram"], 0.1)
    assert np.isclose(row["wl_650"], 0.2)
    assert np.isclose(row["wl_800"], 0.4)
    assert row["valid_pixel_count"] == 4
    assert np.isclose(row["coverage_ratio"], 1.0)

    meta_row = meta_df.iloc[0]
    assert meta_row["sample_id"] == "0164_11"
    assert meta_row["wavelength_count"] == 2
