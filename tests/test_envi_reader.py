from pathlib import Path

import numpy as np

from oil_content_detection.acquisition.envi_reader import (
    load_envi_cube,
    nearest_wavelength_index,
    parse_envi_header,
)


def _write_bsq_cube(tmp_path: Path) -> tuple[Path, np.ndarray]:
    hdr_path = tmp_path / "sample.hdr"
    dat_path = tmp_path / "sample.dat"
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
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=np.float32,
    )
    cube_bsq = np.transpose(cube, (2, 0, 1))
    cube_bsq.tofile(dat_path)
    return hdr_path, cube


def test_parse_and_load_envi_cube(tmp_path: Path) -> None:
    hdr_path, cube_expected = _write_bsq_cube(tmp_path)
    header = parse_envi_header(hdr_path)

    assert header.samples == 2
    assert header.lines == 2
    assert header.bands == 2
    assert header.wavelengths == [650.0, 800.0]

    cube_loaded, parsed_header = load_envi_cube(hdr_path, memmap=False)
    assert cube_loaded.shape == cube_expected.shape
    np.testing.assert_allclose(cube_loaded, cube_expected)

    idx = nearest_wavelength_index(parsed_header.wavelengths, 780)
    assert idx == 1
