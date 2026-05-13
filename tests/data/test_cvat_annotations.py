from pathlib import Path

import numpy as np

from oil_content_detection.data.cvat_annotations import load_cvat_image_annotations


def test_load_cvat_image_annotations_parses_polygons(tmp_path: Path):
    xml = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <image id="0" name="a.png" width="10" height="8">
    <polygon label="pepper" points="1,1;8,1;8,6;1,6"/>
  </image>
</annotations>
"""
    xml_path = tmp_path / "annotations.xml"
    xml_path.write_text(xml, encoding="utf-8")

    ann = load_cvat_image_annotations(xml_path)
    assert "a.png" in ann
    assert ann["a.png"].width == 10
    assert ann["a.png"].height == 8

    polys = ann["a.png"].polygons["pepper"]
    assert len(polys) == 1
    pts = polys[0]
    assert isinstance(pts, np.ndarray)
    assert pts.shape == (4, 2)
