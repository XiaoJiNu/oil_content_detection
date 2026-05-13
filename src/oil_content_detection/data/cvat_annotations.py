"""CVAT annotation parsing utilities.

This module currently supports CVAT 1.1 XML dumps where polygons are stored
under each <image> element.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
from xml.etree import ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class CvatImageAnnotations:
    """Per-image CVAT annotations.

    Attributes:
        name: Image filename in the CVAT dump.
        width: Image width in pixels (as recorded by CVAT).
        height: Image height in pixels (as recorded by CVAT).
        polygons: Mapping label -> list of polygon points arrays shaped (N,2),
            stored as float32 in (x,y) image coordinates.
    """

    name: str
    width: int
    height: int
    polygons: Dict[str, List[np.ndarray]]


def _parse_points(points: str) -> np.ndarray:
    pts: List[List[float]] = []
    for token in str(points).strip().split(";"):
        token = token.strip()
        if not token:
            continue
        xy = token.split(",")
        if len(xy) != 2:
            continue
        try:
            x = float(xy[0])
            y = float(xy[1])
        except ValueError:
            continue
        pts.append([x, y])
    return np.asarray(pts, dtype=np.float32)


def load_cvat_image_annotations(xml_path: str | Path) -> Dict[str, CvatImageAnnotations]:
    """Load per-image polygons from a CVAT XML dump.

    Returns:
        Dict keyed by image name (e.g. "0038_11.png").
    """

    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"CVAT xml not found: {xml_path}")

    root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
    out: Dict[str, CvatImageAnnotations] = {}

    for image_el in root.findall("image"):
        name = str(image_el.get("name", "")).strip()
        if not name:
            continue
        try:
            width = int(image_el.get("width", "0"))
            height = int(image_el.get("height", "0"))
        except ValueError:
            width, height = 0, 0

        polygons: Dict[str, List[np.ndarray]] = {}
        for poly_el in image_el.findall("polygon"):
            label = str(poly_el.get("label", "")).strip()
            pts = _parse_points(poly_el.get("points", ""))
            if pts.shape[0] < 3:
                continue
            polygons.setdefault(label, []).append(pts)

        out[name] = CvatImageAnnotations(name=name, width=width, height=height, polygons=polygons)

    if not out:
        raise ValueError(f"No <image> entries found in {xml_path}")

    return out
