"""花椒样本 ROI 分割（纸张/标签/花椒）.

该模块针对红外打光下的花椒样本图像：
- 先定位暗红色纸张区域（Paper mask）
- 在纸张内剔除白底红边标签（Label mask）
- 在纸张内去除红色纸底，保留花椒区域，并可选用纹理能量增强鲁棒性（Pepper mask）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


def _ensure_odd(value: int, minimum: int = 1) -> int:
    value = int(value)
    if value < minimum:
        value = minimum
    return value if value % 2 == 1 else value + 1


def _mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(np.uint8) * 255).astype(np.uint8)


def polygons_to_mask(polygons_xy: Sequence[np.ndarray], shape_hw: Tuple[int, int]) -> np.ndarray:
    """将一个或多个多边形 (x,y) 点集转为 bool 掩膜.

    Args:
        polygons_xy: 每个元素为形状 (N,2) 的点集，坐标系为图像像素坐标 (x,y)。
        shape_hw: (height, width)
    """

    height, width = int(shape_hw[0]), int(shape_hw[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid shape_hw={shape_hw}")

    mask_u8 = np.zeros((height, width), dtype=np.uint8)
    for pts in polygons_xy:
        if pts is None:
            continue
        pts = np.asarray(pts)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
            continue
        pts_i = np.rint(pts).astype(np.int32)
        pts_i[:, 0] = np.clip(pts_i[:, 0], 0, width - 1)
        pts_i[:, 1] = np.clip(pts_i[:, 1], 0, height - 1)
        cv2.fillPoly(mask_u8, [pts_i.reshape((-1, 1, 2))], 255)
    return mask_u8 > 0


@dataclass(frozen=True)
class PepperROIConfig:
    """ROI 分割参数."""

    # Preprocess
    clahe_clip_limit: float = 2.5
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)

    # Red (paper/background) threshold in HSV
    red_hue1: Tuple[int, int] = (0, 12)
    red_hue2: Tuple[int, int] = (156, 179)  # OpenCV hue max is 179
    red_sat_min: int = 20
    red_val_min: int = 5

    # Paper mask extraction
    paper_close_size: int = 25
    paper_min_area_ratio: float = 0.10
    paper_edge_fallback: bool = True
    paper_edge_close_size: int = 25

    # Label extraction
    label_s_max: int = 90
    label_search_bottom_ratio: float = 0.55
    label_search_left_ratio: float = 0.65
    label_min_area: int = 600
    label_max_area_ratio: float = 0.10
    label_min_aspect: float = 0.25
    label_max_aspect: float = 6.0
    label_close_size: int = 7
    label_dilate_size: int = 15
    label_final_dilate_size: int = 25
    label_fallback_enabled: bool = True
    label_fallback_left_ratio: float = 0.25
    label_fallback_bottom_ratio: float = 0.25

    # Pepper extraction
    bg_dilate_size: int = 3
    pepper_open_size: int = 3
    pepper_close_size: int = 7
    min_component_area: int = 25
    hole_fill_max_area: int = 0  # New parameter for hole filling

    # Texture refinement (optional)
    use_texture: bool = True
    texture_window_size: int = 21
    texture_min_threshold: int = 10
    texture_min_pixels: int = 800
    texture_min_keep_ratio: float = 0.15

    # Segmentation Mode
    segmentation_mode: str = "color_texture"  # "color_texture" or "gray_otsu"

    # Red background fallback (ROI mode)
    use_lab_red_bg_fallback: bool = True
    lab_red_bg_min_hsv_ratio: float = 0.10
    lab_red_a_p90_min: int = 140
    lab_red_otsu_min: int = 120


@dataclass
class PepperROIResult:
    """ROI 分割结果（含中间掩膜）."""

    paper_mask: np.ndarray  # bool
    label_mask: np.ndarray  # bool
    red_bg_mask: np.ndarray  # bool
    pepper_candidate: np.ndarray  # bool
    texture_mask: Optional[np.ndarray]  # bool
    pepper_mask: np.ndarray  # bool
    info: Dict[str, float]

    def pepper_mask_uint8(self) -> np.ndarray:
        return _mask_to_uint8(self.pepper_mask)


def _create_clahe(config: PepperROIConfig) -> cv2.CLAHE:
    return cv2.createCLAHE(clipLimit=float(config.clahe_clip_limit), tileGridSize=tuple(config.clahe_tile_grid_size))


def _compute_red_mask(hsv_img: np.ndarray, config: PepperROIConfig) -> np.ndarray:
    lower1 = np.array([config.red_hue1[0], config.red_sat_min, config.red_val_min], dtype=np.uint8)
    upper1 = np.array([config.red_hue1[1], 255, 255], dtype=np.uint8)
    lower2 = np.array([config.red_hue2[0], config.red_sat_min, config.red_val_min], dtype=np.uint8)
    upper2 = np.array([config.red_hue2[1], 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv_img, lower1, upper1) > 0
    mask2 = cv2.inRange(hsv_img, lower2, upper2) > 0
    return mask1 | mask2


def _largest_contour(contours: Tuple[np.ndarray, ...]) -> Optional[np.ndarray]:
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("Empty mask: cannot compute bbox")
    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())
    return x0, y0, x1, y1


def _detect_paper_mask_from_red(
    hsv_img: np.ndarray, config: PepperROIConfig
) -> Tuple[Optional[np.ndarray], Dict[str, float], np.ndarray]:
    """返回 (paper_mask(bool) or None, info, red_mask(bool))."""
    h, w = hsv_img.shape[:2]
    red_mask = _compute_red_mask(hsv_img, config)
    close_k = _ensure_odd(config.paper_close_size, minimum=3)
    red_u8 = _mask_to_uint8(red_mask)
    red_closed = cv2.morphologyEx(
        red_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k)),
    )
    contours, _ = cv2.findContours(red_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = _largest_contour(tuple(contours))
    if cnt is None:
        return None, {"paper_source": "red", "paper_area_ratio": 0.0}, red_mask

    area = float(cv2.contourArea(cnt))
    area_ratio = area / float(h * w)
    if area_ratio < config.paper_min_area_ratio:
        return None, {"paper_source": "red", "paper_area_ratio": area_ratio}, red_mask

    paper_u8 = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(cnt)
    cv2.drawContours(paper_u8, [hull], -1, 255, thickness=cv2.FILLED)
    paper_mask = paper_u8 > 0
    return paper_mask, {"paper_source": "red", "paper_area_ratio": area_ratio}, red_mask


def _detect_paper_mask_from_edges(gray_img: np.ndarray, config: PepperROIConfig) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    h, w = gray_img.shape[:2]
    gray_blur = cv2.GaussianBlur(gray_img, (7, 7), 0)
    gx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edge = cv2.threshold(mag_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    close_k = _ensure_odd(config.paper_edge_close_size, minimum=3)
    edge_closed = cv2.morphologyEx(
        edge,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k)),
    )

    contours, _ = cv2.findContours(edge_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, {"paper_source": "edge", "paper_area_ratio": 0.0}

    cx0, cy0 = w / 2.0, h / 2.0
    best = None
    best_score = float("-inf")
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 0.05 * h * w:
            continue
        x, y, ww, hh = cv2.boundingRect(cnt)
        cx, cy = x + ww / 2.0, y + hh / 2.0
        dist2 = ((cx - cx0) / w) ** 2 + ((cy - cy0) / h) ** 2
        score = area * (1.0 - dist2)
        if score > best_score:
            best_score = score
            best = cnt

    if best is None:
        return None, {"paper_source": "edge", "paper_area_ratio": 0.0}

    area_ratio = float(cv2.contourArea(best)) / float(h * w)
    paper_u8 = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(best)
    cv2.drawContours(paper_u8, [hull], -1, 255, thickness=cv2.FILLED)
    return paper_u8 > 0, {"paper_source": "edge", "paper_area_ratio": area_ratio}


def _otsu_threshold(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    arr = values.reshape(-1, 1)
    thr, _ = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(thr)


def _search_mask_from_bbox(
    paper_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    bottom_ratio: float,
    left_ratio: float,
) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    width = max(1, x1 - x0 + 1)
    height = max(1, y1 - y0 + 1)
    bottom_ratio = float(np.clip(bottom_ratio, 0.0, 1.0))
    left_ratio = float(np.clip(left_ratio, 0.0, 1.0))
    bottom_h = max(1, int(round(height * bottom_ratio)))
    left_w = max(1, int(round(width * left_ratio)))
    ys = max(y0, y1 - bottom_h + 1)
    xe = min(x1, x0 + left_w - 1)

    mask = np.zeros_like(paper_mask, dtype=bool)
    mask[ys : y1 + 1, x0 : xe + 1] = True
    return mask & paper_mask


def _detect_label_mask(
    hsv_img: np.ndarray,
    paper_mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    config: PepperROIConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    h, w = paper_mask.shape[:2]
    s_channel = hsv_img[:, :, 1]
    v_channel = hsv_img[:, :, 2]
    paper_area = float(paper_mask.sum())

    def detect_once(search_mask: np.ndarray, expected_xy: Tuple[float, float]) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
        roi = search_mask & paper_mask
        if roi.sum() < 50:
            return None, {"label_found": 0.0}

        thr = _otsu_threshold(v_channel[roi])
        cand = (v_channel >= thr) & (s_channel <= config.label_s_max) & roi
        cand_u8 = _mask_to_uint8(cand)
        close_k = _ensure_odd(config.label_close_size, minimum=3)
        cand_u8 = cv2.morphologyEx(
            cand_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
        )
        dil_k = _ensure_odd(config.label_dilate_size, minimum=3)
        cand_u8 = cv2.dilate(cand_u8, cv2.getStructuringElement(cv2.MORPH_RECT, (dil_k, dil_k)), iterations=1)

        contours, _ = cv2.findContours(cand_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, {"label_found": 0.0, "label_v_thr": thr}

        exp_x, exp_y = expected_xy
        x0, y0, x1, y1 = bbox
        width = max(1, x1 - x0 + 1)
        height = max(1, y1 - y0 + 1)

        best = None
        best_score = float("-inf")
        max_area = config.label_max_area_ratio * paper_area if paper_area > 0 else float("inf")
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < config.label_min_area or area > max_area:
                continue
            x, y, ww, hh = cv2.boundingRect(cnt)
            if hh <= 0:
                continue
            aspect = float(ww) / float(hh)
            if aspect < config.label_min_aspect or aspect > config.label_max_aspect:
                continue
            rect_area = float(ww * hh)
            if rect_area <= 1:
                continue
            fill_ratio = area / rect_area
            if fill_ratio < 0.15:
                continue

            cx, cy = x + ww / 2.0, y + hh / 2.0
            dist2 = ((cx - exp_x) / width) ** 2 + ((cy - exp_y) / height) ** 2
            score = area * (1.0 - dist2)
            if score > best_score:
                best_score = score
                best = cnt

        if best is None:
            return None, {"label_found": 0.0, "label_v_thr": thr}

        label_u8 = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(label_u8, [best], -1, 255, thickness=cv2.FILLED)
        final_k = _ensure_odd(config.label_final_dilate_size, minimum=3)
        label_u8 = cv2.dilate(label_u8, cv2.getStructuringElement(cv2.MORPH_RECT, (final_k, final_k)), iterations=1)
        return label_u8 > 0, {"label_found": 1.0, "label_v_thr": thr}

    x0, y0, x1, y1 = bbox
    width = max(1, x1 - x0 + 1)
    height = max(1, y1 - y0 + 1)

    # Pass 1: bottom-left prior.
    search1 = _search_mask_from_bbox(paper_mask, bbox, config.label_search_bottom_ratio, config.label_search_left_ratio)
    expected1 = (x0 + 0.15 * width, y1 - 0.10 * height)
    label_mask, info = detect_once(search1, expected1)

    # Pass 2: bottom-wide prior (in case label is not left).
    if label_mask is None:
        search2 = _search_mask_from_bbox(paper_mask, bbox, config.label_search_bottom_ratio, 1.0)
        expected2 = (x0 + 0.50 * width, y1 - 0.10 * height)
        label_mask, info2 = detect_once(search2, expected2)
        info.update({f"pass2_{k}": v for k, v in info2.items()})

    if label_mask is not None:
        return label_mask, {"label_source": 1.0, **info}

    if not config.label_fallback_enabled:
        return np.zeros_like(paper_mask, dtype=bool), {"label_source": 0.0, **info}

    # Fallback: block bottom-left region of paper bbox.
    fallback = np.zeros_like(paper_mask, dtype=bool)
    fb_left = max(1, int(round(width * float(np.clip(config.label_fallback_left_ratio, 0.0, 1.0)))))
    fb_bottom = max(1, int(round(height * float(np.clip(config.label_fallback_bottom_ratio, 0.0, 1.0)))))
    ys = max(y0, y1 - fb_bottom + 1)
    xe = min(x1, x0 + fb_left - 1)
    fallback[ys : y1 + 1, x0 : xe + 1] = True
    fallback &= paper_mask
    return fallback, {"label_source": -1.0, **info, "fallback_used": 1.0}


def _local_std_map(gray_img: np.ndarray, window: int) -> np.ndarray:
    """局部标准差（纹理能量）图，输出 uint8 [0,255]."""
    gray_f = gray_img.astype(np.float32)
    k = _ensure_odd(window, minimum=3)
    mu = cv2.blur(gray_f, (k, k))
    mu2 = cv2.blur(gray_f * gray_f, (k, k))
    var = mu2 - mu * mu
    sigma = np.sqrt(np.maximum(var, 0))
    return cv2.normalize(sigma, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _fill_small_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    """填充掩膜中的小孔洞 (黑色区域)."""
    if max_area <= 0:
        return mask
    
    mask_u8 = _mask_to_uint8(mask)
    # Invert to find holes (connected components of 0s)
    mask_inv = cv2.bitwise_not(mask_u8)
    
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_inv, connectivity=8)
    
    # labels==0 is the background of mask_inv (i.e., the original foreground)
    # labels>=1 are the holes
    
    new_mask = mask_u8.copy()
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area <= max_area:
            new_mask[labels == i] = 255
            
    return new_mask > 0


def segment_pepper_roi(image_bgr: np.ndarray, config: PepperROIConfig | None = None) -> PepperROIResult:
    """对单张 BGR 图像分割花椒 ROI，返回包含中间结果的结构体."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Invalid image: empty input")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR image with 3 channels, got shape={image_bgr.shape}")

    cfg = config or PepperROIConfig()
    clahe = _create_clahe(cfg)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)
    v_enhanced = clahe.apply(v_channel)
    hsv_enhanced = cv2.merge([h_channel, s_channel, v_enhanced])

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_enhanced = clahe.apply(gray)

    paper_mask, paper_info, red_mask = _detect_paper_mask_from_red(hsv_enhanced, cfg)
    if paper_mask is None and cfg.paper_edge_fallback:
        paper_mask, edge_info = _detect_paper_mask_from_edges(gray_enhanced, cfg)
        paper_info.update(edge_info)

    if paper_mask is None or paper_mask.sum() == 0:
        raise ValueError("Paper mask detection failed")

    bbox = _bbox_from_mask(paper_mask)
    label_mask, label_info = _detect_label_mask(hsv_enhanced, paper_mask, bbox, cfg)

    texture_mask: Optional[np.ndarray] = None
    red_bg_mask = np.zeros_like(paper_mask)

    if cfg.segmentation_mode == "gray_otsu":
        # Gray + Otsu Strategy
        roi_for_otsu = paper_mask & (~label_mask)
        roi_pixels = gray_enhanced[roi_for_otsu]
        
        if roi_pixels.size > 0:
            thr_val = _otsu_threshold(roi_pixels)
            # Create binary mask: pixels > threshold
            # We don't know yet if peppers are dark or light relative to paper.
            # But usually Otsu separates two classes.
            binary_mask = (gray_enhanced > thr_val) & roi_for_otsu
            
            # Polarity check: Assume background (paper) area > pepper area
            # If the resulting mask covers > 50% of the ROI, it's likely the background.
            if binary_mask.sum() > (roi_for_otsu.sum() * 0.5):
                pepper_candidate = (~binary_mask) & roi_for_otsu
            else:
                pepper_candidate = binary_mask
        else:
            pepper_candidate = np.zeros_like(paper_mask)
            
        pepper_mask = pepper_candidate.copy()
        
    else:
        # Default: Color + Texture Strategy
        red_bg_mask = red_mask & paper_mask
        bg_dilate = _ensure_odd(cfg.bg_dilate_size, minimum=1)
        if bg_dilate > 1:
            red_bg_u8 = _mask_to_uint8(red_bg_mask)
            red_bg_u8 = cv2.dilate(
                red_bg_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_dilate, bg_dilate)), iterations=1
            )
            red_bg_mask = red_bg_u8 > 0

        pepper_candidate = paper_mask & (~label_mask) & (~red_bg_mask)

        pepper_mask = pepper_candidate.copy()
        texture_thr = float("nan")
        if cfg.use_texture:
            sigma_norm = _local_std_map(gray_enhanced, cfg.texture_window_size)
            roi = paper_mask & (~label_mask)
            vals = sigma_norm[roi]
            texture_thr = max(_otsu_threshold(vals), float(cfg.texture_min_threshold))
            texture_mask = (sigma_norm >= texture_thr) & roi

            # Use OR logic: Pepper is either (Not Red) OR (High Texture)
            refined = pepper_candidate | texture_mask
            pepper_mask = refined

    pepper_u8 = _mask_to_uint8(pepper_mask)
    open_k = _ensure_odd(cfg.pepper_open_size, minimum=1)
    close_k = _ensure_odd(cfg.pepper_close_size, minimum=1)

    # New Step 1: Initial Close to fill internal holes (especially in purple peppers)
    if close_k > 1:
        pepper_u8 = cv2.morphologyEx(
            pepper_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        )

    # Step 2: Open to break connections/gaps between peppers
    if open_k > 1:
        pepper_u8 = cv2.morphologyEx(
            pepper_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        )
    
    # Step 3: Final Close to smooth edges and fill minor holes reopened by 'Open'
    if close_k > 1:
        pepper_u8 = cv2.morphologyEx(
            pepper_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        )
    
    # Step 4: Explicit small hole filling
    if cfg.hole_fill_max_area > 0:
        filled_mask = _fill_small_holes(pepper_u8 > 0, cfg.hole_fill_max_area)
        pepper_u8 = _mask_to_uint8(filled_mask)

    if cfg.min_component_area > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(pepper_u8, connectivity=8)
        filtered = np.zeros_like(pepper_u8)
        for i in range(1, num):
            if int(stats[i, cv2.CC_STAT_AREA]) >= int(cfg.min_component_area):
                filtered[labels == i] = 255
        pepper_u8 = filtered

    pepper_mask = (pepper_u8 > 0) & paper_mask & (~label_mask)
    info: Dict[str, float] = {
        **{f"paper_{k}": float(v) for k, v in paper_info.items() if isinstance(v, (int, float))},
        **{f"label_{k}": float(v) for k, v in label_info.items() if isinstance(v, (int, float))},
        "texture_thr": float(texture_thr) if np.isfinite(texture_thr) else float("nan"),
        "paper_pixels": float(paper_mask.sum()),
        "label_pixels": float(label_mask.sum()),
        "pepper_pixels": float(pepper_mask.sum()),
    }

    return PepperROIResult(
        paper_mask=paper_mask,
        label_mask=label_mask,
        red_bg_mask=red_bg_mask,
        pepper_candidate=pepper_candidate,
        texture_mask=texture_mask,
        pepper_mask=pepper_mask,
        info=info,
    )


def segment_pepper_in_mask(
    image_bgr: np.ndarray,
    roi_mask: np.ndarray,
    config: PepperROIConfig | None = None,
) -> PepperROIResult:
    """在给定 ROI 掩膜内分割花椒区域.

    与 `segment_pepper_roi` 不同，本函数跳过“纸张区域检测”，直接使用外部提供的 ROI
    作为搜索空间（适用于已给定轮廓/多边形区域的场景）。
    """

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Invalid image: empty input")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR image with 3 channels, got shape={image_bgr.shape}")

    if roi_mask is None or roi_mask.size == 0:
        raise ValueError("Invalid roi_mask: empty input")
    if roi_mask.shape[:2] != image_bgr.shape[:2]:
        raise ValueError(f"roi_mask shape {roi_mask.shape[:2]} does not match image shape {image_bgr.shape[:2]}")

    cfg = config or PepperROIConfig()
    clahe = _create_clahe(cfg)

    paper_mask = roi_mask.astype(bool)
    if paper_mask.sum() == 0:
        raise ValueError("Invalid roi_mask: no foreground pixels")

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)
    v_enhanced = clahe.apply(v_channel)
    hsv_enhanced = cv2.merge([h_channel, s_channel, v_enhanced])

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_enhanced = clahe.apply(gray)

    bbox = _bbox_from_mask(paper_mask)
    label_mask, label_info = _detect_label_mask(hsv_enhanced, paper_mask, bbox, cfg)

    texture_mask: Optional[np.ndarray] = None
    red_bg_mask = np.zeros_like(paper_mask)
    red_ratio = 0.0
    used_lab_fallback = 0.0
    texture_thr = float("nan")

    if cfg.segmentation_mode == "gray_otsu":
        # Gray + Otsu Strategy
        roi_for_otsu = paper_mask & (~label_mask)
        roi_pixels = gray_enhanced[roi_for_otsu]
        
        if roi_pixels.size > 0:
            thr_val = _otsu_threshold(roi_pixels)
            # Create binary mask: pixels > threshold
            binary_mask = (gray_enhanced > thr_val) & roi_for_otsu
            
            # Polarity check: Assume background (paper) area > pepper area
            if binary_mask.sum() > (roi_for_otsu.sum() * 0.5):
                pepper_candidate = (~binary_mask) & roi_for_otsu
            else:
                pepper_candidate = binary_mask
        else:
            pepper_candidate = np.zeros_like(paper_mask)
            
        pepper_mask = pepper_candidate.copy()

    else:
        # Default: Color + Texture Strategy
        red_mask = _compute_red_mask(hsv_enhanced, cfg)
        red_bg_mask = red_mask & paper_mask

        paper_pixels = float(paper_mask.sum())
        red_ratio = float(red_bg_mask.sum()) / paper_pixels if paper_pixels > 0 else 0.0

        if cfg.use_lab_red_bg_fallback and red_ratio < float(cfg.lab_red_bg_min_hsv_ratio):
            lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
            a_channel = lab[:, :, 1]
            roi = paper_mask & (~label_mask)
            vals = a_channel[roi]
            if vals.size > 0:
                a_p90 = float(np.percentile(vals, 90))
                if a_p90 >= float(cfg.lab_red_a_p90_min):
                    thr_a = max(_otsu_threshold(vals.astype(np.uint8)), float(cfg.lab_red_otsu_min))
                    lab_red_mask = (a_channel >= thr_a) & paper_mask
                    red_bg_mask = red_bg_mask | lab_red_mask
                    used_lab_fallback = 1.0

        bg_dilate = _ensure_odd(cfg.bg_dilate_size, minimum=1)
        if bg_dilate > 1:
            red_bg_u8 = _mask_to_uint8(red_bg_mask)
            red_bg_u8 = cv2.dilate(
                red_bg_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_dilate, bg_dilate)), iterations=1
            )
            red_bg_mask = red_bg_u8 > 0

        pepper_candidate = paper_mask & (~label_mask) & (~red_bg_mask)

        pepper_mask = pepper_candidate.copy()
        if cfg.use_texture:
            sigma_norm = _local_std_map(gray_enhanced, cfg.texture_window_size)
            roi = paper_mask & (~label_mask)
            vals = sigma_norm[roi]
            texture_thr = max(_otsu_threshold(vals), float(cfg.texture_min_threshold))
            texture_mask = (sigma_norm >= texture_thr) & roi

            # Use OR logic: Pepper is either (Not Red) OR (High Texture)
            refined = pepper_candidate | texture_mask
            pepper_mask = refined

    pepper_u8 = _mask_to_uint8(pepper_mask)
    open_k = _ensure_odd(cfg.pepper_open_size, minimum=1)
    close_k = _ensure_odd(cfg.pepper_close_size, minimum=1)

    # New Step 1: Initial Close to fill internal holes (especially in purple peppers)
    if close_k > 1:
        pepper_u8 = cv2.morphologyEx(
            pepper_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        )

    # Step 2: Open to break connections/gaps between peppers
    if open_k > 1:
        pepper_u8 = cv2.morphologyEx(
            pepper_u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        )
    
    # Step 3: Final Close to smooth edges and fill minor holes reopened by 'Open'
    if close_k > 1:
        pepper_u8 = cv2.morphologyEx(
            pepper_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        )
    
    # Step 4: Explicit small hole filling
    if cfg.hole_fill_max_area > 0:
        filled_mask = _fill_small_holes(pepper_u8 > 0, cfg.hole_fill_max_area)
        pepper_u8 = _mask_to_uint8(filled_mask)

    if cfg.min_component_area > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(pepper_u8, connectivity=8)
        filtered = np.zeros_like(pepper_u8)
        for i in range(1, num):
            if int(stats[i, cv2.CC_STAT_AREA]) >= int(cfg.min_component_area):
                filtered[labels == i] = 255
        pepper_u8 = filtered

    pepper_mask = (pepper_u8 > 0) & paper_mask & (~label_mask)
    info: Dict[str, float] = {
        "paper_source": 2.0,
        "paper_red_ratio_hsv": float(red_ratio),
        "paper_used_lab_red_fallback": float(used_lab_fallback),
        **{f"label_{k}": float(v) for k, v in label_info.items() if isinstance(v, (int, float))},
        "texture_thr": float(texture_thr) if np.isfinite(texture_thr) else float("nan"),
        "paper_pixels": float(paper_mask.sum()),
        "label_pixels": float(label_mask.sum()),
        "pepper_pixels": float(pepper_mask.sum()),
    }

    return PepperROIResult(
        paper_mask=paper_mask,
        label_mask=label_mask,
        red_bg_mask=red_bg_mask,
        pepper_candidate=pepper_candidate,
        texture_mask=texture_mask,
        pepper_mask=pepper_mask,
        info=info,
    )


def segment_pepper_roi_from_path(
    image_path: str | Path,
    config: PepperROIConfig | None = None,
    debug_dir: Path | None = None,
) -> PepperROIResult:
    """读取图像并分割 ROI；debug_dir 不为空时保存中间结果."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    result = segment_pepper_roi(img, config=config)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem
        cv2.imwrite(str(debug_dir / f"{stem}_paper_mask.png"), _mask_to_uint8(result.paper_mask))
        cv2.imwrite(str(debug_dir / f"{stem}_label_mask.png"), _mask_to_uint8(result.label_mask))
        cv2.imwrite(str(debug_dir / f"{stem}_red_bg_mask.png"), _mask_to_uint8(result.red_bg_mask))
        cv2.imwrite(str(debug_dir / f"{stem}_pepper_candidate.png"), _mask_to_uint8(result.pepper_candidate))
        if result.texture_mask is not None:
            cv2.imwrite(str(debug_dir / f"{stem}_texture_mask.png"), _mask_to_uint8(result.texture_mask))
        cv2.imwrite(str(debug_dir / f"{stem}_pepper_mask.png"), _mask_to_uint8(result.pepper_mask))
    return result
