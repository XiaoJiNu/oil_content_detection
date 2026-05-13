import cv2
import numpy as np

from oil_content_detection.preprocessing.roi_segmentation import PepperROIConfig, segment_pepper_in_mask, segment_pepper_roi


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    union = (a | b).sum()
    return float((a & b).sum()) / float(union) if union else 1.0


def _make_synthetic_sample(seed: int = 0):
    rng = np.random.default_rng(seed)
    height, width = 360, 280

    # Background (table)
    img = np.full((height, width, 3), 20, dtype=np.uint8)

    # Paper (dark red)
    paper_mask = np.zeros((height, width), dtype=np.uint8)
    x0, y0, x1, y1 = 40, 40, 240, 320
    cv2.rectangle(img, (x0, y0), (x1, y1), (30, 30, 150), thickness=-1)
    cv2.rectangle(paper_mask, (x0, y0), (x1, y1), 1, thickness=-1)

    # Label (white with red border) in bottom-left
    label_mask = np.zeros((height, width), dtype=np.uint8)
    lx0, ly0, lx1, ly1 = x0 + 12, y1 - 55, x0 + 75, y1 - 12
    cv2.rectangle(img, (lx0, ly0), (lx1, ly1), (245, 245, 245), thickness=-1)
    cv2.rectangle(img, (lx0, ly0), (lx1, ly1), (0, 0, 255), thickness=3)
    cv2.rectangle(label_mask, (lx0, ly0), (lx1, ly1), 1, thickness=-1)

    # Peppercorns (green/purple dots) away from label with generous margin
    pepper_mask = np.zeros((height, width), dtype=np.uint8)
    for _ in range(180):
        cx = int(rng.integers(x0 + 90, x1 - 30))
        cy = int(rng.integers(y0 + 30, y1 - 110))
        radius = int(rng.integers(3, 5))
        color = (0, 200, 0) if rng.random() < 0.8 else (120, 0, 120)
        cv2.circle(img, (cx, cy), radius, color, thickness=-1)
        cv2.circle(pepper_mask, (cx, cy), radius, 1, thickness=-1)

    return img, pepper_mask.astype(bool), label_mask.astype(bool), paper_mask.astype(bool)


def test_segment_pepper_roi_synthetic_removes_label_and_background():
    img, gt_pepper, gt_label, gt_paper = _make_synthetic_sample()
    # Use conservative morphology for synthetic dots; real images should use defaults.
    cfg = PepperROIConfig(
        use_texture=False,
        bg_dilate_size=1,
        pepper_open_size=1,
        pepper_close_size=1,
        label_close_size=5,
        label_dilate_size=9,
        label_final_dilate_size=15,
        min_component_area=5,
    )
    result = segment_pepper_roi(img, config=cfg)

    assert result.paper_mask.sum() >= int(gt_paper.sum() * 0.95)
    assert (result.label_mask & gt_label).sum() >= int(gt_label.sum() * 0.80)

    pepper_recall = (result.pepper_mask & gt_pepper).sum() / float(gt_pepper.sum())
    assert pepper_recall >= 0.95

    assert (result.pepper_mask & gt_label).sum() == 0
    assert (result.pepper_mask & (~gt_paper)).sum() == 0


def test_segment_pepper_roi_texture_does_not_crash():
    img, gt_pepper, gt_label, gt_paper = _make_synthetic_sample(seed=1)
    cfg = PepperROIConfig(use_texture=True, texture_min_pixels=50, texture_min_keep_ratio=0.05, min_component_area=5)
    result = segment_pepper_roi(img, config=cfg)

    assert result.pepper_mask.sum() > 0
    assert (result.pepper_mask & gt_label).sum() == 0
    assert (result.pepper_mask & (~gt_paper)).sum() == 0


def test_segment_pepper_in_mask_respects_roi_and_removes_label():
    img, gt_pepper, gt_label, gt_paper = _make_synthetic_sample(seed=2)
    cfg = PepperROIConfig(
        use_texture=False,
        bg_dilate_size=1,
        pepper_open_size=1,
        pepper_close_size=1,
        label_close_size=5,
        label_dilate_size=9,
        label_final_dilate_size=15,
        min_component_area=5,
    )
    result = segment_pepper_in_mask(img, gt_paper, config=cfg)

    assert result.paper_mask.sum() == gt_paper.sum()
    assert (result.pepper_mask & (~gt_paper)).sum() == 0
    assert (result.pepper_mask & gt_label).sum() == 0

    pepper_recall = (result.pepper_mask & gt_pepper).sum() / float(gt_pepper.sum())
    assert pepper_recall >= 0.95


def test_segment_pepper_roi_gray_otsu_does_not_crash_and_records_strategy():
    img, gt_pepper, gt_label, gt_paper = _make_synthetic_sample(seed=3)
    cfg = PepperROIConfig(
        segmentation_mode="gray_otsu",
        use_texture=False,
        bg_dilate_size=1,
        pepper_open_size=1,
        pepper_close_size=1,
        label_close_size=5,
        label_dilate_size=9,
        label_final_dilate_size=15,
        min_component_area=5,
    )
    result = segment_pepper_roi(img, config=cfg)

    assert result.info["segmentation_strategy"] == "gray_otsu"
    assert result.pepper_mask.sum() > 0
    assert (result.pepper_mask & gt_label).sum() == 0
    assert (result.pepper_mask & (~gt_paper)).sum() == 0


def test_segment_pepper_roi_hybrid_selects_reasonable_mask_when_texture_oversegments():
    rng = np.random.default_rng(4)
    height, width = 360, 280
    img = np.full((height, width, 3), 20, dtype=np.uint8)
    paper = np.zeros((height, width), dtype=np.uint8)
    x0, y0, x1, y1 = 40, 40, 240, 320
    cv2.rectangle(img, (x0, y0), (x1, y1), (30, 30, 150), thickness=-1)
    cv2.rectangle(paper, (x0, y0), (x1, y1), 1, thickness=-1)

    # Add red-paper texture/noise that tends to make texture-OR masks too large.
    noise = rng.normal(0, 18, size=(y1 - y0 + 1, x1 - x0 + 1, 1)).astype(np.int16)
    patch = img[y0 : y1 + 1, x0 : x1 + 1].astype(np.int16)
    patch = np.clip(patch + np.concatenate([noise, noise, noise], axis=2), 0, 255).astype(np.uint8)
    img[y0 : y1 + 1, x0 : x1 + 1] = patch

    pepper = np.zeros((height, width), dtype=np.uint8)
    for _ in range(90):
        cx = int(rng.integers(x0 + 45, x1 - 35))
        cy = int(rng.integers(y0 + 35, y1 - 75))
        radius = int(rng.integers(4, 7))
        cv2.circle(img, (cx, cy), radius, (35, 35, 35), thickness=-1)
        cv2.circle(pepper, (cx, cy), radius, 1, thickness=-1)

    target_cov = float(pepper.sum()) / float(height * width)
    cfg = PepperROIConfig(
        segmentation_mode="hybrid",
        use_texture=True,
        hybrid_target_coverage_ratio=target_cov,
        hybrid_min_coverage_ratio=target_cov * 0.40,
        hybrid_max_coverage_ratio=target_cov * 2.50,
        bg_dilate_size=1,
        pepper_open_size=1,
        pepper_close_size=3,
        label_fallback_enabled=False,
        min_component_area=10,
    )
    result = segment_pepper_roi(img, config=cfg)

    assert result.info["segmentation_strategy"] in {"gray_otsu", "hybrid_intersection", "color_no_texture"}
    assert result.pepper_mask.mean() <= target_cov * 2.5
    assert _mask_iou(result.pepper_mask, pepper.astype(bool)) >= 0.45
