"""HSV 调参器与可复用的批处理掩膜函数."""
from __future__ import annotations

import cv2
import numpy as np


def _auto_kernel(k: int) -> int:
    return k if k % 2 == 1 else k + 1


def apply_hsv_mask(
    image_path: str,
    lower_hsv: tuple[int, int, int] = (30, 70, 30),
    upper_hsv: tuple[int, int, int] = (65, 255, 255),
    max_width: int | None = 400,
    border_crop: float = 0.0,
    blur_ksize: int = 0,
    closing_size: int = 0,
    opening_size: int = 0,
    keep_largest: bool = True,
) -> np.ndarray:
    """非交互式 HSV 掩膜生成，供批处理使用."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = image.shape[:2]
    if border_crop > 0:
        dy = int(h * border_crop)
        dx = int(w * border_crop)
        image = image[dy : h - dy, dx : w - dx]

    if max_width and image.shape[1] > max_width:
        scale_ratio = max_width / image.shape[1]
        image = cv2.resize(image, (0, 0), fx=scale_ratio, fy=scale_ratio)

    if blur_ksize and blur_ksize > 1:
        k = _auto_kernel(blur_ksize)
        image = cv2.GaussianBlur(image, (k, k), 0)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    if closing_size and closing_size > 1:
        kernel = np.ones((closing_size, closing_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if opening_size and opening_size > 1:
        kernel = np.ones((opening_size, opening_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if keep_largest:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            areas = stats[:, cv2.CC_STAT_AREA]
            areas[0] = 0  # background
            largest = int(areas.argmax())
            mask = (labels == largest).astype(np.uint8) * 255

    return mask


def apply_hsv_mask_simple(
    image_path: str,
    lower_hsv: tuple[int, int, int] = (30, 70, 30),
    upper_hsv: tuple[int, int, int] = (65, 255, 255),
    max_width: int | None = None,
) -> np.ndarray:
    """完全模拟 hsv_debugger 的分割逻辑，可选缩放。

    这个函数复制 hsv_debugger 的核心逻辑，确保批处理时得到与调参时一致的结果。

    与 apply_hsv_mask 的区别：
    - 无边界裁剪 (border_crop)
    - 无高斯模糊 (blur)
    - 无形态学操作 (closing/opening)
    - 无连通域筛选 (keep_largest)

    Args:
        image_path: 图像路径
        lower_hsv: HSV下界 (H, S, V)
        upper_hsv: HSV上界 (H, S, V)
        max_width: 缩放宽度，None表示不缩放（保持原始尺寸）

    Returns:
        二值掩膜 (0-255)，尺寸与输入图像一致（如果指定max_width则为缩放后的尺寸）
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # 可选缩放（与 hsv_debugger 第76-80行一致）
    if max_width and image.shape[1] > max_width:
        scale_ratio = max_width / image.shape[1]
        image = cv2.resize(image, (0, 0), fx=scale_ratio, fy=scale_ratio)

    # HSV分割（与 hsv_debugger 第111-112行一致）
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    return mask


def nothing(x):
    pass


def hsv_debugger(image_path: str):
    """交互式调参工具，退出后打印当前阈值。"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: 无法找到图像 {image_path}")
        return

    height, width = image.shape[:2]
    max_width = 400
    if width > max_width:
        scale_ratio = max_width / width
        image = cv2.resize(image, (0, 0), fx=scale_ratio, fy=scale_ratio)

    cv2.namedWindow("HSV Tuner")

    cv2.createTrackbar("Low H", "HSV Tuner", 0, 179, nothing)
    cv2.createTrackbar("High H", "HSV Tuner", 179, 179, nothing)
    cv2.createTrackbar("Low S", "HSV Tuner", 0, 255, nothing)
    cv2.createTrackbar("High S", "HSV Tuner", 255, 255, nothing)
    cv2.createTrackbar("Low V", "HSV Tuner", 0, 255, nothing)
    cv2.createTrackbar("High V", "HSV Tuner", 255, 255, nothing)

    cv2.setTrackbarPos("Low H", "HSV Tuner", 30)
    cv2.setTrackbarPos("High H", "HSV Tuner", 65)
    cv2.setTrackbarPos("Low S", "HSV Tuner", 70)
    cv2.setTrackbarPos("High S", "HSV Tuner", 255)
    cv2.setTrackbarPos("Low V", "HSV Tuner", 30)
    cv2.setTrackbarPos("High V", "HSV Tuner", 255)

    print("调试器已启动。按 'q' 或 'ESC' 退出...")

    while True:
        l_h = cv2.getTrackbarPos("Low H", "HSV Tuner")
        h_h = cv2.getTrackbarPos("High H", "HSV Tuner")
        l_s = cv2.getTrackbarPos("Low S", "HSV Tuner")
        h_s = cv2.getTrackbarPos("High S", "HSV Tuner")
        l_v = cv2.getTrackbarPos("Low V", "HSV Tuner")
        h_v = cv2.getTrackbarPos("High V", "HSV Tuner")

        lower_hsv = np.array([l_h, l_s, l_v])
        upper_hsv = np.array([h_h, h_s, h_v])

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        result = cv2.bitwise_and(image, image, mask=mask)

        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        stacked_img = np.hstack((image, mask_3ch, result))
        cv2.imshow("HSV Tuner", stacked_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    print("\n--- 调试结束，最佳参数 ---")
    print(f"lower_green = np.array([{l_h}, {l_s}, {l_v}])")
    print(f"upper_green = np.array([{h_h}, {h_s}, {h_v}])")
    print("--------------------------")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_file = "/home/yr/yr/data/科研数据/云南藤椒1-5/0204_11/0204_11.png"
    hsv_debugger(image_file)
