import cv2
import numpy as np
import os
import argparse
import sys

class HuajiaoSegmentorV3:
    def __init__(self, debug=False):
        self.debug = debug

    def preprocess_image(self, img):
        """
        预处理：转换到 LAB 空间，并对 L 通道进行 CLAHE 增强。
        LAB 空间能更好地分离亮度 (L) 和色彩信息 (A:红绿, B:黄蓝)。
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE 增强 L 通道，提升暗部细节
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # 合并回 LAB (仅用于调试或后续可能的基于增强图的计算)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        return lab_enhanced, l_enhanced, a, b

    def get_paper_mask(self, a_channel, shape):
        """
        提取纸张区域 (利用 A 通道)。
        返回:
        1. paper_spatial_mask: 纸张的完整凸包范围 (包含内部的花椒和标签)
        2. red_pixel_mask: 实际识别为红色的像素掩膜
        """
        h, w = shape[:2]
        
        # 高斯模糊减少噪声
        a_blurred = cv2.GaussianBlur(a_channel, (5, 5), 0)
        
        # Otsu 阈值分割
        # 识别红色区域 (A 值较高)
        _, red_pixel_mask = cv2.threshold(a_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 形态学闭运算连接断点，得到更完整的红色区域
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed_mask = cv2.morphologyEx(red_pixel_mask, cv2.MORPH_CLOSE, kernel)
        
        # 寻找最大轮廓并计算凸包
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        paper_spatial_mask = np.zeros((h, w), dtype=np.uint8)
        
        if contours:
            # 假设面积最大的轮廓是纸张
            max_cnt = max(contours, key=cv2.contourArea)
            
            # 计算凸包
            hull = cv2.convexHull(max_cnt)
            cv2.drawContours(paper_spatial_mask, [hull], -1, 255, thickness=cv2.FILLED)
            
        return paper_spatial_mask, red_pixel_mask

    def get_label_mask(self, img, paper_mask):
        """
        在纸张区域内提取标签。
        标签特征：白色底 (高亮度 L, 低饱和度 S)。
        """
        # 转 HSV 获取饱和度 S
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 仅在纸张区域内计算
        s_masked = cv2.bitwise_and(s, s, mask=paper_mask)
        v_masked = cv2.bitwise_and(v, v, mask=paper_mask)
        
        # 阈值条件：
        # 1. 饱和度低 (白/灰)
        # 2. 亮度高 (白)
        # S < 60, V > 140 (根据经验调整)
        _, mask_s = cv2.threshold(s_masked, 60, 255, cv2.THRESH_BINARY_INV)
        _, mask_v = cv2.threshold(v_masked, 140, 255, cv2.THRESH_BINARY)
        
        white_candidate = cv2.bitwise_and(mask_s, mask_v)
        white_candidate = cv2.bitwise_and(white_candidate, white_candidate, mask=paper_mask)
        
        # 形态学去噪和连接
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        white_candidate = cv2.morphologyEx(white_candidate, cv2.MORPH_OPEN, kernel_open)
        
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        white_candidate = cv2.dilate(white_candidate, kernel_dilate, iterations=2)
        
        # 轮廓筛选 (面积和形状)
        contours, _ = cv2.findContours(white_candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        label_mask = np.zeros_like(paper_mask)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 标签面积通常较大，过滤掉小的反光点
            if area > 800:
                cv2.drawContours(label_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                
        # 再次膨胀，确保覆盖标签边缘
        label_mask = cv2.dilate(label_mask, kernel_dilate, iterations=1)
        
        return label_mask

    def get_texture_mask(self, gray_img, ksize=5):
        """
        利用局部标准差提取纹理区域。
        花椒表面粗糙 -> 高标准差
        纸张/背景平滑 -> 低标准差
        """
        img32 = gray_img.astype(np.float32)
        
        # 计算局部均值和平方的均值
        # Var(X) = E[X^2] - (E[X])^2
        blur = cv2.blur(img32, (ksize, ksize))
        blur_sq = cv2.blur(img32*img32, (ksize, ksize))
        
        variance = blur_sq - blur*blur
        variance[variance < 0] = 0 # 防止浮点误差导致的负数
        std_dev = np.sqrt(variance)
        
        # 归一化到 0-255
        std_dev_norm = cv2.normalize(std_dev, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 阈值分割：提取高纹理区域
        # 使用 Otsu 自动确定阈值
        thresh_val, texture_mask = cv2.threshold(std_dev_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 如果阈值过高(说明整体都很平滑)，强制降低阈值以保留微弱纹理
        # 经验值：如果 Otsu 阈值 > 50，可能太严格了 (视 std_dev_norm 分布而定)
        # 这里做一个简单的保护：如果 texture_mask 像素太少，尝试降低阈值
        if cv2.countNonZero(texture_mask) < (gray_img.shape[0] * gray_img.shape[1] * 0.01):
             _, texture_mask = cv2.threshold(std_dev_norm, max(10, thresh_val * 0.5), 255, cv2.THRESH_BINARY)
        
        return texture_mask, std_dev_norm

    def process(self, image_path, output_path):
        if not os.path.exists(image_path):
            print(f"[Error] 文件不存在: {image_path}")
            return

        img = cv2.imread(image_path)
        if img is None:
            print(f"[Error] 无法读取图像: {image_path}")
            return
        
        original_img = img.copy()
        h, w = img.shape[:2]

        # 1. 预处理 (LAB 转换)
        lab_enhanced, l_enhanced, a_channel, b_channel = self.preprocess_image(img)

        # 2. 获取纸张掩膜 (凸包范围 + 红色像素)
        paper_spatial_mask, red_pixel_mask = self.get_paper_mask(a_channel, img.shape)
        
        if cv2.countNonZero(paper_spatial_mask) == 0:
            print(f"[Warning] 未检测到纸张区域: {image_path}")
            return

        # 3. 获取标签掩膜 (白色区域)
        label_mask = self.get_label_mask(img, paper_spatial_mask)

        # 4. 确定感兴趣区域 (纸张范围 - 标签)
        # 这是我们搜索花椒的“候选区域”
        roi_search_area = cv2.bitwise_and(paper_spatial_mask, cv2.bitwise_not(label_mask))

        # 5. 在 ROI 中提取花椒
        # 策略：花椒 = (候选区域) AND [ (高纹理) OR (非红色底色) ]
        # - 绿色花椒：非红色，会被 (非红色底色) 捕获
        # - 紫色花椒：可能是红色，但有纹理，会被 (高纹理) 捕获
        # - 红色纸张：是红色 且 无纹理 -> 被剔除
        
        # 5.1 纹理分析
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        texture_mask, std_dev_img = self.get_texture_mask(gray, ksize=5)
        
        # 5.2 颜色分析 (非红色)
        # red_pixel_mask 是主要红色区域。
        # 我们对 red_pixel_mask 做一点膨胀，确保边缘也被认为是“红色背景”
        kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        red_bg_dilated = cv2.dilate(red_pixel_mask, kernel_bg, iterations=1)
        not_red_mask = cv2.bitwise_not(red_bg_dilated)
        
        # 5.3 组合特征
        # 位于 ROI 内
        # 并且 (有纹理 或 不是红色)
        feature_mask = cv2.bitwise_or(texture_mask, not_red_mask)
        peppercorn_candidate = cv2.bitwise_and(roi_search_area, feature_mask)
        
        # 6. 后处理 (去噪和填充)
        # 开运算去除小的噪点
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        peppercorn_mask = cv2.morphologyEx(peppercorn_candidate, cv2.MORPH_OPEN, kernel_open)
        
        # 闭运算填充花椒内部空隙
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        peppercorn_mask = cv2.morphologyEx(peppercorn_mask, cv2.MORPH_CLOSE, kernel_close)

        # 7. 保存结果
        result = cv2.bitwise_and(original_img, original_img, mask=peppercorn_mask)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, result)
        print(f"[Success] 保存结果: {output_path}")

        # 8. Debug 模式保存中间图
        if self.debug:
            base, ext = os.path.splitext(output_path)
            cv2.imwrite(f"{base}_dbg_1_a_channel.jpg", a_channel)
            cv2.imwrite(f"{base}_dbg_2_paper_spatial.jpg", paper_spatial_mask)
            cv2.imwrite(f"{base}_dbg_2_paper_red_pixels.jpg", red_pixel_mask)
            cv2.imwrite(f"{base}_dbg_3_label.jpg", label_mask)
            cv2.imwrite(f"{base}_dbg_4_std_dev.jpg", std_dev_img)
            cv2.imwrite(f"{base}_dbg_5_texture_mask.jpg", texture_mask)
            cv2.imwrite(f"{base}_dbg_6_not_red.jpg", not_red_mask)
            cv2.imwrite(f"{base}_dbg_7_final.jpg", peppercorn_mask)

def main():
    parser = argparse.ArgumentParser(description="花椒 ROI 分割工具 V3 (LAB色彩+纹理分析)")
    parser.add_argument('--input', '-i', type=str, required=True, help="输入图片路径 或 文件夹路径")
    parser.add_argument('--output', '-o', type=str, default=None, help="输出结果保存路径")
    parser.add_argument('--debug', action='store_true', help="开启调试模式，保存中间掩膜")
    
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    
    segmentor = HuajiaoSegmentorV3(debug=args.debug)
    
    if os.path.isfile(input_path):
        if output_path is None:
            folder, name = os.path.split(input_path)
            filename, ext = os.path.splitext(name)
            output_path = os.path.join(folder, f"{filename}_roi_v3{ext}")
        
        print(f"处理单张图片: {input_path}")
        segmentor.process(input_path, output_path)
        
    elif os.path.isdir(input_path):
        if output_path is None:
            output_path = os.path.join(input_path, "roi_results_v3")
            
        print(f"处理文件夹: {input_path} -> 输出至: {output_path}")
        
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        files = [f for f in os.listdir(input_path) if os.path.splitext(f)[1].lower() in img_extensions]
        
        for f in files:
            in_file = os.path.join(input_path, f)
            out_file = os.path.join(output_path, f)
            segmentor.process(in_file, out_file)
            
    else:
        print("[Error] 输入路径无效")

if __name__ == "__main__":
    main()