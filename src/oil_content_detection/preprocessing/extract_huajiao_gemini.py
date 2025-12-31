import cv2
import numpy as np
import os
import argparse
import sys

class HuajiaoSegmentor:
    def __init__(self, debug=False):
        self.debug = debug

    def preprocess_image(self, img):
        """
        预处理：将图像转为 HSV，并对 V 通道进行 CLAHE 增强。
        解决暗光样本 (如 0221_11) 纸张与背景区分度低的问题。
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # 创建 CLAHE 对象 (自适应直方图均衡化)
        # clipLimit=2.0 适度增强，过大会引入噪声
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        v_enhanced = clahe.apply(v)

        # 合并回 HSV
        hsv_enhanced = cv2.merge([h, s, v_enhanced])
        return hsv_enhanced, v_enhanced

    def get_paper_spatial_mask(self, hsv_img, shape):
        """
        提取纸张的'物理范围'掩膜。
        方法：提取红色 -> 闭运算连接断点 -> 寻找最大轮廓 -> 计算凸包(Convex Hull)
        这样即使中间有花椒（非红色），也能得到一张完整的纸的掩膜。
        """
        h, w = shape[:2]
        
        # 定义红色的 HSV 范围 (纸张是暗红色)
        # 区间1: 0-10 (红色一端)
        lower_red1 = np.array([0, 30, 30])
        upper_red1 = np.array([10, 255, 255])
        # 区间2: 156-180 (红色另一端)
        lower_red2 = np.array([156, 30, 30])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
        raw_red_mask = mask1 | mask2

        # 形态学操作：先闭运算连接碎片
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed_mask = cv2.morphologyEx(raw_red_mask, cv2.MORPH_CLOSE, kernel)

        # 寻找轮廓
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        paper_spatial_mask = np.zeros((h, w), dtype=np.uint8)
        
        if contours:
            # 找到面积最大的轮廓 (假设是纸)
            max_cnt = max(contours, key=cv2.contourArea)
            
            # 使用凸包 (Convex Hull) 确保获得完整的纸张矩形区域，填补所有内部空洞
            hull = cv2.convexHull(max_cnt)
            cv2.drawContours(paper_spatial_mask, [hull], -1, 255, thickness=cv2.FILLED)
        
        return paper_spatial_mask, raw_red_mask

    def get_label_mask(self, hsv_img, paper_mask):
        """
        在纸张区域内提取标签。
        标签特征：白色底 (低饱和度, 高亮度)。
        """
        # 仅在纸张区域内搜索
        masked_hsv = cv2.bitwise_and(hsv_img, hsv_img, mask=paper_mask)
        
        # 白色的 HSV 阈值
        # S: 0-60 (偏白/灰)
        # V: 150-255 (够亮)
        lower_white = np.array([0, 0, 140])
        upper_white = np.array([180, 80, 255])
        
        white_mask = cv2.inRange(masked_hsv, lower_white, upper_white)
        
        # 形态学连接
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        white_mask = cv2.dilate(white_mask, kernel, iterations=2) # 膨胀以覆盖字体

        # 轮廓筛选
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        label_mask = np.zeros_like(paper_mask)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            
            # 筛选条件：面积不能太小，且形状像标签(通常是矩形)
            if area > 800: 
                # 这里可以稍微扩大一点标签区域，防止边缘残留
                cv2.drawContours(label_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                
        # 再次膨胀标签掩膜，确保切除干净
        label_mask = cv2.dilate(label_mask, kernel, iterations=1)
        return label_mask

    def process(self, image_path, output_path):
        if not os.path.exists(image_path):
            print(f"[Error] 文件不存在: {image_path}")
            return

        # 1. 读取
        img = cv2.imread(image_path)
        if img is None:
            print(f"[Error] 无法读取图像: {image_path}")
            return
        original_img = img.copy()

        # 2. 预处理 (CLAHE增强)
        hsv_enhanced, v_enhanced = self.preprocess_image(img)

        # 3. 获取纸张的空间范围 (Spatial Mask) 和 背景红色掩膜 (Bg Mask)
        paper_spatial_mask, paper_red_bg_mask = self.get_paper_spatial_mask(hsv_enhanced, img.shape)
        
        if cv2.countNonZero(paper_spatial_mask) == 0:
            print(f"[Warning] 未检测到纸张区域: {image_path}")
            return

        # 4. 获取标签掩膜
        label_mask = self.get_label_mask(hsv_enhanced, paper_spatial_mask)

        # 5. 核心分割逻辑
        # 目标(花椒) = (纸张空间范围) - (标签区域) - (红色的纸张底色)
        
        # 步骤 5.1: 排除标签
        roi_no_label = cv2.bitwise_and(paper_spatial_mask, cv2.bitwise_not(label_mask))
        
        # 步骤 5.2: 排除红色背景
        # 注意：paper_red_bg_mask 是严格的红色像素。
        # 我们对红色背景做一点点膨胀，确保边缘的红色光晕也被去掉
        bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        paper_red_bg_dilated = cv2.dilate(paper_red_bg_mask, bg_kernel, iterations=1)
        
        peppercorn_mask = cv2.bitwise_and(roi_no_label, cv2.bitwise_not(paper_red_bg_dilated))

        # 6. 后处理 (去噪)
        # 开运算去除孤立噪点
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        peppercorn_mask = cv2.morphologyEx(peppercorn_mask, cv2.MORPH_OPEN, kernel_open)
        # 闭运算填充花椒内部空隙
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        peppercorn_mask = cv2.morphologyEx(peppercorn_mask, cv2.MORPH_CLOSE, kernel_close)

        # 7. 保存结果
        # 生成黑底图
        result = cv2.bitwise_and(original_img, original_img, mask=peppercorn_mask)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, result)
        print(f"[Success] 保存结果: {output_path}")

        # 8. Debug 模式保存中间图
        if self.debug:
            base, ext = os.path.splitext(output_path)
            cv2.imwrite(f"{base}_debug_1_hsv_enhanced.jpg", cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR))
            cv2.imwrite(f"{base}_debug_2_paper_spatial.jpg", paper_spatial_mask)
            cv2.imwrite(f"{base}_debug_3_label.jpg", label_mask)
            cv2.imwrite(f"{base}_debug_4_red_bg.jpg", paper_red_bg_mask)
            cv2.imwrite(f"{base}_debug_5_mask_final.jpg", peppercorn_mask)

def main():
    parser = argparse.ArgumentParser(description="花椒 ROI 智能分割工具 (抗暗光/抗标签)")
    parser.add_argument('--input', '-i', type=str, default="/home/yr/yr/data/huajiao_test/0221_11.png", help="输入图片路径 或 文件夹路径")
    parser.add_argument('--output', '-o', type=str, default=None, help="输出结果保存路径 (默认保存在输入同级的 _roi 目录)")
    parser.add_argument('--debug', default=True, help="开启调试模式，保存中间掩膜")
    
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    
    segmentor = HuajiaoSegmentor(debug=args.debug)
    
    # 逻辑：判断是单张图片还是文件夹
    if os.path.isfile(input_path):
        # 单张处理
        if output_path is None:
            folder, name = os.path.split(input_path)
            filename, ext = os.path.splitext(name)
            output_path = os.path.join(folder,  f"{filename}_roi{ext}")
        
        print(f"处理单张图片: {input_path}")
        segmentor.process(input_path, output_path)
        
    elif os.path.isdir(input_path):
        # 文件夹批量处理
        if output_path is None:
            output_path = os.path.join(input_path, "roi_results")
            
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