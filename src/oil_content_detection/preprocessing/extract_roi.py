import cv2
import numpy as np
import os
import argparse

def process_huajiao_image(image_path, save_path, debug=False):
    """
    处理花椒样本图片：剔除标签纸，提取花椒区域。
    """
    # 1. 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"[Error] 无法读取图片: {image_path}")
        return

    h, w = img.shape[:2]
    
    # ==========================================
    # Step 1: 标签纸剔除 (基于空间几何特征)
    # ==========================================
    
    # 转灰度，提取高亮区域 (标签纸白底)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 阈值设为 200，只要是比较亮的地方都认为是潜在的标签
    # 即使标签上有字，通过膨胀也能连成一片
    _, label_mask_raw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # 强力膨胀：让标签纸上的字、红块都被白底吞噬，形成一个完整的连通域
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    label_mask_dilated = cv2.dilate(label_mask_raw, kernel_dilate, iterations=2)
    
    # 寻找最大轮廓 (假设最大的高亮块就是标签纸)
    contours, _ = cv2.findContours(label_mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建一个全白的掩膜，准备把标签区域挖掉 (设为黑色)
    roi_mask_geo = np.ones((h, w), dtype=np.uint8) * 255
    
    if contours:
        # 找到面积最大的轮廓
        max_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_cnt)
        
        # 只有当高亮区域足够大时才认为是标签 (防止把花椒上的高光点误删)
        if area > 1000: 
            x, y, w_rect, h_rect = cv2.boundingRect(max_cnt)
            # 在掩膜上画黑色矩形，盖住标签
            # 适当扩大一点矩形范围 (+10像素)，确保切得干净
            cv2.rectangle(roi_mask_geo, (x-10, y-10), (x+w_rect+10, y+h_rect+10), 0, -1)
            if debug:
                print(f"[Info] 检测到标签，已剔除区域: x={x}, y={y}, w={w_rect}, h={h_rect}")

    # ==========================================
    # Step 2: 花椒提取 (基于 S 通道饱和度)
    # ==========================================
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)
    
    # 核心逻辑：花椒的饱和度(S)比背景高
    # 策略：使用 Otsu 阈值自动寻找分割点，但限制最低阈值不能太低(防止背景噪点)
    otsu_thresh, _ = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 如果 Otsu 算出来的阈值太低(比如背景太黑)，就强制提升到 40
    final_thresh_val = max(40, otsu_thresh)
    _, s_mask = cv2.threshold(s_channel, final_thresh_val, 255, cv2.THRESH_BINARY)
    
    if debug:
        print(f"[Info] 饱和度分割阈值: {final_thresh_val}")

    # ==========================================
    # Step 3: 融合与形态学优化
    # ==========================================
    
    # 1. 结合几何掩膜：去掉标签区域
    final_mask = cv2.bitwise_and(s_mask, s_mask, mask=roi_mask_geo)
    
    # 2. 开运算：去除背景上的细小噪点 (椒盐噪声)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_open)
    
    # 3. 闭运算：填补花椒颗粒内部的空洞 (反光点或纹理造成的空隙)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_close)

    # ==========================================
    # Step 4: 输出结果
    # ==========================================
    
    # 生成黑色背景的结果图
    result_img = cv2.bitwise_and(img, img, mask=final_mask)
    
    # 确保保存路径目录存在
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    cv2.imwrite(save_path, result_img)
    print(f"[Success] 处理完成: {save_path}")

    # 如果开启调试，保存中间掩膜以便观察
    if debug:
        debug_path = save_path.replace(".jpg", "_mask.png")
        cv2.imwrite(debug_path, final_mask)


if __name__ == "__main__":
    # 简单的命令行交互
    import sys
    
    print("--- 花椒 ROI 智能提取工具 (抗标签/多颜色版) ---")
    
    # 默认模式：直接修改这里路径运行
    # 你可以修改这里的 input_path 为你电脑上的实际路径
    # input_path = input("请输入图片路径 (例如 0254_11.jpg): ").strip().strip('"')
    input_path = "/home/yr/yr/data/huajiao/test/0254_11.png"
    
    if not input_path:
        print("未输入路径，程序退出。")
    else:
        # 自动生成输出文件名
        folder, filename = os.path.split(input_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(folder, f"{name}_roi{ext}")
        
        print(f"正在处理: {input_path} ...")
        process_huajiao_image(input_path, output_path, debug=True)
        print(f"结果已保存至: {output_path}")