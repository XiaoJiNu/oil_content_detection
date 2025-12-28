import cv2
import numpy as np
import os

def process_huajiao_advanced_v5(image_path, save_path, debug=False):
    """
    V5 版本：基于局部标准差（纹理能量）的分割。
    
    核心改进：
    放弃 Canny 边缘检测，改用局部标准差算法。
    花椒是"粗糙"的(高方差)，背景是"平滑"的(低方差)。
    这种方法比边缘检测更不易受光照和模糊影响。
    """
    # 1. 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"[Error] 无法读取图片: {image_path}")
        return

    h, w = img.shape[:2]
    
    # ==========================================
    # Step 1: 标签纸定位与剔除 (保持 V4 的几何法)
    # ==========================================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 提取高亮区域 (标签纸)
    _, label_thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    # 强力膨胀，锁定标签区域
    kernel_label = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40)) 
    mask_label_dilated = cv2.dilate(label_thresh, kernel_label, iterations=2)
    
    # 生成标签屏蔽罩 mask_ignore
    mask_ignore = np.zeros((h, w), dtype=np.uint8)
    contours_label, _ = cv2.findContours(mask_label_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours_label:
        max_label = max(contours_label, key=cv2.contourArea)
        if cv2.contourArea(max_label) > 3000:
            cv2.drawContours(mask_ignore, [max_label], -1, 255, thickness=cv2.FILLED)
            # 额外膨胀，确保边缘不残留
            mask_ignore = cv2.dilate(mask_ignore, cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30)))

    # ==========================================
    # Step 2: 纹理能量提取 (局部标准差法) - 核心改进
    # ==========================================
    
    # 转换为浮点型进行计算
    img_f = gray.astype(np.float32)
    
    # 计算局部方差: Var(X) = E[X^2] - (E[X])^2
    # 窗口大小：取 21x21，足够覆盖花椒颗粒的起伏，反映局部区域的"混乱度"
    ksize = (21, 21)
    
    # E[X]
    mu = cv2.blur(img_f, ksize)
    # E[X^2]
    mu2 = cv2.blur(img_f * img_f, ksize)
    
    # Var = E[X^2] - (E[X])^2
    variance = mu2 - mu * mu
    # Std Dev = sqrt(Var)
    sigma = np.sqrt(np.maximum(variance, 0))
    
    # 归一化 sigma 到 0-255 以便观察和阈值处理
    sigma_norm = cv2.normalize(sigma, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 阈值处理：
    # 背景平滑 -> sigma 极低
    # 花椒粗糙 -> sigma 高
    # 使用 Otsu 自动阈值，通常能完美分割"平滑"与"粗糙"区域
    otsu_thresh, mask_texture = cv2.threshold(sigma_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    print(f"[Info] 纹理分割阈值 (Otsu): {otsu_thresh}")
    
    # 如果 Otsu 阈值过低（说明全图都很平滑或全图都很粗糙），强制提高门槛
    # 防止背景中的微小噪点被选中
    if otsu_thresh < 5:
        _, mask_texture = cv2.threshold(sigma_norm, 10, 255, cv2.THRESH_BINARY)
        print(f"[Info] 调整纹理阈值为安全下限: 10")

    # ==========================================
    # Step 3: 后处理与筛选
    # ==========================================
    
    # 1. 剔除标签区域
    current_mask = cv2.bitwise_and(mask_texture, cv2.bitwise_not(mask_ignore))
    
    # 2. 形态学清理
    # 开运算：去噪点
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    current_mask = cv2.morphologyEx(current_mask, cv2.MORPH_OPEN, kernel_open)
    
    # 闭运算：填充花椒内部可能存在的平滑小区域
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    current_mask = cv2.morphologyEx(current_mask, cv2.MORPH_CLOSE, kernel_close)

    # 3. 面积筛选 (保留多连通域)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(current_mask, connectivity=8)
    final_mask = np.zeros((h, w), dtype=np.uint8)
    
    MIN_AREA_THRESHOLD = 500
    count_kept = 0
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > MIN_AREA_THRESHOLD:
            final_mask[labels == i] = 255
            count_kept += 1
            
    print(f"[Info] 保留了 {count_kept} 个独立花椒区域")

    # ==========================================
    # Step 4: 输出
    # ==========================================
    
    result_img = cv2.bitwise_and(img, img, mask=final_mask)
    
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cv2.imwrite(save_path, result_img)
    print(f"[Success] 处理完成: {save_path}")
    
    if debug:
        debug_dir = os.path.join(save_dir, "debug")
        if not os.path.exists(debug_dir): os.makedirs(debug_dir)
        base_name = os.path.basename(save_path)
        
        cv2.imwrite(os.path.join(debug_dir, f"1_sigma_map_{base_name}"), sigma_norm) # 极具参考价值：看哪亮就是哪粗糙
        cv2.imwrite(os.path.join(debug_dir, f"2_mask_texture_raw_{base_name}"), mask_texture)
        cv2.imwrite(os.path.join(debug_dir, f"3_mask_ignore_{base_name}"), mask_ignore)
        cv2.imwrite(os.path.join(debug_dir, f"4_final_mask_{base_name}"), final_mask)


if __name__ == "__main__":
    print("--- 花椒 ROI 提取 V5 (局部标准差/纹理能量版) ---")
    # input_path = input("请输入图片路径: ").strip().strip('"')
    input_path = "/home/yr/yr/data/huajiao/test/0254_11.png"

    
    if input_path:
        folder, filename = os.path.split(input_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(folder, f"{name}_roi_v5{ext}")
        
        print(f"正在处理: {input_path}")
        process_huajiao_advanced_v5(input_path, output_path, debug=True)
        print(f"结果已保存: {output_path}")
        print("请查看 debug 文件夹下的 '1_sigma_map'，越亮代表纹理越强。")