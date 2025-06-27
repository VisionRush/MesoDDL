import numpy as np
import cv2

def otsu_threshold(image):
    """
    输入: 
        image: 灰度图像 (单通道, 0-255)
    输出:
        best_threshold: 最佳阈值 (int)
    """
    # 计算直方图
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    total_pixels = image.size
    sum_total = np.sum(np.arange(256) * hist)
    
    best_var = -1
    best_threshold = 0
    
    sum_back = 0      # 背景像素的加权和
    weight_back = 0   # 背景像素权重 (像素数占比)
    
    for threshold in range(256):
        weight_back += hist[threshold]
        if weight_back == 0:
            continue
        
        weight_fore = total_pixels - weight_back
        if weight_fore == 0:
            break
        
        sum_back += threshold * hist[threshold]
        
        # 计算背景和前景的均值
        mean_back = sum_back / weight_back
        mean_fore = (sum_total - sum_back) / weight_fore
        
        # 计算类间方差
        var = weight_back * weight_fore * (mean_back - mean_fore) ** 2
        
        # 更新最佳阈值
        if var > best_var:
            best_var = var
            best_threshold = threshold
    
    return best_threshold

# 示例用法
if __name__ == "__main__":
    # 读取图像并转为灰度图
    image = cv2.imread("test_image.jpg", cv2.IMREAD_GRAYSCALE)
    
    # 计算 Otsu 阈值
    threshold = otsu_threshold(image)
    print(f"最佳阈值: {threshold}")
    
    # 应用阈值二值化
    binary = np.where(image > threshold, 255, 0).astype(np.uint8)
    
    # 显示结果
    cv2.imshow("Original", image)
    cv2.imshow("Otsu Binary", binary)
    cv2.waitKey(0)