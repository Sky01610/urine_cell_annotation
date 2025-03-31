import cv2
import numpy as np

def overlay_mask(image_path, mask_path, output_path, alpha=0.5):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 调整mask的尺寸以匹配原始图像
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # 将mask转换为三通道
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 创建一个红色的mask
    red_mask = np.zeros_like(mask_color)
    red_mask[:, :, 2] = mask

    # 叠加图像
    overlay = cv2.addWeighted(image, 1, red_mask, alpha, 0)

    cv2.imwrite(output_path, overlay)

overlay_mask("content/Cyto_Urine_24_180325_01.jpeg", "content/seg/mask.png", "content/overlay_image.jpg")