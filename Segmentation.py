import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import os
import numpy as np

# 基本卷积块
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

# U-Net自编码器
class UNetAutoencoder(nn.Module):
    def __init__(self):
        super(UNetAutoencoder, self).__init__()
        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.pool1(x1)
        x3 = self.down2(x2)
        x4 = self.pool2(x3)
        x5 = self.bottleneck(x4)
        x6 = self.up2(x5)
        x6 = torch.cat([x6, x3], dim=1)
        x7 = self.conv2(x6)
        x8 = self.up1(x7)
        x8 = torch.cat([x8, x1], dim=1)
        x9 = self.conv1(x8)
        out = self.out_conv(x9)
        return out

def predict_mask(model_path: str, image_path: str):
    # 加载模型
    model = UNetAutoencoder()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # 定义图像预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # 读入测试图像
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    # 前向推断
    with torch.no_grad():
        output_tensor = model(input_tensor)
    diff = torch.abs(input_tensor - output_tensor).mean(dim=1, keepdim=True)
    mask = (diff > 0.05).float()  # 根据需要调整阈值

    return mask

import matplotlib.pyplot as plt

def show_mask(mask_tensor):
    # 将张量转为CPU上的NumPy数组
    mask_np = mask_tensor.squeeze(0).detach().cpu().numpy()
    # 简单显示为灰度图
    plt.imshow(mask_np[0], cmap='gray')
    plt.axis('off')
    plt.show()

def save_mask(mask_tensor, output_path):
    # 转为CPU上的NumPy数组
    mask_np = mask_tensor.squeeze(0).detach().cpu().numpy()
    # 将单通道的数值缩放到0\~255
    mask_np = (mask_np[0] * 255).astype(np.uint8)
    # 转为PIL图像后保存
    mask_img = Image.fromarray(mask_np)
    mask_img.save(output_path)

if __name__ == '__main__':
    mask_result = predict_mask(model_path='content/unet_autoencoder.pth', image_path='content/Cyto_Urine_24_180325_01.jpeg')
    save_mask(mask_result, "content/seg/mask.png")
    #show_mask(mask_result)