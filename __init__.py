import os
import torch
import numpy as np
import cv2
from PIL import Image

from .colorizator import MangaColorizator

class MangaColorizationNode:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "networks")
        self.colorizer = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # 改为复数形式，表明支持批量
                "size": ("INT", {"default": 576, "min": 32, "max": 2048, "step": 32, "description": "处理分辨率，越大质量越好但更耗内存"}),
                "denoise": ("BOOLEAN", {"default": True}),
                "denoise_sigma": ("INT", {"default": 25, "min": 0, "max": 100, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "colorize_manga"
    CATEGORY = "image/processing"

    def colorize_manga(self, images, size=576, denoise=True, denoise_sigma=25):
        # 延迟加载模型
        if self.colorizer is None:
            generator_path = os.path.join(self.model_path, "generator.zip")
            extractor_path = os.path.join(self.model_path, "extractor.pth")
            self.colorizer = MangaColorizator(self.device, generator_path, extractor_path)
        
        # 处理每张图片
        batch_size = images.shape[0]
        colored_images = []
        
        for i in range(batch_size):
            # 获取单张图片
            image_np = images[i].cpu().numpy()
            original_h, original_w = image_np.shape[:2]
            
            # 设置图像并进行着色
            self.colorizer.set_image(image_np, size, denoise, denoise_sigma)
            colored_image = self.colorizer.colorize()
            
            # 将结果缩放回原始尺寸
            if colored_image.shape[:2] != (original_h, original_w):
                colored_image = cv2.resize(colored_image, (original_w, original_h), interpolation=cv2.INTER_LANCZOS4)
            
            # 将结果转换为张量并添加到列表
            colored_tensor = torch.from_numpy(colored_image).float()
            colored_images.append(colored_tensor)
        
        # 将所有处理后的图片堆叠成批次
        if len(colored_images) > 0:
            colored_batch = torch.stack(colored_images)
            return (colored_batch,)
        else:
            # 如果没有图片，返回空批次
            return (torch.empty((0, 3, size, size)),)

NODE_CLASS_MAPPINGS = {
    "MangaColorization": MangaColorizationNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MangaColorization": "Manga Colorization"
} 