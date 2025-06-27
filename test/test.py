import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from skimage.measure import label, regionprops
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from mesorch import Mesorch
from tta import batch_tta_inference
import cv2
from otsu import otsu_threshold
from typing import Optional
from delete import remove_duplicates
import argparse
# 配置参数

parser = argparse.ArgumentParser('Model Testing', add_help=True)

# -------------------------------
# Model name
parser.add_argument('--root_dir', default=None, type=str,
                    help='test imgs dir', required=True)
parser.add_argument('--ckpt', default=None, type=str,
                    help='model weight path',required=True)
parser.add_argument('--output_mask_dir', default=None, type=str,
                    help='预测掩码路径')
# ----Dataset parameters 数据集相关的参数----
parser.add_argument('--output_txt_path', default=None, type=str,
                    help='提交的置信度文件路径')
parser.add_argument('--input_size', default=512, type=int,
                    help='输入大小')    
args = parser.parse_args()

class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()  # 使用的GPU数量
    world_size = num_gpus                 # 分布式训练的进程数

def pil_loader(path: str) -> Image.Image:
    """PIL image loader

    Args:
        path (str): image path

    Returns:
        Image.Image: PIL image (after np.array(x) becomes [0,255] int8)
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
# 自定义数据集
class ForgeryTestDataset(Dataset):
    def __init__(self, root_dir):
        self.tp_list = []
        self.transform = albu.Compose([
            albu.Resize(args.input_size, args.input_size),
            albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            albu.Crop(0, 0, args.input_size, args.input_size),
            ToTensorV2(transpose_mask=True)
        ])
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith('.png'):
                    self.tp_list.append(os.path.join(root, file))
        print("加载图片完毕！")
        print(len(self.tp_list))
    
    def __len__(self):
        return len(self.tp_list)

    def __getitem__(self, idx):
        img_path = self.tp_list[idx]
        image = pil_loader(img_path)
        original_size = image.size  # (W, H)
        image = np.array(image)
        Dict = self.transform(image=image)
        return Dict['image'], original_size, img_path

# 加载模型
def load_model():
    model = Mesorch()  # 替换为您的模型定义
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model

# 后处理函数
def morphological_operations(
    binary_mask: np.ndarray,
    kernel_size: int = 3,
    kernel_type: str = "ellipse",
    erode_iter: int = 1,
    dilate_iter: int = 1,
    close_iter: int = 0,
    min_area: int = 50
) -> np.ndarray:
    # 确保kernel_size为正奇数
    kernel_size = max(3, kernel_size)
    kernel_size = kernel_size // 2 * 2 + 1

    # 转换为0/255的uint8
    mask_uint8 = (binary_mask.astype(np.float32) > 0).astype(np.uint8) * 255

    # 创建结构元素
    kernel = cv2.getStructuringElement(
        {"ellipse": cv2.MORPH_ELLIPSE, 
         "rect": cv2.MORPH_RECT, 
         "cross": cv2.MORPH_CROSS}[kernel_type],
        (kernel_size, kernel_size)
    )

    # 开运算
    opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=erode_iter)
    
    # 闭运算
    # closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

    # 连通区域过滤
    # if min_area > 0:
    #     contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     filtered = np.zeros_like(opened)
    #     for cnt in contours:
    #         if cv2.contourArea(cnt) >= min_area:
    #             cv2.drawContours(filtered, [cnt], -1, 255, cv2.FILLED)
    #     opened = filtered

    return opened

# 单个进程的测试函数
def test(gpu, ngpus_per_node):
    # 设置当前进程使用的GPU
    torch.cuda.set_device(gpu)
    
    # 初始化进程组
    init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=Config.world_size,
        rank=gpu
    )
    
    # 创建模型并分布到GPU
    model = load_model().to(gpu)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    
    # 创建分布式采样器
    dataset = ForgeryTestDataset(args.root_dir)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=Config.world_size,
        rank=gpu,
        shuffle=False
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=200 // Config.world_size,  # 每个GPU的batch_size
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=sampler
    )
    
    # 创建输出文件（仅在主进程中创建）
    if gpu == 0:
        os.makedirs(args.output_mask_dir, exist_ok=True)
        open(args.output_txt_path, 'w').close()  # 清空文件
    
    # 等待主进程创建文件
    torch.distributed.barrier()
    # 创建输出文件（每个进程创建自己的临时文件）
    temp_output_path = f"{args.output_txt_path}.gpu{gpu}"
    open(temp_output_path, 'w').close()
    # 主测试流程
    with open(temp_output_path, 'a') as f_txt:
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, disable=gpu != 0):
                images, orig_sizes, img_paths = batch
                images = images.to(gpu)
                
                # 模型预测
                # pred_masks, pred_label=model(images)
                pred_masks, pred_label=batch_tta_inference(gpu,model,images,fusion_strategy='max')
  
                # 计算置信度（取最大值）
                confidences = pred_label.cpu().numpy()
                
                # 处理每个样本
                for i in range(len(img_paths)):
                    w, h = orig_sizes[0][i].item(), orig_sizes[1][i].item()
                    base_name = os.path.basename(img_paths[i])
                    mask_filename = f"{os.path.splitext(base_name)[0]}.png"
                    output_path = os.path.join(args.output_mask_dir, mask_filename)
                    

                    # 生成预测掩码（原图尺寸）
                    mask = pred_masks[i].unsqueeze(0)  # (1,1, H, W)
                    mask = F.interpolate(mask, size=(h, w), mode='nearest')[0]
                    # binary_mask = (mask > 0.5).float().squeeze().cpu().numpy()
                    # binary_mask = (binary_mask * 255).astype(np.uint8)

                    #########otsu
                    prob_map = mask.squeeze().cpu().numpy()  # 移除批次和通道维度，转为 numpy
                    # 将概率图转为 0~255 的灰度图（Otsu 需要整数输入）
                    prob_map_uint8 = (prob_map * 255).astype(np.uint8)  # 转换为 [0,255] 的 uint8 图像
                    # 调用 Otsu 算法获取最佳阈值（基于概率图的分布）
                    threshold = otsu_threshold(prob_map_uint8)  # 返回 0~255 的整数阈值
                    threshold_normalized = threshold / 255.0    # 归一化到 0~1 范围（与概率图对齐）
                    # --- 生成二值化 mask ---
                    # 用 Otsu 阈值代替固定阈值 0.5
                    binary_mask = (prob_map > threshold_normalized).astype(np.uint8)  # 直接基于概率图比较
                    binary_mask = (binary_mask * 255).astype(np.uint8)
                    # 后处理
                    binary_mask = morphological_operations(binary_mask)
                    # 保存掩码
                    Image.fromarray(binary_mask).save(output_path)
                    f_txt.write(f"{mask_filename},{confidences[i]:.4f}\n")
    # 等待所有进程完成
    torch.distributed.barrier()
    # 主进程合并所有临时文件
    if gpu == 0:
        with open(args.output_txt_path, 'w') as outfile:
            first_file = True
            for i in range(Config.world_size):
                temp_path = f"{args.output_txt_path}.gpu{i}"
                with open(temp_path, 'r') as infile:
                    content = infile.read()
                    # 去除内容末尾的换行符（如果有）
                    if content.endswith('\n'):
                        content = content[:-1]
                    # 除了第一个文件外，每个文件前添加一个换行符
                    if not first_file:
                        outfile.write('\n')
                    outfile.write(content)
                    first_file = False
                os.remove(temp_path)  # 删除临时文件
    
    # 销毁进程组
    destroy_process_group()
    remove_duplicates(args.output_txt_path)

if __name__ == "__main__":
    # 设置环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # 创建多个进程
    mp.spawn(test, args=(Config.num_gpus,), nprocs=Config.num_gpus, join=True)