import torch
import torch.nn.functional as F
from typing import List, Dict, Union
from mesorch import Mesorch
# 增强型TTA变换配置（修正维度错误并添加新功能）
tta_transforms = [
    # 原图 (基准)
    {
        'preprocess':  lambda x: x,
        'postprocess': lambda x: x
    },
    # 水平翻转 (修正为dim=3)
    {
        'preprocess':  lambda x: x.flip(-1),
        'postprocess': lambda x: x.flip(-1)
    },
    # 垂直翻转
    {
        'preprocess':  lambda x: x.flip(-2),
        'postprocess': lambda x: x.flip(-2)
    },
    # 旋转90度+水平翻转 (组合增强)
    {
        'preprocess':  lambda x: x.rot90(1, [-2, -1]).flip(-1),
        'postprocess': lambda x: x.flip(-1).rot90(3, [-2, -1])
    },
]

def batch_tta_inference(
    gpu,
    model: torch.nn.Module,
    batch_images: torch.Tensor,
    tta_transforms: List[Dict]=tta_transforms,
    fusion_strategy: str = 'mean',
    # activation_fn: str = 'sigmoid'
) -> torch.Tensor:
    """
    批量TTA推理函数 (支持多图像并行处理)
    
    参数:
        model: 预训练模型
        batch_images: 输入图像批次 [B, C, H, W]
        tta_transforms: TTA变换配置列表
        fusion_strategy: 融合策略 ('mean', 'max', 'sum')
        activation_fn: 激活函数 ('sigmoid', 'softmax', 'none')
    
    返回:
        融合后的预测掩码 [B, C, H, W]
    """
    # 预分配内存提升性能
    num_transforms = len(tta_transforms)
    batch_size, _, h, w = batch_images.shape
    all_preds_mask= torch.empty((num_transforms, batch_size, 1, h, w), device=gpu)
    all_preds_label=torch.empty((num_transforms,batch_size))
    # 并行处理所有变换
    for t_idx, transform in enumerate(tta_transforms):
        # 应用预处理
        processed = transform['preprocess'](batch_images)
        # 模型推理
        pred_mask,pred_label = model(processed)
        
        # 应用后处理
        recovered_pred_mask = transform['postprocess'](pred_mask)
        
        # 存储结果
        all_preds_mask[t_idx] = recovered_pred_mask
        all_preds_label[t_idx] = pred_label    
        # 应用激活函数
        # if activation_fn == 'sigmoid':
        #     all_preds_mask = torch.sigmoid(all_preds)
        # elif activation_fn == 'softmax':
        #     all_preds_mask = F.softmax(all_preds, dim=2)
            
        # 多变换融合
    if fusion_strategy == 'mean':
        fused = all_preds_mask.mean(dim=0)
    elif fusion_strategy == 'max':
        fused = all_preds_mask.max(dim=0)[0]
    elif fusion_strategy == 'sum':
        fused = all_preds_mask.sum(dim=0)
    else:
        raise ValueError(f"未知融合策略: {fusion_strategy}")
    pred_label=all_preds_label.mean(dim=0)    
    return fused,pred_label  # 保持批次维度 [B, C, H, W]

# 使用示例 ---------------------------------------------------
if __name__ == "__main__":
    # 模拟输入 (批量大小=8, 256x256 RGB图像)
    dummy_batch = torch.randn(2, 3, 512, 512)
    model = Mesorch()  # 加载你的模型
    
    # 执行批量推理
    mask,label = batch_tta_inference(
        gpu='cuda:0',
        model=model,
        batch_images=dummy_batch,
        tta_transforms=tta_transforms,
        fusion_strategy='mean',
    )
    
    # 结果后处理 (示例：生成二值掩码)
    print("运行成功！")