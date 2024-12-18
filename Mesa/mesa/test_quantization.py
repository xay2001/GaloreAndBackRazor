import torch

def quantize_bfloat16_to_int8(tensor_bf16):
    # 确保输入张量为 bfloat16 类型
    assert tensor_bf16.dtype == torch.bfloat16, "输入张量必须为 bfloat16 类型"

    # 计算最小值和最大值
    min_val = tensor_bf16.min()
    max_val = tensor_bf16.max()

    # 计算缩放因子
    scale = (max_val - min_val) / 255.0  # int8 范围为 0-255

    # 量化
    tensor_int8 = ((tensor_bf16 - min_val) / scale).round().to(torch.int8)

    return tensor_int8, min_val, scale

def dequantize_int8_to_bfloat16(tensor_int8, min_val, scale):
    # 反量化
    tensor_bf16 = (tensor_int8.to(torch.float32) * scale + min_val).to(torch.bfloat16)

    return tensor_bf16

# 示例
tensor_bf16 = torch.randn(1000, dtype=torch.bfloat16)  # 创建一个随机 bfloat16 张量
tensor_int8, min_val, scale = quantize_bfloat16_to_int8(tensor_bf16)
tensor_bf16_recovered = dequantize_int8_to_bfloat16(tensor_int8, min_val, scale)

# 验证恢复的张量与原始张量的差异
print("原始张量与恢复张量的均方误差：", torch.mean((tensor_bf16 - tensor_bf16_recovered) ** 2).item())
