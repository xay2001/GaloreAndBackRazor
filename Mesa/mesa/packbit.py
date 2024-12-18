# # Copyright (c) 2021-present, Zhuang AI Group.
# # All rights reserved.

# import torch

# def packbits_padded(tensor, dim = -1, mask = 0b1, out = None, dtype = torch.uint8):
#     # print("调用了压缩函数")
#     # 显示调用 packbits_padded 前的显存占用情况
#     print(f"调用 packbits_padded 前: 已分配显存: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB, 已保留显存: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
#     print(f"输入 tensor 的数据类型: {tensor.dtype}")
#     dim = dim if dim >= 0 else dim + tensor.dim()
#     nbits_element, nbits = 8, (1 if mask == 0b00000001 else 2 if mask == 0b00000011 else 4 if mask == 0b00001111 else 8 if mask == 0b11111111 else None)
#     nibbles = nbits_element // nbits
#     assert tensor.shape[dim] % nibbles == 0, "shape: {}, dim: {}, nibbles: {}".format(tensor.shape, dim, nibbles)
    
#     out = out if out is not None else torch.empty(*tensor.shape[:dim], tensor.shape[dim] // nibbles, *tensor.shape[1 + dim:], dtype = dtype, device = tensor.device)
#     shift = torch.arange(nbits_element - nbits, -1, -nbits, dtype = torch.uint8, device = tensor.device)
#     shift = shift.view(nibbles, *((1,) * (tensor.dim() - dim - 1)))
#     torch.sum(tensor.view(*tensor.shape[:dim], -1, nibbles, *tensor.shape[1 + dim:]) << shift , dim = 1 + dim, out = out)
#     # 显示调用 packbits_padded 后的显存占用情况
#     print(f"调用 packbits_padded 后: 已分配显存: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB, 已保留显存: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
#     print(f"输出 out 的数据类型: {out.dtype}")
#     return out

    
# # def packbits_padded(tensor, dim=-1, mask=0b1, out=None, dtype=torch.uint8):
# #     print("调用了压缩函数")
# #     # 显示调用 packbits_padded 前的显存占用情况
# #     print(f"调用 packbits_padded 前: 已分配显存: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB, 已保留显存: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
# #     print(f"输入 tensor 的数据类型: {tensor.dtype}")
    
# #     dim = dim if dim >= 0 else dim + tensor.dim()
# #     nbits_element, nbits = 8, (1 if mask == 0b00000001 else 2 if mask == 0b00000011 else 4 if mask == 0b00001111 else 8 if mask == 0b11111111 else None)
# #     nibbles = nbits_element // nbits
# #     assert tensor.shape[dim] % nibbles == 0, "shape: {}, dim: {}, nibbles: {}".format(tensor.shape, dim, nibbles)
    
# #     # 创建 out 张量
# #     out = out if out is not None else torch.empty(*tensor.shape[:dim], tensor.shape[dim] // nibbles, *tensor.shape[1 + dim:], dtype=dtype, device=tensor.device)
# #     print(f"创建 out 后: 已分配显存: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB, 已保留显存: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
    
# #     # 创建 shift 张量
# #     shift = torch.arange(nbits_element - nbits, -1, -nbits, dtype=torch.uint8, device=tensor.device)
# #     print(f"创建 shift 后: 已分配显存: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB, 已保留显存: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")

# #     # 调整 shift 的形状
# #     shift = shift.view(nibbles, *((1,) * (tensor.dim() - dim - 1)))
# #     print(f"调整 shift 形状后: 已分配显存: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB, 已保留显存: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")

# #     # 计算压缩结果
# #     torch.sum(tensor.view(*tensor.shape[:dim], -1, nibbles, *tensor.shape[1 + dim:]) << shift, dim=1 + dim, out=out)
# #     print(f"执行压缩操作后: 已分配显存: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB, 已保留显存: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")

# #     # 显示调用 packbits_padded 后的显存占用情况
# #     print(f"调用 packbits_padded 后: 已分配显存: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB, 已保留显存: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
# #     print(f"输出 out 的数据类型: {out.dtype}")
# #     return out

# def unpackbits_padded(tensor, dim = -1, mask = 0b1, out = None):
#     print("调用了解压函数")
#     # 显示调用 unpackbits_padded 前的显存占用情况
#     print(f"调用 unpackbits_padded 前: 已分配显存: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB, 已保留显存: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
#     print(f"输入 tensor 的数据类型: {tensor.dtype}")
#     dim = dim if dim >= 0 else dim + tensor.dim()
#     nbits_element, nbits = 8, (1 if mask == 0b00000001 else 2 if mask == 0b00000011 else 4 if mask == 0b00001111 else 8 if mask == 0b11111111 else None)
#     nibbles = nbits_element // nbits
    
#     out = out if out is not None else \
#             torch.empty(*tensor.shape[:dim], tensor.shape[dim] * nibbles, *tensor.shape[1 + dim:], dtype = torch.uint8, device = tensor.device)
#     shift = torch.arange(nbits_element - nbits, -1, -nbits, dtype = torch.uint8, device = tensor.device)
#     shift = shift.view(nibbles, *((1,) * (tensor.dim() - dim - 1)))
#     torch.bitwise_and((tensor.unsqueeze(1 + dim) >> shift).view_as(out), mask, out = out)
#     # 显示调用 unpackbits_padded 后的显存占用情况
#     print(f"调用 unpackbits_padded 后: 已分配显存: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB, 已保留显存: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
#     print(f"输出 out 的数据类型: {out.dtype}")
#     return out



# Copyright (c) 2021-present, Zhuang AI Group.
# All rights reserved.

import torch

def packbits_padded(tensor, dim = -1, mask = 0b1, out = None, dtype = torch.uint8):
    dim = dim if dim >= 0 else dim + tensor.dim()
    nbits_element, nbits = 8, (1 if mask == 0b00000001 else 2 if mask == 0b00000011 else 4 if mask == 0b00001111 else 8 if mask == 0b11111111 else None)
    nibbles = nbits_element // nbits
    assert tensor.shape[dim] % nibbles == 0, "shape: {}, dim: {}, nibbles: {}".format(tensor.shape, dim, nibbles)
    
    out = out if out is not None else torch.empty(*tensor.shape[:dim], tensor.shape[dim] // nibbles, *tensor.shape[1 + dim:], dtype = dtype, device = tensor.device)
    shift = torch.arange(nbits_element - nbits, -1, -nbits, dtype = torch.uint8, device = tensor.device)
    shift = shift.view(nibbles, *((1,) * (tensor.dim() - dim - 1)))
    torch.sum(tensor.view(*tensor.shape[:dim], -1, nibbles, *tensor.shape[1 + dim:]) << shift , dim = 1 + dim, out = out)
    return out

def unpackbits_padded(tensor, dim = -1, mask = 0b1, out = None):
    dim = dim if dim >= 0 else dim + tensor.dim()
    nbits_element, nbits = 8, (1 if mask == 0b00000001 else 2 if mask == 0b00000011 else 4 if mask == 0b00001111 else 8 if mask == 0b11111111 else None)
    nibbles = nbits_element // nbits
    
    out = out if out is not None else \
            torch.empty(*tensor.shape[:dim], tensor.shape[dim] * nibbles, *tensor.shape[1 + dim:], dtype = torch.uint8, device = tensor.device)
    shift = torch.arange(nbits_element - nbits, -1, -nbits, dtype = torch.uint8, device = tensor.device)
    shift = shift.view(nibbles, *((1,) * (tensor.dim() - dim - 1)))
    torch.bitwise_and((tensor.unsqueeze(1 + dim) >> shift).view_as(out), mask, out = out)
    return out

