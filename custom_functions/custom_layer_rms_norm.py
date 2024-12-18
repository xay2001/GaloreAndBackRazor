import torch
import torch.nn as nn
import torch.nn.functional as F

from Mesa.mesa import custom_quant
from Mesa.mesa import native
from Mesa.mesa import packbit

from .sparse_matrix import sparsify, unsparsify

class rms_norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, mask, quantize, half, eps, clip_val=None, level=256, iteration=None, ema_decay=None, quant_groups=None, shift=None):
        if x.dtype != weight.data.dtype:
            x = x.to(dtype=weight.data.dtype)

        shape_x, mask_x, sparse_x = sparsify(x, mask)

        if half and (not quantize):
            sparse_x = sparse_x.half()

        if quantize:
            custom_quant.Quant.forward(ctx, sparse_x, clip_val, level, iteration, ema_decay, quant_groups, shift)
            ctx.save_for_backward(shape_x, mask_x)
        else:
            ctx.save_for_backward(shape_x, mask_x, sparse_x)

        # 恢复稀疏矩阵
        x = unsparsify(shape_x, mask_x, sparse_x, with_batch_size=True)

        # 计算 RMSNorm
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + eps)

        # 如果需要，将精度转换为 half
        if weight.dtype in [torch.float16, torch.bfloat16]:
            x_norm = x_norm.to(weight.dtype)

        y = weight * x_norm

        # 保存必要的参数以供反向传播
        ctx.rms_norm_parameters = (x_norm, variance, weight, eps)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        tensors = ctx.saved_tensors

        if len(tensors) == 2:
            shape_x, mask_x = tensors
            grad_output = grad_output.contiguous()
            sparse_x = custom_quant.Quant.restore(ctx)
        else:
            shape_x, mask_x, sparse_x = tensors

        sparse_x = sparse_x.float()
        x = unsparsify(shape_x, mask_x, sparse_x, with_batch_size=True)

        x_norm, variance, weight, eps = ctx.rms_norm_parameters

        grad_input = grad_weight = None

        # 计算梯度
        grad_output = grad_output * weight

        grad_variance = (grad_output * x * (-0.5) * torch.pow(variance + eps, -1.5)).sum(dim=-1, keepdim=True)
        grad_input = grad_output * torch.rsqrt(variance + eps) + grad_variance * 2 * x / x.size(-1)

        grad_weight = (grad_output * x_norm).sum(dim=0)

        return grad_input, None, grad_weight, None, None, None, None, None, None, None, None, None, None, None

class RMSNormSparse(nn.Module, custom_quant.Quant):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, args=None, logger=None, quant_groups=1,
                 masker=None, quantize=False, half=False, backrazor_bits=32):
        super(RMSNormSparse, self).__init__()
        custom_quant.Quant.__init__(self, args=args, logger=logger, quant_groups=quant_groups)
        self.tag = 'rmsnorm'
        self.masker = masker
        self.quantize = quantize
        self.half = half
        self.eps = eps

        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))

    def __repr__(self):
        return self.__str__()

    def forward(self, x):
        if self.masker is not None and self.training:
            mask = self.masker(x)
            y = rms_norm.apply(x, self.normalized_shape, self.weight, mask, self.quantize, self.half, self.eps,
                                self.clip_val, self.level,
                                self.iteration, self.ema_decay, self.quant_groups, self.shift)
        else:
            # 计算 RMSNorm
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            x_norm = x * torch.rsqrt(variance + self.eps)

            # 如果需要，将精度转换为 half
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                x_norm = x_norm.to(self.weight.dtype)

            y = self.weight * x_norm
        return y
 