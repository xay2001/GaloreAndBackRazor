import torch
import torch.nn as nn
import torch.nn.functional as F

from Mesa.mesa import custom_quant
from Mesa.mesa import native
from Mesa.mesa import packbit

from .sparse_matrix import sparsify, unsparsify

class silu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, quantize=False, half=False, clip_val=None, level=256, iteration=None, ema_decay=None, quant_groups=None, shift=None):
        shape_x, mask_x, sparse_x = sparsify(x, mask)
        
        if half and (not quantize):
            sparse_x = sparse_x.half()
            
        if quantize:
            custom_quant.Quant.forward(ctx, sparse_x, clip_val, level, iteration, ema_decay, quant_groups, shift)
            ctx.save_for_backward(shape_x, mask_x)
        else:
            ctx.save_for_backward(shape_x, mask_x, sparse_x)
            
        # 使用SiLU作为前向传播函数
        y = F.silu(x)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        tensors = ctx.saved_tensors
        
        if len(tensors) == 2:
            shape_x, mask_x = tensors
            sparse_x = custom_quant.Quant.restore(ctx)
        else:
            shape_x, mask_x, sparse_x = tensors
            
        sparse_x = sparse_x.float()
        x = unsparsify(shape_x, mask_x, sparse_x)
        
        # 手动计算SiLU的梯度
        sigmoid = torch.sigmoid(x)
        grad_input = grad_output * (sigmoid * (1 + x * (1 - sigmoid)))
        
        return grad_input, None, None, None, None, None, None, None, None, None

class SiLUSparse(nn.SiLU, custom_quant.Quant):
    def __init__(self, args=None, logger=None, quant_groups=1, masker=None, quantize=False, half=False):
        super(SiLUSparse, self).__init__()
        self.masker = masker
        self.quantize = quantize
        self.half = half
        custom_quant.Quant.__init__(self, args=args, logger=logger, quant_groups=quant_groups)
        self.tag = 'silu'

    def __repr__(self):
        return self.__str__()

    def forward(self, x):
        if self.masker is not None and self.training:
            mask = self.masker(x)
            y = silu.apply(x, mask, self.quantize, self.half, self.clip_val, self.level,
                          self.iteration, self.ema_decay, self.quant_groups, self.shift)
        else:
            y = F.silu(x)
        return y

if __name__ == "__main__":
    model = SiLUSparse()
    print(model)
    model.enable = True
    print(model)