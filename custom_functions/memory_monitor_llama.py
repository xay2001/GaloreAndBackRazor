import copy
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP
import Mesa.mesa as ms
from custom_functions.custom_fc import LinearSparse
from custom_functions.custom_softmax import SoftmaxSparse  
from custom_functions.custom_gelu import GELUSparse
from custom_functions.custom_layer_norm import LayerNormSparse
from custom_functions.custom_silu import SiLUSparse

__all__ = ['count_model_size', 'count_activation_size', 'profile_memory_cost']

def count_model_size(net, trainable_param_bits=32, frozen_param_bits=8, print_log=True):
    """统计模型参数大小"""
    frozen_param_bits = 32 if frozen_param_bits is None else frozen_param_bits

    trainable_param_size = 0
    frozen_param_size = 0
    for p in net.parameters():
        if p.requires_grad:
            trainable_param_size += trainable_param_bits / 8 * p.numel()
        else:
            frozen_param_size += frozen_param_bits / 8 * p.numel()
    model_size = trainable_param_size + frozen_param_size
    
    if print_log:
        print('Total: %d' % model_size,
              '\tTrainable: %d (data bits %d)' % (trainable_param_size, trainable_param_bits),
              '\tFrozen: %d (data bits %d)' % (frozen_param_size, frozen_param_bits))
    return model_size

def is_leaf(m_):
    """判断是否为叶子节点"""
    return (len(list(m_.children())) == 0 or 
            isinstance(m_, (LinearSparse, SoftmaxSparse, GELUSparse, SiLUSparse, LayerNormSparse)) or
            isinstance(m_, LlamaAttention) or isinstance(m_, LlamaMLP) or
            (len(list(m_.children())) == 1 and isinstance(next(m_.children()), nn.Identity)))

def count_activation_size(net, input_size=(1, 2048), require_backward=True, activation_bits=32):
    """计算激活值大小"""
    act_byte = activation_bits / 8
    model = copy.deepcopy(net)

    def count_linear(m, x, y):
        """计算线性层激活值"""
        if m.weight is not None and m.weight.requires_grad:
            m.grad_activations = torch.Tensor([x[0].numel() * act_byte])
        else:
            m.grad_activations = torch.Tensor([0])

        if isinstance(m, LinearSparse) and m.masker is not None:
            ratio = 0.5 if m.half else 1
            mask = m.masker(x[0])
            m.grad_activations *= mask.float().mean().cpu() * ratio
            if m.quantize:
                m.grad_activations *= 0.25
            m.grad_activations += (mask.numel() / 8)

        m.tmp_activations = torch.Tensor([x[0].numel() * act_byte + y.numel() * act_byte])

    def count_attention(m, x, y):
        """计算注意力层激活值"""
        bsz, seq_len, _ = x[0].shape
        qkv_size = bsz * m.num_heads * seq_len * m.head_dim
        
        # 计算Q,K,V的激活值
        m.grad_activations = torch.Tensor([3 * qkv_size * act_byte])
        
        # 注意力矩阵的激活值
        attn_size = bsz * m.num_heads * seq_len * seq_len
        m.grad_activations += torch.Tensor([attn_size * act_byte])
        
        # 临时激活值包括输入、中间状态和输出
        m.tmp_activations = torch.Tensor([x[0].numel() * act_byte + qkv_size * 3 * act_byte + y.numel() * act_byte])

    def count_mlp(m, x, y):
        """计算MLP层激活值"""
        bsz, seq_len, _ = x[0].shape
        
        # 计算gate和up投影的激活值
        intermediate_size = m.gate_proj.out_features
        m.grad_activations = torch.Tensor([2 * bsz * seq_len * intermediate_size * act_byte])
        
        # 激活函数的激活值
        m.grad_activations += torch.Tensor([bsz * seq_len * intermediate_size * act_byte])
        
        # 临时激活值
        m.tmp_activations = torch.Tensor([x[0].numel() * act_byte + y.numel() * act_byte])

    def count_norm(m, x, y):
        """计算归一化层激活值"""
        if m.weight is not None and m.weight.requires_grad:
            m.grad_activations = torch.Tensor([x[0].numel() * act_byte])
        else:
            m.grad_activations = torch.Tensor([0])
            
        m.tmp_activations = torch.Tensor([x[0].numel() * act_byte])

    def add_hooks(m_):
        """添加hook来统计内存使用"""
        if not is_leaf(m_):
            return

        m_.register_buffer('grad_activations', torch.zeros(1))
        m_.register_buffer('tmp_activations', torch.zeros(1))

        if isinstance(m_, (nn.Linear, LinearSparse)):
            fn = count_linear
        elif isinstance(m_, LlamaAttention):
            fn = count_attention
        elif isinstance(m_, LlamaMLP):
            fn = count_mlp
        elif isinstance(m_, (nn.LayerNorm, LayerNormSparse)):
            fn = count_norm
        else:
            fn = None

        if fn is not None:
            _handler = m_.register_forward_hook(fn)

    model.train()
    model.apply(add_hooks)

    # 为Llama模型准备输入
    input_size = (input_size[0], input_size[1])  # batch_size, seq_length
    x = torch.randint(0, model.config.vocab_size, input_size).to(next(model.parameters()).device)
    
    with torch.no_grad():
        model(x)

    memory_info = {
        'peak_activation_size': torch.zeros(1),
        'grad_activation_size': torch.zeros(1),
    }

    # 收集内存使用信息
    for m in model.modules():
        if hasattr(m, 'grad_activations'):
            memory_info['grad_activation_size'] += m.grad_activations
            current_activation = m.tmp_activations + memory_info['grad_activation_size']
            memory_info['peak_activation_size'] = torch.max(
                memory_info['peak_activation_size'], 
                current_activation
            )

    return (memory_info['peak_activation_size'].item(), 
            memory_info['grad_activation_size'].item())

def profile_memory_cost(net, input_size=(1, 2048), require_backward=True,
                       activation_bits=32, trainable_param_bits=32, 
                       frozen_param_bits=8, batch_size=8):
    """整体内存分析"""
    param_size = count_model_size(net, trainable_param_bits, frozen_param_bits, print_log=True)
    activation_size, grad_activation_size = count_activation_size(
        net, input_size, require_backward, activation_bits
    )

    MB = 1024 * 1024
    print(f"Gradient activation size: {grad_activation_size / MB:.1f} MB")
    
    memory_cost = activation_size * batch_size + param_size
    memory_breakdown = {
        'param_size': param_size,
        'act_size': activation_size,
        'grad_size': grad_activation_size
    }
    
    print(f"Total memory cost: {memory_cost / MB:.1f} MB")
    print(f"Parameter size: {param_size / MB:.1f} MB")
    print(f"Activation size per sample: {activation_size / MB:.1f} MB")
    
    return memory_cost, memory_breakdown
