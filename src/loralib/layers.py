import jittor as jt
from jittor import nn as jt_nn

import math
from typing import Optional, List

class LoRALayer():
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = jt_nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

class Embedding(jt_nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs
    ):
        # mention the super class's input parameters are only num_embeddings, embedding_dim, padding_idx and dtype
        jt_nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = jt.init.constant((r, num_embeddings))
            self.lora_B = jt.init.constant((embedding_dim, r))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            with jt.no_grad():
                self.weight.requires_grad = False
            self.reset_parameters()

    def reset_parameters(self):
        # Initialize the weights of the embedding layer
        jt.init.uniform_(self.weight, -0.5, 0.5)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            jt.init.zero_(self.lora_A)
            jt.init.gauss_(self.lora_B)

    def train(self, mode: bool = True):
        jt_nn.Embedding.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= jt_nn.matmul(self.lora_B, self.lora_A).transpose(0, 1) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += jt_nn.matmul(self.lora_B, self.lora_A).transpose(0, 1) * self.scaling
                self.merged = True

    def forward(self, x: jt.array):
        if self.r > 0 and not self.merged:
            result = jt_nn.Embedding.execute(self, x)
            after_A = self.lora_A.transpose(0, 1)[x]
            result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return jt_nn.Embedding.execute(self, x)

    def execute(self, x):
        return self.forward(x)

class Linear(jt_nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        jt_nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = jt.init.constant((r, in_features))
            self.lora_B = jt.init.constant((out_features, r))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            with jt.no_grad():
                self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight = self.weight.transpose(0, 1)


    def reset_parameters(self):
        # Initialize weight since Linear in jittor does not have reset_parameters method
        jt.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = self.weight.shape
            bound = 1 / math.sqrt(fan_in)
            jt.init.uniform_(self.bias, -bound, bound)
        if hasattr(self, 'lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            jt.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            jt.init.zero_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        jt_nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(jt_nn.matmul(self.lora_B, self.lora_A )) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(jt_nn.matmul(self.lora_B, self.lora_A )) * self.scaling
                self.merged = True

    def forward(self, x: jt.array):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = jt_nn.linear(x, T(self.weight), bias=self.bias)
            result += jt_nn.matmul(jt_nn.matmul(self.lora_dropout(x), self.lora_A.transpose(0, 1)), self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return jt_nn.linear(x, T(self.weight), bias=self.bias)

class MergedLinear(jt_nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        jt_nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = jt.init.constant((r * sum(enable_lora), in_features))
            self.lora_B = jt.init.constant((out_features // len(enable_lora) * sum(enable_lora), r)) # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            with jt.no_grad():
                self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = jt.init.constant((out_features, ), dtype=bool).reshape(len(enable_lora), -1)
            self.lora_ind[jt.array(enable_lora), :] = True
            self.lora_ind = self.lora_ind.reshape(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight = self.weight.transpose(0, 1)


    def reset_parameters(self):
        # Initialize weight since Linear in jittor does not have reset_parameters method
        jt.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = self.weight.shape
            bound = 1 / math.sqrt(fan_in)
            jt.init.uniform_(self.bias, -bound, bound)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            jt_nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            jt_nn.init.zero_(self.lora_B)

    def zero_pad(self, x):
        result = jt_nn.init.constant((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        # 计算 in_channels 和 out_channels
        in_channels = self.lora_A.shape[0]
        out_channels = self.lora_B.shape[0]

        # 创建 Conv1d 对象
        conv1d = jt_nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            groups=sum(self.enable_lora)
        )

        # 设置权重
        conv1d.weight = self.lora_B.unsqueeze(-1)

        # 进行卷积运算
        delta_w = conv1d(self.lora_A.unsqueeze(0)).squeeze(0)

        return T(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        jt_nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data += self.merge_AB() * self.scaling
                self.merged = True

    def forward(self, x: jt.array):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.merged:
            return jt_nn.linear(x, T(self.weight), bias=self.bias)
        else:
            result = jt_nn.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += jt_nn.matmul(self.lora_dropout(x), T(self.merge_AB().transpose(0,1))) * self.scaling
            return result

    def execute(self, x):
        return self.forward(x)

    def __str__(self) -> str:
        bias_str = 'True' if self.bias is not None else 'False'
        dropout_str = f'(lora_dropout): Dropout(p={self.lora_dropout.p}, inplace=False)' if self.lora_dropout else ''
        return f"MergedLinear(\n  in_features={self.in_features}, out_features={self.out_features}, bias={bias_str}\n  {dropout_str}\n)"

class ConvLoRA(jt_nn.Module, LoRALayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = jt.init.constant((r * kernel_size, in_channels * kernel_size))
            self.lora_B = jt.init.constant((out_channels//self.conv.groups*kernel_size, r*kernel_size))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            with jt.no_grad():
                self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            jt_nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            jt_nn.init.zero_(self.lora_B)

    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.conv.weight.data -= jt.matmul(self.lora_B, self.lora_A).reshape(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.conv.weight.data += jt.matmul(self.lora_B, self.lora_A).reshape(self.conv.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(
                x,
                self.conv.weight + jt.matmul(self.lora_B, self.lora_A).reshape(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        return self.conv(x)

class Conv2d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(jt_nn.Conv, *args, **kwargs)

class Conv1d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(jt_nn.Conv1d, *args, **kwargs)

# Can Extend to other ones like this

class Conv3d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv3d, self).__init__(jt_nn.Conv3d, *args, **kwargs)