import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple

class PrefixTuning(nn.Module):
    """Layer-wise prefix for encoder or decoder."""

    def __init__(self, config, num_heads, embed_dim):
        super().__init__()
        self.prefix_length = config.prefix_length
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(p=config.prefix_dropout)
        if 'prefix-tuning' in config.efficient_methods:
            self.method = 'prefix-tuning'
            self.prefix_embedding = nn.Embedding(config.prefix_length, embed_dim)
            self.prefix_trans = nn.Sequential(
                nn.Linear(embed_dim, config.prefix_mid_dim),
                nn.ReLU(),
                nn.Linear(config.prefix_mid_dim, 2 * embed_dim),
            )
        else:
            self.method = 'p-tuning-v2'
            self.prefix_embedding = nn.Embedding(config.prefix_length, 2 * embed_dim)

    def forward(self, key_states: torch.Tensor, value_states: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = key_states.size(0)
        device = key_states.device
        
        if self.method == 'prefix-tuning':
            prefix = self.prefix_trans(self.prefix_embedding.weight) # p_l, 2*e
        else: # 'p-tuning-v2'
            prefix = self.prefix_embedding.weight # p_l, 2*e
        prefix = prefix.view(self.prefix_length, 2, self.num_heads, self.head_dim) # p_l, 2, h_n, h_e
        prefix = self.dropout(prefix)
        prefix = prefix.permute([1, 2, 0, 3]) # 2, h_n, p_l, h_e

        if key_states.dim() == 4:
            key_states = torch.cat([prefix[0].expand(bsz, -1, -1, -1), key_states], dim=-2) # 2, h_n, p_l+t_l, h_e
            value_states = torch.cat([prefix[1].expand(bsz, -1, -1, -1), value_states], dim=-2)
        elif key_states.dim() == 5:
            key_states = torch.cat([prefix[0].expand(bsz, key_states.size(1), -1, -1, -1), key_states], dim=-2) # 2, beam_num, h_n, p_l+t_l, h_e
            value_states = torch.cat([prefix[1].expand(bsz, value_states.size(1), -1, -1, -1), value_states], dim=-2)
        if attention_mask is not None:
            prompt_mask = torch.zeros(bsz, 1, attention_mask.size(2), prefix.size(2)).to(device)
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=(-1))
        return key_states, value_states, attention_mask

class LoRALinear(nn.Linear):

    def __init__(self, in_features, out_features, config, bias):
        super().__init__(in_features, out_features, bias=bias)
        self.r = config.lora_r
        self.alpha = config.lora_alpha
        self.scaling = self.alpha / self.r
        self.A = nn.Linear(in_features, self.r, bias=False)
        self.B = nn.Linear(self.r, out_features, bias=False)
        self.dropout = nn.Dropout(p=config.lora_dropout)
        self.A.weight.data.normal_(std=0.02)
        self.B.weight.data.zero_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias) + self.B(self.A(self.dropout(input))) * self.scaling

class Adapter(nn.Module):

    def __init__(self, embed_dim, config):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(embed_dim, config.adapter_mid_dim),
            nn.ReLU(),
            nn.Linear(config.adapter_mid_dim, embed_dim),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + self.adapter(input)
    
class LoRAMergedLinear(nn.Linear):

    def __init__(self, in_features, out_features, config):
        super().__init__(in_features, out_features)
        self.r = config.lora_r
        self.alpha = config.lora_alpha
        self.scaling = self.alpha / self.r
        # self.enable_lora= self.enable_lora
        # self.lora_dropout=config.lora_dropout
        self.enable_lora= [True, False, True]
        self.lora_A = nn.Parameter(
            self.weight.new_zeros((config.lora_r * sum(self.enable_lora), in_features)))
        self.lora_B = nn.Parameter(
            self.weight.new_zeros((out_features // len(self.enable_lora) * sum(self.enable_lora), config.lora_r))
        )  # weights for Conv1D with groups=sum(enable_lora)
        self.lora_ind = self.weight.new_zeros(
            (out_features,), dtype=torch.bool
        ).view(len(self.enable_lora), -1) #torch.Size([3, 768])
        self.lora_ind[self.enable_lora, :] = True
        self.lora_ind = self.lora_ind.view(-1) #变成一个torch.Size([2304])形状的tensor，[ True,  True,  True, False, False, False,  True,  True,  True]
        self.lora_dropout = nn.Dropout(p=config.lora_dropout)
        self.my_reset_parameters()
        self.weight.requires_grad = False # Freezing the pre-trained weight matrix
        self.weight.data = self.weight.data.T

    def my_reset_parameters(self):
        # nn.Linear.reset_parameters(self) #没用了
        # if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def zero_pad(self, x): #中间列pad一半的0
        result = x.new_zeros((*x.shape[:-1], self.out_features)) #相当于把x的最后一个维度换成out features，生成一个全0的tensor
        result = result.view(-1, self.out_features)
        # print(self.lora_ind)
        # print(self.out_features) #2304
        # print(self.lora_ind.shape) #torch.Size([2304])
        # print(result.shape,x.shape) #torch.Size([202, 2304]) torch.Size([1, 202, 1536])
        # print(x.reshape(-1,self.out_features//len(self.enable_lora)*sum(self.enable_lora)).shape)
        # return
        result[:, self.lora_ind] = x.reshape(
            -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        )
        return result.view((*x.shape[:-1], self.out_features))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        
        result = F.linear(input, self.weight.T, bias=self.bias)
        # print(result.shape) #torch.Size([1, 202, 2304])
        after_A = F.linear(self.lora_dropout(input), self.lora_A) 
        # print(input.shape) #torch.Size([1, 202, 768])
        # print(after_A.shape) #torch.Size([1, 202, 8])
        # print(after_A.transpose(-2, -1).shape) #torch.Size([1, 8, 202])
        # print(self.lora_A.shape,self.lora_B.shape) #torch.Size([8, 768]) torch.Size([1536, 4])
        # print(self.lora_B.unsqueeze(-1).shape) #torch.Size([1536, 4, 1])
        after_B = F.conv1d( #torch.Size([1, 202, 1536])
            after_A.transpose(-2, -1), #torch.Size([1, 8, 202]) ——(Batch_Size,In_Channel,Length)
            self.lora_B.unsqueeze(-1), #torch.Size([1536, 4, 1]) ——(Out_Channel,In_Channel/Group,Kernel_Size)
            groups=sum(self.enable_lora) #2
        ).transpose(-2, -1)
        result += self.zero_pad(after_B) * self.scaling
        return result            