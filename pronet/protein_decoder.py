import torch
import torch.nn as nn
import math
#6

class BlockCausalDecoder(nn.Module):
    def __init__(self, input_dim=128, d_model=256, n_layers=6, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 解码器嵌入（每个位置独立预测）
        self.pos_embed = PositionalEncoding(d_model)
        
        # 解码器层
        self.layers = nn.ModuleList([
            BlockCausalLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        
        # 输出层
        self.fc_out = nn.Linear(d_model, 20)
        
    def forward(self, x, lengths):
        """
        Args:
            x: (sum(len_i), 128) 输入向量
            lengths: [len1, len2, ..., len32] 每个样本的长度
        """
        # 投影输入到d_model
        x_proj = self.input_proj(x)  # (sum(len_i), d_model)
        
        # 添加位置编码
        x_embed = self.pos_embed(x_proj.unsqueeze(0)).squeeze(0)  # (sum(len_i), d_model)
        
        # 生成块级因果掩码
        device = x.device
        total_len = x.size(0)
        mask = self._generate_block_causal_mask(lengths, total_len).to(device)
        
        # 逐层处理
        for layer in self.layers:
            x_embed = layer(x_embed, mask)
        
        # 输出映射到氨基酸概率
        return self.fc_out(x_embed)  # (sum(len_i), 20)
    
    def _generate_block_causal_mask(self, lengths, total_len):
        """生成块级因果掩码矩阵"""
        mask = torch.full((total_len, total_len), float('-inf'), dtype=torch.float32)
        start = 0
        for l in lengths:
            end = start + l
            # 每个块内的因果掩码（允许关注自身及之前的token）
            mask[start:end, start:end] = torch.triu(
                torch.ones(l, l) * float('-inf'), diagonal=1
            )
            start = end
        return mask

class BlockCausalLayer(nn.Module):
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = PositionWiseFeedForward(d_model, d_model * 4)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # 自注意力（输入为(sum(len_i), d_model)）
        x = x.unsqueeze(1)  # (sum(len_i), 1, d_model)
        attn_out, _ = self.self_attn(
            query=x, key=x, value=x, 
            attn_mask=mask, 
            need_weights=False
        )
        x = x + self.dropout(attn_out)
        x = self.norm1(x.squeeze(1))
        
        # 前馈网络
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)
        return x

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        return self.linear2(self.gelu(self.linear1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(1), :]
# # 初始化
# model = BlockCausalDecoder(input_dim=128, d_model=256)

# # 输入数据
# lengths = [5, 8, 3]  # 3个样本，总长度为5+8+3=16
# x = torch.randn(16, 128)  # (sum(len_i), 128)

# # 前向计算
# output = model(x, lengths)  # 输出形状 (16, 20)

# # 生成氨基酸索引序列
# amino_indices = torch.argmax(output, dim=1)  # (16,)