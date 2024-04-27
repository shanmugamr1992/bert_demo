import math
import torch 
from torch import nn 

class CoreDotProductAttention(nn.Module):
    def __init__(self, num_attention_heads = 8, embedding_dim = 768, dropout=0.1):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.dropout = torch.nn.Dropout(dropout)
        self.output_linear = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, q, k, v, padding_mask):
        bs, seq_len, embed_dim = q.shape
        num_elem_per_attn_head = embed_dim//self.num_attention_heads

        qr = q.view(bs, seq_len, self.num_attention_heads, -1).permute(0,2,1,3)
        vr = v.view(bs, seq_len, self.num_attention_heads, -1).permute(0,2,1,3)
        kr = k.view(bs, seq_len, self.num_attention_heads, -1).permute(0,2,1,3)

        attn_scores = torch.matmul(qr, kr.permute(0,1,3,2))/math.sqrt(num_elem_per_attn_head)
        #ATTN SCORES SHAPE torch.Size([16, 12, 64, 64]) , PADDING MASK SHAPE torch.Size([16, 1, 64, 64])
        attn_scores_masked = attn_scores.masked_fill(padding_mask == 0, -1e9)
        attn_scores_normalized = torch.nn.functional.softmax(attn_scores_masked, dim=-1)
        attn_scores_regularized = self.dropout(attn_scores_normalized)
                        
        attn_output = torch.matmul(attn_scores_regularized, vr).permute(0,2,1,3).contiguous().view(bs, seq_len, -1)
        return self.output_linear(attn_output)