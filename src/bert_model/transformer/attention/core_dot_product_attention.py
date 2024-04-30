import math
import torch 
from torch import nn 

class CoreDotProductAttention(nn.Module):
    def __init__(self, num_attention_heads, embedding_dim, dropout=0.1):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.dropout = torch.nn.Dropout(dropout)
        self.output_linear = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, q, k, v, padding_mask):
        """
        Args:
            q : Queries of shape [bs, seq_len, embedding_dim]
            k : Keys of shape [bs, seq_len, embedding_dim]
            v : Values of shape [bs, seq_len, embedding_dim]
            padding_mask : Mask of shape [bs, 1, seq_len, seq_len]

        Returns:
            Attention otuput of shape [bs , seq_len, embedding_dim]
        """
        bs, seq_len, embed_dim = q.shape
        num_elem_per_attn_head = embed_dim//self.num_attention_heads
        
        # [bs, seq_len, embedding_dim] -> [bs, seq_len, attn_head, dim_per_head] -> [bs, attn_head, seq_len, dim_per_head]
        qr = q.view(bs, seq_len, self.num_attention_heads, -1).permute(0,2,1,3)
        vr = v.view(bs, seq_len, self.num_attention_heads, -1).permute(0,2,1,3)
        kr = k.view(bs, seq_len, self.num_attention_heads, -1).permute(0,2,1,3)

        # qr : [bs, attn_head, seq_len, dim_per_head,] , kr.permute(0,1,3,2) : [bs, attn_head, dim_per_head, seq_len]
        attn_scores = torch.matmul(qr, kr.permute(0,1,3,2))/math.sqrt(num_elem_per_attn_head)

        # attn scores [bs, attn_head, seq_len, seq_len] padding mask [bs, 1, seq_len, seq_len]
        attn_scores_masked = attn_scores.masked_fill(padding_mask == 0, -1e9)

        # Normalized across each row (i.e ) Each row sums up to 1
        attn_scores_normalized = torch.nn.functional.softmax(attn_scores_masked, dim=-1)
        attn_scores_regularized = self.dropout(attn_scores_normalized)

        #  attn_scores_regularized [bs, 1, seq_len, seq_len], vr : [bs, attn_head, seq_len, dim_per_head]    
        # Product [bs, attn_head, seq_len, dim_per_head] -> .permute(0,2,1,3) [bs, seq_len, attn_head, dim_per_head]  
        # View -> [bs , seq_len, embedding_dim] 
        attn_output = torch.matmul(attn_scores_regularized, vr).permute(0,2,1,3).contiguous().view(bs, seq_len, -1)
        return self.output_linear(attn_output)