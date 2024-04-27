from torch import nn

from .core_dot_product_attention import CoreDotProductAttention
from .projection import Projection 

class Attention(nn.Module): 
    def __init__(self,  embedding_dim = 768, num_attention_heads = 8, dropout = 0.1):
        super().__init__()
        self.projection = Projection(embedding_dim)
        self.attention = CoreDotProductAttention(num_attention_heads, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask):
        queries, keys, values = self.projection(x)
        attention_ouptut = self.attention(queries, keys, values, padding_mask)
        attention_output_regularized = self.dropout(attention_ouptut)
        return attention_output_regularized