from torch import nn

from .core_dot_product_attention import CoreDotProductAttention
from .projection import Projection 

class Attention(nn.Module): 
    def __init__(self,  embedding_dim, num_attention_heads, dropout = 0.1):
        super().__init__()
        self.projection = Projection(embedding_dim)
        self.core_attention = CoreDotProductAttention(num_attention_heads, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask):
        """ 
        Args:
            x : Output of the previous transformer block of shape [bs, seq_len, embedding_dim]
            padding_mask : Mask of shape [bs, 1, seq_len, seq_len]

        Returns:
            attention_output_regularized of shape [bs, seq_len, embedding_dim]
        """
        queries, keys, values = self.projection(x)
        attention_ouptut = self.core_attention(queries, keys, values, padding_mask)
        attention_output_regularized = self.dropout(attention_ouptut)
        return attention_output_regularized
    