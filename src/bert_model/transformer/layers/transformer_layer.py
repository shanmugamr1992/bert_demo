from torch import nn

from ..attention.attention import Attention
from ..fully_connected.fully_connected import FullyConnected 

class TransformerLayer(nn.Module):
    def __init__(self, embedding_dim = 768, num_attention_heads = 8, expansion_factor = 4, dropout = 0.1):
        super().__init__()
        self.attention = Attention(embedding_dim, num_attention_heads, dropout)
        self.layernorm1 = nn.LayerNorm(embedding_dim)
        self.fully_connected = FullyConnected(embedding_dim, expansion_factor)
        self.layernorm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask):
        attention_output_regularized = self.attention(x, padding_mask)
        residual_1 = x + attention_output_regularized
        layer_norm_1 = self.layernorm1(residual_1)
        fully_connected_output = self.fully_connected(layer_norm_1)
        fully_connected_output_regularized = self.dropout(fully_connected_output)
        residual_2 = layer_norm_1 + fully_connected_output_regularized
        final_output = self.layernorm2(residual_2)
        return final_output