from torch import nn

from ..attention.attention import Attention
from ..fully_connected.fully_connected import FullyConnected 

class TransformerLayer(nn.Module):
    def __init__(self, embedding_dim, num_attention_heads, expansion_factor = 4, dropout = 0.1):
        super().__init__()
        self.attention = Attention(embedding_dim, num_attention_heads, dropout)
        self.layernorm1 = nn.LayerNorm(embedding_dim)
        self.fully_connected = FullyConnected(embedding_dim, expansion_factor)
        self.layernorm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask):
        """ 
        Args:
            x : Output of the previous transformer layer / embedding layer of shape [bs, seq_len, embedding_dim]
            padding_mask : The mask applied during attention of shape [bs, 1, seq_len, seq_len]

        Returns:
            Output of the transformer layer  of shape [bs, seq_len, embedding_dim]        
        """
        attention_output_regularized = self.attention(x, padding_mask)
        residual_1 = x + attention_output_regularized
        layer_norm_1 = self.layernorm1(residual_1)
        fully_connected_output = self.fully_connected(layer_norm_1)
        fully_connected_output_regularized = self.dropout(fully_connected_output)
        residual_2 = layer_norm_1 + fully_connected_output_regularized
        final_output = self.layernorm2(residual_2)
        return final_output