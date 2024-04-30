from torch import nn

from .layers.transformer_layer import TransformerLayer 


class TransformerBlock(nn.Module):
    def __init__(self, num_layers, embedding_dim, num_attention_heads, expansion_factor = 4):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(embedding_dim, num_attention_heads, expansion_factor) for _ in range(num_layers)
                       ])
          
    def forward(self, embededings, padding_mask):
        """
        Args:
            embededings : Output of the embedding layer of shape [bs, seq_len, embedding_dim]
            padding_mask  The mask applied during attention of shape [bs, 1, seq_len, seq_len]

        Returns:
            encoded_representation of the input of shape [bs, seq_len, embedding_dim]
        """
        x = embededings 

        for l in range(1, len(self.layers)):
            x = self.layers[l](x, padding_mask)

        encoded_representation = x   
        return encoded_representation