from torch import nn

from .layers.transformer_layer import TransformerLayer 


class TransformerBlock(nn.Module):
    def __init__(self, num_layers = 8, embedding_dim = 768, num_attention_heads = 12, expansion_factor = 4):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(embedding_dim, num_attention_heads, expansion_factor) for _ in range(num_layers)
                       ])
          
    def forward(self, x, padding_mask):
        for l in range(1, len(self.layers)):
            x = self.layers[l](x, padding_mask)
        return x