from torch import nn

from bert_model.transformer.layers.embedding_layer import BertEmbeddingLayer
from bert_model.transformer.transformer_block import TransformerBlock 

class Encoder(nn.Module):
    def __init__(self, vocab_size, seq_len = 64, num_token_types = 3, num_layers = 12, embedding_dim = 768, num_attention_heads = 12, expansion_factor = 4):
        super().__init__()
        self.embedding_layer = BertEmbeddingLayer(vocab_size, seq_len, embedding_dim, num_token_types)
        self.transformer_blocks = TransformerBlock(num_layers, embedding_dim, num_attention_heads, expansion_factor)

    def forward(self, inp_tokens, input_token_types):
        seq_len = inp_tokens.size(1)
        padding_mask = (inp_tokens > 0).unsqueeze(1).repeat(1, seq_len, 1).unsqueeze(1)
        embeddings = self.embedding_layer(inp_tokens, input_token_types)
        encoded_representation = self.transformer_blocks(embeddings, padding_mask)
        return encoded_representation