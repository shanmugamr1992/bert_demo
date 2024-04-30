from torch import nn

from bert_model.transformer.layers.embedding_layer import BertEmbeddingLayer
from bert_model.transformer.transformer_block import TransformerBlock 

class Encoder(nn.Module):
    def __init__(self, vocab_size, seq_len, num_token_types, num_layers, embedding_dim, num_attention_heads, expansion_factor = 4):
        super().__init__()
        self.embedding_layer = BertEmbeddingLayer(vocab_size, seq_len, embedding_dim, num_token_types)
        self.transformer_blocks = TransformerBlock(num_layers, embedding_dim, num_attention_heads, expansion_factor)

    def forward(self, inp_tokens, input_token_types):
        """
        Args:
            inp_tokens : Input token ids of shape [bs, seq_len]
            input_token_types : The token segements of whether it belongs to sentance  or 2 or padding(1 1 1 2 2 2 0 0 0 ) of shape [bs, seq_len]

        Returns:
            encoded_representation of the input of shape [bs, seq_len, embedding_dim]
        """
        seq_len = inp_tokens.size(1)
        # All padded areas are set to False [bs, seq_len] -> [bs , 1, seq_len] -> [bs, 1, seq_len, seq_len]
        padding_mask = (inp_tokens > 0).unsqueeze(1).repeat(1, seq_len, 1).unsqueeze(1)

        embeddings = self.embedding_layer(inp_tokens, input_token_types)
        encoded_representation = self.transformer_blocks(embeddings, padding_mask)
        return encoded_representation