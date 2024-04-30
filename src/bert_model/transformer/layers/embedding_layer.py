import torch
from torch import nn

from ..embeddings.positional_embedding import PositionalEmbeddings
from ..embeddings.token_type_emeddings import TokenTypelEmbeddings
from ..embeddings.word_embeddings import WordEmbeddings

class BertEmbeddingLayer(nn.Module):
    
    def __init__(self, vocab_size, seq_len, embedding_dim, num_token_types, dropout=0.1):
        super().__init__()
        self.positional_embeddings = PositionalEmbeddings(seq_len, embedding_dim)
        self.token_type_embeddings = TokenTypelEmbeddings(num_token_types, embedding_dim)
        self.word_embeddings = WordEmbeddings(vocab_size, embedding_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input_tokens, input_token_types):
        """
        Args:
            input_tokens : Input token ids of shape [bs, seq_len]
            input_token_types : The token segements of whether it belongs to sentance  or 2 or padding(1 1 1 2 2 2 0 0 0 ) of shape [bs, seq_len]

        Returns:
            embededings of shape [bs, seq_len, embedding_dim]
        """        
        embeddings = self.positional_embeddings(input_tokens) + self.word_embeddings(input_tokens) + self.token_type_embeddings(input_token_types)
        return self.dropout(embeddings)