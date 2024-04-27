import torch
from torch import nn

from ..embeddings.positional_embedding import PositionalEmbeddings
from ..embeddings.token_type_emeddings import TokenTypelEmbeddings
from ..embeddings.word_embeddings import WordEmbeddings

class BertEmbeddingLayer(nn.Module):
    
    def __init__(self, vocab_size, seq_len = 64, embedding_dim = 768, num_token_types = 3, dropout=0.1):
        super().__init__()
        self.positional_embeddings = PositionalEmbeddings(seq_len, embedding_dim)
        self.token_type_embeddings = TokenTypelEmbeddings(num_token_types, embedding_dim)
        self.word_embeddings = WordEmbeddings(vocab_size, embedding_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input_tokens, input_token_types):
        x = self.positional_embeddings(input_tokens) + self.word_embeddings(input_tokens) + self.token_type_embeddings(input_token_types)
        return self.dropout(x)