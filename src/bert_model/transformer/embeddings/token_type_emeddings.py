from torch import nn
class TokenTypelEmbeddings(nn.Module):
    def __init__(self, num_token_types, embedding_dim):
        super().__init__()
        self.token_type_embeddings = nn.Embedding(num_token_types, embedding_dim, padding_idx=0) # TokenizEr pad id

    def forward(self, input):
        return self.token_type_embeddings(input)