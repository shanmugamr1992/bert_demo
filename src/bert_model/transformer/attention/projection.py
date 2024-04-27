from torch import nn 

class Projection(nn.Module):
    def __init__(self, embedding_dim = 768):
        super().__init__()
        self.queries = nn.Linear(embedding_dim, embedding_dim)
        self.keys = nn.Linear(embedding_dim, embedding_dim)
        self.values = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, input_embeddings):
        return self.queries(input_embeddings), self.keys(input_embeddings), self.values(input_embeddings)