from torch import nn 

class Projection(nn.Module):
    def __init__(self, embedding_dim = 768):
        super().__init__()
        self.queries = nn.Linear(embedding_dim, embedding_dim)
        self.keys = nn.Linear(embedding_dim, embedding_dim)
        self.values = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, input_embeddings):
        """ 
        Args:
            input_embeddings : Output of the previous transformer block of shape [bs, seq_len, embedding_dim]

        Returns:
            queries, keys, values of shape [bs, seq_len, embedding_dim]
        """        
        return self.queries(input_embeddings), self.keys(input_embeddings), self.values(input_embeddings)
