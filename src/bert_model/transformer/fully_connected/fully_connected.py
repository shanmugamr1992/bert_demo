from torch import nn 

class FullyConnected(nn.Module):
    def __init__(self, embedding_dim = 768, expansion_factor = 4, dropout = 0.1):
        super().__init__()
        intermediate_dim = embedding_dim * expansion_factor 
        self.fc1 = nn.Linear(embedding_dim, intermediate_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_dim, embedding_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation(x)
        return x