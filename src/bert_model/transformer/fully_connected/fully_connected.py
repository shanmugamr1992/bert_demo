from torch import nn 

class FullyConnected(nn.Module):
    def __init__(self, embedding_dim, expansion_factor = 4, dropout = 0.1):
        super().__init__()
        intermediate_dim = embedding_dim * expansion_factor 
        self.fc1 = nn.Linear(embedding_dim, intermediate_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_dim, embedding_dim)
        self.activation = nn.GELU()

    def forward(self, attention_output):
        """
        Args:
            attention_output : Output from the attention block of size [bs, seq_len, embedding_dim]

        Returns:
            fc2_out : Output of the fully connected layer of shape [bs_seq_len, embedding_dim]
        """
        fc1_out = self.activation(self.fc1(attention_output))
        fc1_out_regularized = self.dropout(fc1_out)
        fc2_out = self.fc2(fc1_out_regularized)
        return fc2_out