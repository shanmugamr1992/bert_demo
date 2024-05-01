from torch import nn
import math
import torch 

class PositionalEmbeddings(nn.Module):
    def __init__(self, seq_len, embedding_dim, device='cpu'):
        super().__init__()
        self.seq_len = seq_len
        positional_embedding = torch.zeros((seq_len, embedding_dim)).to(device)
        positional_embedding.require_grad = False # Set this to false since these are not learnable
        
        for pos in range(seq_len):
            for i in range(0, embedding_dim, 2):
                positional_embedding[pos, i] = math.sin(pos / (10000 ** ((2 * i)/embedding_dim)))
                positional_embedding[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/embedding_dim)))
            
        self.positional_embeddings = positional_embedding.unsqueeze(0)

    def forward(self, input):
        return self.positional_embeddings
        