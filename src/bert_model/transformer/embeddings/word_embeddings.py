from torch import nn
class WordEmbeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim = 768):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

    def forward(self, input):
        return self.word_embeddings(input)