from torch import nn 

class MaskedLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim = 768):
        super().__init__()
        self.linear_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, encoded_representation, target_labels):
        bs, seq_len , embedding_dim = encoded_representation.shape
        target_labels_flattened = target_labels.view(bs*seq_len)
        mlm_logits = self.linear_layer(encoded_representation)
        logits_flattenend = mlm_logits.view((bs*seq_len, -1))
        mlm_loss = nn.CrossEntropyLoss(ignore_index=0)(logits_flattenend, target_labels_flattened) 
        return mlm_logits, mlm_loss 