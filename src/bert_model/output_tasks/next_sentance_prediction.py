from torch import nn 
import torch 

class NextSentancePrediction(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.linear_layer = nn.Linear(embedding_dim, 2)

    def forward(self, encoded_representation, target_labels):
        """
        Args:
            encoded_representation: Output of the encoder [bs, seq_len , embedding_dim]
            target_labels : 1s and 0s representing if the second sentance is the correct sentance or not [bs]

        Returns:
            nsp_logits of shape [bs, seq_len, 2], nlp_loss (float)
        """
        cls_token_representation = encoded_representation[:,0,:]
        nsp_logits = self.linear_layer(cls_token_representation)
        nsp_loss = torch.nn.CrossEntropyLoss()(nsp_logits, target_labels)
        return nsp_logits, nsp_loss