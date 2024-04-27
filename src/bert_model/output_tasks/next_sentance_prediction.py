from torch import nn 
import torch 

class NextSentancePrediction(nn.Module):
    def __init__(self, embedding_dim = 768):
        super().__init__()
        self.linear_layer = nn.Linear(embedding_dim, 2)

    def forward(self, encoded_representation, target_labels):
        cls_token_representation = encoded_representation[:,0,:]
        nsp_logits = self.linear_layer(cls_token_representation)
        nsp_loss = torch.nn.CrossEntropyLoss()(nsp_logits, target_labels)
        return nsp_logits, nsp_loss