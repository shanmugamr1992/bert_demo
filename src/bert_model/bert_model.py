from torch import nn

from bert_model.encoder import Encoder
from bert_model.output_tasks.masked_language_model import MaskedLanguageModel
from bert_model.output_tasks.next_sentance_prediction import NextSentancePrediction 

class BertModel(nn.Module):
    def __init__(self, vocab_size, seq_len = 64, num_token_types=3, num_layers=12, embedding_dim = 768, num_attention_heads = 12, expansion_factor=4):
        super().__init__()
        self.encoder = Encoder(vocab_size, seq_len, num_token_types, num_layers, embedding_dim, num_attention_heads, expansion_factor)
        self.masked_language_model = MaskedLanguageModel(vocab_size, embedding_dim) 
        self.next_sentance_prediction = NextSentancePrediction(embedding_dim)

    def forward(self, input_tokens, input_token_types, target_mlm_labels, target_nsp_labels):
        encoded_representation = self.encoder(input_tokens, input_token_types)
        nsp_logits, nsp_loss = self.next_sentance_prediction(encoded_representation, target_nsp_labels)
        mlm_logits, mlm_loss = self.masked_language_model(encoded_representation, target_mlm_labels)
        total_loss = mlm_loss + nsp_loss
        return nsp_logits, mlm_logits, total_loss