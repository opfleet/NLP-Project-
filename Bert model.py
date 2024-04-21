import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class HateSpeechClassificationModel(nn.Module):
    """Write your HateSpeechClassification model here"""
    def __init__(self, freeze_bert: bool = False):
        super().__init__()

        self.BERT = AutoModel.from_pretrained("distilbert-base-uncased")
        self.linear_layer = nn.Linear(768, 1)

        if (freeze_bert):
            for parameter in self.BERT.parameters():
                parameter.requires_grad = False
        #raise NotImplementedError
        

    def forward(self, input_ids, attention_mask):
        output = self.BERT(input_ids= input_ids, attention_mask= attention_mask)
        CLS_tokens = output.last_hidden_state[:, 0, :]
        z = self.linear_layer(CLS_tokens)
        return F.sigmoid(z).squeeze(-1)