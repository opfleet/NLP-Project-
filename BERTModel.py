import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cuda, manual_seed
from torch.utils.data import DataLoader
from transformers import AutoModel


class BERTGenreClassification(nn.Module):

    def __init__(self, freeze_bert: bool = False):
        super().__init__()

        self.BERT = AutoModel.from_pretrained("distilbert-base-uncased")
        self.linear_layer = nn.Linear(768, 1)

        if (freeze_bert):
            for parameter in self.BERT.parameters():
                parameter.requires_grad = False
        

    def forward(self, input_ids, attention_mask):
        output = self.BERT(input_ids= input_ids, attention_mask= attention_mask)
        CLS_tokens = output.last_hidden_state[:, 0, :]
        z = self.linear_layer(CLS_tokens)
        return F.sigmoid(z).squeeze(-1)
    

    
def train_model(model : BERTGenreClassification, train_dataloader: DataLoader, 
            dev_dataloader: DataLoader, epochs: int, learning_rate: float):
    """
    Trains model and prints accuracy on dev data after training

    Arguments:
        model (HateSpeechClassificationModel): the model to train
        train_dataloader (DataLoader): a pytorch dataloader containing the training data
        dev_dataloader (DataLoader): a pytorch dataloader containing the development data
        epochs (int): the number of epochs to train for (full iterations through the dataset)
        learning_rate (float): the learning rate

    Returns:
        float: the accuracy on the development set
    """
    # import within function to avoid circular import
    from util import model_accuracy

    # Edit device if need to use virtual machine/ada cluster
    device = 'cpu' #"cuda" if cuda.is_available() else "cpu"
    loss_func = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)

    for e in range(epochs):
        e_loss = 0.0
        b_num = 0

        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label_int'].to(device)

            model.zero_grad()
            preds = model(input_ids, attention_mask)
            b_loss = loss_func(preds.float(), labels.float())
            e_loss = e_loss + b_loss
            b_loss.backward()
            optimizer.step()
            b_num = b_num + 1

        train_accuracy = model_accuracy(model, train_dataloader, device)
        test_accuracy = model_accuracy(model, dev_dataloader, device)