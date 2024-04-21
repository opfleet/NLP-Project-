import argparse

import torch
import torch.nn as nn
from torch import cuda, manual_seed
from torch.utils.data import DataLoader

from model import HateSpeechClassificationModel
from util import get_dataloader, model_accuracy


def train_model(model: HateSpeechClassificationModel, train_dataloader: DataLoader, 
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
    device = "cuda" if cuda.is_available() else "cpu"
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
        print(f"For epoch {e}:")
        print(f"Average loss (over batches in epoch) = {e_loss/b_num}")
        print(f"Accuracy on training set = {train_accuracy}")
        print(f"Accuracy on development set = {test_accuracy}")
        print()
        
'''
def train_test( model: HateSpeechClassificationModel, train_dataloader: DataLoader, 
                dev_dataloader: DataLoader, epochs: int, learning_rate: float):
    device = "cuda" if cuda.is_available() else "cpu"

    loss_func = nn.BCELoss()
    preds = model.forward(input_ids= train_dataloader['input_ids'], attention_mask= train_dataloader['attention_mask'])
    print(torch.where(preds == train_dataloader['label_int'], 1.0, 0.0).mean())
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=5, type=int, 
                        help="The number of epochs to train for")
    parser.add_argument("--learning_rate", default=1e-2, type=float, 
                        help="The learning rate")
    parser.add_argument("--freeze_bert", action="store_true", 
                        help="True to freeze BERT parameters (no fine-tuning), False otherwise")
    parser.add_argument("--batch_size", default=16, type=int, 
                        help="The batch size")
    args = parser.parse_args()

    print(args.batch_size)

    # initialize model and dataloaders
    device = "cuda" if cuda.is_available() else "cpu"

    # seed the model before initializing weights so that your code is deterministic
    manual_seed(457)

    model = HateSpeechClassificationModel(freeze_bert=args.freeze_bert).to(device)
    train_dataloader = get_dataloader("train", batch_size=args.batch_size)
    dev_dataloader = get_dataloader("dev", batch_size=args.batch_size)

    train_model(model, train_dataloader, dev_dataloader,
                args.epochs, args.learning_rate)

if __name__ == "__main__":
    main()

