import csv
import itertools
import math
import random
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
#from transformers import AutoTokenizer

#from BERTModel import BERTGenreClassification

LANGUAGES = ["eng", "rus", "ita", "spa", "fra", "tur", "deu", "cmn"]


def load_data(filename: str, avg_samples_per_language: Optional[int] = None) \
    -> Tuple[List[str], List[str]]:
    """
    Load sentence-language pairs from a .tsv file and optionally randomly sample
    a smaller number of them

    Args:
        filename (str): the data file path
        avg_samples_per_language (Optional[int]): number of samples to return (if None, return all).
            Defaults to None.

    Returns:
        Tuple[List[str], List[str]]: first list is sentences, second list is languages
    """
    # data will be initially stored as (sentence, language) tuples
    data = []
    with open(filename, "r") as f:
        datareader = csv.reader(f, delimiter="\t")
        # skip header row
        next(datareader)
        for row in datareader:
            data.append(tuple(row))

    # choose len(LANGUAGES) * avg_samples_per_language samples
    if avg_samples_per_language is not None:
        random.seed(457)
        samples = random.sample(data, len(LANGUAGES) * avg_samples_per_language)
    else:
        samples = data

    # split to two lists
    sentences = [sent for sent, _ in samples]
    langs = [lang for _, lang in samples]

    return sentences, langs


def get_char_ngrams(string: str, n: int) -> List[str]:
    """
    Gets a list of character n-grams from a string

    Args:
        string (str): the string
        n (int): the n-gram size

    Returns:
        List[str]: the character n-grams
    """
    char_ngrams = []
    for i in range(len(string) - n + 1):
        char_ngrams.append(string[i:i+n])
    # special case: there are no n-grams because n < len(string)
    # in that case, return a list with just the string in it
    if len(char_ngrams) == 0:
        return [string]
    return char_ngrams


def normalize(count_dict: Dict[Any, int], log_prob: bool = True) -> Dict[Any, float]:
    """
    Normalize counts in a dictionary to probabilities. Optionally, convert to log probabilities

    Args:
        count_dict (Dict[Any, int]): dictionary of counts
        log_prob (bool, optional): use log probabilities. Defaults to True.

    Returns:
        Dict[Any, float]: dictionary of probabilities
    """
    prob_dict = {}
    total = sum(count_dict.values())
    for key, value in count_dict.items():
        if log_prob:
            prob_dict[key] = math.log(value) - math.log(total)
        else:
            prob_dict[key] = value / total
    return prob_dict

def argmax(score_dict: Dict[Any, float]) -> Any:
    """
    Returns the key with the highest value in the dictionary

    Args:
        score_dict (Dict[Any, float]): the dictionary

    Returns:
        Any: the key with the highest value
    """
    max_lang = None
    max_score = float("-inf")
    for lang, score in score_dict.items():
        if score > max_score:
            max_lang = lang
            max_score = score
    return max_lang


def _create_row(data: List[Any], cell_width: int, row_label: Optional[str] = None) -> str:
    strings = [str(item) for item in data]
    end = "|".join(string.center(cell_width) for string in strings) + "|"
    if row_label:
        return "|" + row_label.center(cell_width) + "|" + end
    else:
        return "|" + " " * cell_width + "|" + end


def _create_line(cell_width: int, num_cols: int, vbar_edge: bool = True, vbar_sep: bool = True):
    edge = "|" if vbar_edge else "-"
    sep = "|" if vbar_sep else "-"
    return edge + sep.join("-" * cell_width for _ in range(num_cols)) + edge


def print_confusion_matrix(matrix: List[List[int]], labels: List[str],
                           cell_width: int = 5):
    """
    Prints a given confusion matrix

    Args:
        matrix (List[List[int]]): the confusion matrix
        labels (List[Any]): the column/rows labels for the matrix
    """
    # choose a longer cell width if the int values are too large
    longest_num_length = max(len(str(num)) for num
                             in itertools.chain.from_iterable(matrix))
    cell_width = max(longest_num_length + 2, cell_width)
    num_cols = len(labels) + 1

    matrix_strings = []

    # build header
    matrix_strings.append(_create_line(cell_width, num_cols, vbar_edge=False, vbar_sep=False))
    matrix_strings.append("|" + "Actual Label".center((cell_width + 1) * num_cols - 1) + "|")
    matrix_strings.append(_create_line(cell_width, num_cols, vbar_sep=False))
    matrix_strings.append(_create_row(labels, cell_width))

    # build table data
    for i, row in enumerate(matrix):
        matrix_strings.append(_create_line(cell_width, num_cols))
        matrix_strings.append(_create_row(row, cell_width, row_label=labels[i]))
    matrix_strings.append(_create_line(cell_width, num_cols, vbar_edge=False, vbar_sep=False))

    # add in header for the row labels by shifting everything over
    predicted_label = ["Predicted", "Label"]
    labeled_matrix_strings = []
    label_position = len(matrix_strings) // 2
    label_width = max(len(x) for x in predicted_label) + 2
    j = 0
    for i, row in enumerate(matrix_strings):
        if i >= label_position and j < len(predicted_label) :
            # adding in the label
            label = predicted_label[j]
            j += 1
            labeled_matrix_strings.append(f"|{label.center(label_width)}{row}")
        elif i in (0, 4, len(matrix_strings) - 1):
            # special rows
            labeled_matrix_strings.append(f"{'-' * (label_width + 1)}{row}")
        else:
            # most rows
            labeled_matrix_strings.append(f"|{' ' * label_width}{row}")

    # print the final table
    print("\n".join(labeled_matrix_strings))

def model_accuracy(model: BERTGenreClassification, dataloader: DataLoader, device):
    """Compute the accuracy of a binary classification model

    Args:
        model (HateSpeechClassificationModel): a hate speech classification model
        dataloader (DataLoader): a pytorch data loader to test the model with
        device (string): cpu or cuda, the device that the model is on

    Returns:
        float: the accuracy of the model
    """
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in dataloader:
            pred = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            correct += (batch["label_int"] == (pred.to("cpu").squeeze() > 0.5).to(int)).sum().item()
            total += batch["label_int"].shape[0]
        acc = correct / total
        return acc


def get_dataloader(data, batch_size: int = 4):
    """
    Get a pytorch dataloader for a specific data split

    Args:
        data_split (str): the data split
        data_path (str, optional): a data path if the data is not stored at the default path.
            For students using ada, this should always be None. Defaults to None.
        batch_size (int, optional): the desired batch size. Defaults to 4.

    Returns:
        DataLoader: the pytorch dataloader object
    """
    #dataset = Dataset.from_pandas(data)
    dataset = data
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = dataset.map(lambda ex: tokenizer(ex["description"], truncation=True, padding="max_length"), batched=True)
    

    #dataset = data.map(lambda x: toke, batched = True)
    #dataset = dataset.with_format("torch")
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

'''
def BERT_preprocess(data, labels_as_id):
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    for entry in range(len(data)):
        tokens = tokenizer(data.iloc[entry, 1], truncation = True, padding = 'max_length', max_length=256)
        data.iat[entry, 5] = tokens

        labels = [0] * len(labels_as_id)

        for l, i in labels_as_id.items():
            
            if l == data.iloc[entry, 2] or l == data.iloc[entry, 3] or l == data.iloc[entry, 4]: 
                labels[i] = 1

        data.iat[entry, 6] = labels
    return data
'''

def BERT_preprocess(dataset: dict, id2label, tokenizer):
    labels = [0] * len(id2label)
    for k, l in id2label.items():
        if l in [dataset['genre1'], dataset['genre2'], dataset['genre3']]:
            labels[k] = 1
        else:
            labels[k] = 0
    dataset = tokenizer(dataset['description'], truncation= True, padding= 'max_length', max_length = 256)
    dataset['label'] = labels
    return dataset
    