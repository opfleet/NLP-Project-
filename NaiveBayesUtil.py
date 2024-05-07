import csv
import itertools
import math
import random
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import torch
from datasets import Dataset

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
