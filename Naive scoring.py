from typing import Any, List

from util import LANGUAGES


def accuracy_score(y_true: List[Any], y_pred: List[Any]) -> float:
    """
    Compute the accuracy given true and predicted labels

    Args:
        y_true (List[Any]): true labels
        y_pred (List[Any]): predicted labels

    Returns:
        float: accuracy score
    """
    truePos = 0

    for trueLabel, predLabel in zip(y_true, y_pred):
        if (trueLabel == predLabel):
            truePos +=1
    accuracy = truePos / len(y_pred)

    return accuracy
            


def confusion_matrix(y_true: List[Any], y_pred: List[Any], labels: List[Any]) \
    -> List[List[int]]:
    """
    Builds a confusion matrix given predictions
    Uses the labels variable for the row/column order

    Args:
        y_true (List[Any]): true labels
        y_pred (List[Any]): predicted labels
        labels (List[Any]): the column/rows labels for the matrix

    Returns:
        List[List[int]]: the confusion matrix
    """
    # check that all of the labels in y_true and y_pred are in the header list
    for label in y_true + y_pred:
        assert label in labels, \
            f"All labels from y_true and y_pred should be in labels, missing {label}"
   
   #init matrix 
    matrix = [[0 for x in range(len(labels))] for x in range(len(labels))]
    
    for trueLabel, predLabel in zip(y_true, y_pred):
        trueIndex = labels.index(trueLabel)
        predIndex = labels.index(predLabel)
        matrix[predIndex][trueIndex] += 1
    
    return matrix    
