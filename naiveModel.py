from typing import Dict, List
from collections import defaultdict
import util
import math 

class NBLangIDModel:
    def __init__(self):
        """
        NBLangIDModel constructor

        Args:
            ngram_size (int, optional): size of char n-grams. Defaults to 2.
            extension (bool, optional): set to True to use extension code. Defaults to False.
        """
        self._priors = {}
        self._likelihoods = {}
        #added in vocab for all possible char bigrams, no duplicates
        self.vocab = set()
        #added in labels for all possible language labels, no duplicates 
        self.labels = set()

    def fit(self, train_sentences: List[str], train_labels: List[str]):
        """
        Train the Naive Bayes model (by setting self._priors and self._likelihoods)

        Args:
            train_sentences (List[str]): sentences from the training data
            train_labels (List[str]): labels from the training data
        """
        labelCounts = defaultdict(int)
        #lambda to ensure no typeError, no need to initialize new keys
        wordCounts = defaultdict(lambda : defaultdict(int))
        self.labels = train_labels

    

        for summary, label in zip(train_sentences, train_labels):
            print(type(summary))
            wordList = summary.split()
            print(type(wordList))
            labelCounts[label] += 1
            for word in wordList:
                wordCounts[label][word] += 1
                self.vocab.add(word)

        #prior probs
        self._priors = util.normalize(labelCounts, log_prob = True)

        #add-one smoothing 
        for label, wordList in wordCounts.items():
            for word in self.vocab:
                if word not in wordList:
                    wordCounts[label][word] = 1
                else: 
                    wordCounts[label][word] += 1
        
        #normalize denominator and reassign likelihoods
        for label, wordList in wordCounts.items():
            totalWordNum = sum(wordList.values()) + len(self.vocab)
            self._likelihoods[label] = util.normalize(wordList, log_prob = True)

    def predict(self, test_sentences: List[str]) -> List[str]:
        """
        Predict labels for a list of sentences

        Args:
            test_sentences (List[str]): the sentence to predict the language of

        Returns:
            List[str]: the predicted languages (in the same order)
        """
        #so for each sentence, we need to do the same thing, split into bigrams, get language of each bigram then of each sentence 
        #using argmax, then add all to a list, then return 


        #init list of all test_sentence label results 
        labelResults = []

        for sentence in test_sentences: 
            sentenceAllLabelProbs = self.predict_one_log_proba(sentence)
            currSentenceLabel = util.argmax(sentenceAllLabelProbs)
            labelResults.append(currSentenceLabel)

        return labelResults

            
    def predict_one_log_proba(self, test_sentence: str) -> Dict[str, float]:
        """
        Computes the log probability of a single sentence being associated with each language

        Args:
            test_sentence (str): the sentence to predict the language of

        Returns:
            Dict[str, float]: mapping of language --> probability
        """
        assert not (self._priors is None or self._likelihoods is None), \
            "Cannot predict without a model!"
        
        returnDict = defaultdict(float)
        for label in set(self.labels):
            returnDict[label] = self._priors[label]

        summaryWords = test_sentence.split()
        for word in summaryWords:
            #ignores unseen words
            if word in self.vocab:
                for label in set(self.labels):
                #access prior and likelihood, then use to calculate probability of each word in that specific label 
                    returnDict[label] += self._likelihoods[label][word]
                
        return returnDict
