from typing import Dict, List
from collections import defaultdict
import util
import math 

class NBLangIDModel:
    def __init__(self, ngram_size: int = 2, extension: bool = False):
        """
        NBLangIDModel constructor

        Args:
            ngram_size (int, optional): size of char n-grams. Defaults to 2.
            extension (bool, optional): set to True to use extension code. Defaults to False.
        """
        self._priors = {}
        self._likelihoods = {}
        self.ngram_size = ngram_size
        self.extension = extension
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
        bigramCounts = defaultdict(lambda : defaultdict(int))
        self.labels = train_labels

        for sentence, label in zip(train_sentences, train_labels):
            bigrams = util.get_char_ngrams(sentence, self.ngram_size)
            labelCounts[label] += 1
            for bigram in bigrams:
                bigramCounts[label][bigram] += 1
                self.vocab.add(bigram)

        #prior probs
        self._priors = util.normalize(labelCounts, log_prob = True)

        #uniform prior
        #for label in train_labels:
            #self._priors[label] = math.log(1/8)

        """
        #prior likelihoods 
        for label, bigrams in bigramCounts.items(): 
            totalBigramNum = sum(bigrams.values())
            likelihoods = {}
            for bigram, count in bigrams.items():
                likelihoods[bigram] = math.log((count + 1) / (totalBigramNum + len(bigrams)))
                self._likelihoods[label] = likelihoods
        """
        #add-one smoothing 
        for label, bigrams in bigramCounts.items():
            for bigram in self.vocab:
                if bigram not in bigrams:
                    bigramCounts[label][bigram] = 1
                else: 
                    bigramCounts[label][bigram] += 1
        
        #normalize denominator and reassign likelihoods
        for label, bigrams in bigramCounts.items():
            totalBigramNum = sum(bigrams.values()) + len(self.vocab)
            self._likelihoods[label] = util.normalize(bigrams, log_prob = True)

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

        sentenceBigrams = util.get_char_ngrams(test_sentence, self.ngram_size)
        for bigram in sentenceBigrams:
            #ignores unseen words
            if bigram in self.vocab:
                for label in set(self.labels):
                #access prior and likelihood, then use to calculate probability of each bigram in that specific label 
                    returnDict[label] += self._likelihoods[label][bigram]
                
        return returnDict
