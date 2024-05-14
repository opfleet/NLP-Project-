README
Book Genre Classification with Naive Bayes and BERT 


This project explores large-scale book-genre classification, through the implementation of two Natural Language Processing Models: Naive Bayes and BERT. Driven by a keen interest in exploring different methodologies for tackling this classification task, our focus was on examining the contrast in efficiency and performance between complex, large-scale and simpler models while embracing the challenge of multi-genre classification. Working with a diverse dataset comprising book-specific attributes such as title, author, language, rating, ISBN, and multiple genre tags, our exploration began with thorough data preprocessing. Our two models were trained on this cleaned dataset, with comprehensive analysis of our chosen models' efficacy and performance.

This README details how to run our code to reproduce the main results in our associated project paper. 

DATA CLEANING AND PREPROCESSING: 
Initially, our dataset was structured with individual rows corresponding to specific books, encompassing various attributes including title, author, language, rating, isbn, summary, multiple genre tags associated with the book, a list of characters, and more. To refine our dataset, we conducted several stages of data preprocessing to be left with our final data for model training.
We first filtered out all extraneous columns except "description", "genres", "language". Books not in the English language were filtered out, and the language column was subsequently removed. Then we employed multiple regular expressions (regex) to eliminate unnecessary phrases that could potentially hinder the accuracy of our model training. Any missing values or NaNs were systematically excluded from the dataset. We elected to retain only the top three genres for each book for our model prediction task. As the dataset originally contained 311 different potential categories, we identified the top 10 most prevalent genres (Fiction, Romance, Fantasy, Young Adult, COntemporary, Adult, Nonfiction, Historical Fiction, and Audiobook), consolidating all other genres into the category of ‘other’. Left with a CSV dataset comprising of cleaned book description summaries, as well as the top 3 genres associated with each book entry. 

NAIVE BAYES MODEL: 
The NBLangIDModel class implements a Naive Bayes language identification model for classifying sentences into different languages. The core functionality of the NBLangIDModel class revolves around two key methods: fit and predict. All calculations were performed in log space to avoid underflow issues.


BERT MODEL: 






GRADESCOPE COMMENT... DELETE IN FUTURE: 
In your code, you should include thorough documentation. This includes at a bare minimum a README.md file that describes how to run your code to reproduce the main results in your paper. For instance, if you report 56% accuracy from your baseline model and 80% accuracy from your new and improved model, you should describe the series of scripts or commands that you would run to land at both of those results.