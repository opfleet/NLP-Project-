README
Book Genre Classification with Naive Bayes and BERT 


This project explores large-scale book-genre classification, through the implementation of two Natural Language Processing Models: Naive Bayes and BERT. Driven by a keen interest in exploring different methodologies for tackling this classification task, our focus was on examining the contrast in efficiency and performance between complex, large-scale and simpler models while embracing the challenge of multi-genre classification. Working with a diverse dataset comprising book-specific attributes such as title, author, language, rating, ISBN, and multiple genre tags, our exploration began with thorough data preprocessing. Our two models were trained on this cleaned dataset, with comprehensive analysis of our chosen models' efficacy and performance. 

This README details how to run our code to reproduce the main results in our associated project paper. 

DATA CLEANING AND PREPROCESSING: 
Initially, our dataset was structured with individual rows corresponding to specific books, encompassing various attributes including title, author, language, rating, isbn, summary, multiple genre tags associated with the book, a list of characters, and more. To refine our dataset, we conducted several stages of data preprocessing to be left with our final data for model training.
We first filtered out all extraneous columns except "description", "genres", "language". Books not in the English language were filtered out, and the language column was subsequently removed. Then we employed multiple regular expressions (regex) to eliminate unnecessary phrases that could potentially hinder the accuracy of our model training. Any missing values or NaNs were systematically excluded from the dataset. We elected to retain only the top three genres for each book for our model prediction task. As the dataset originally contained 311 different potential categories, we identified the top 10 most prevalent genres (Fiction, Romance, Fantasy, Young Adult, COntemporary, Adult, Nonfiction, Historical Fiction, and Audiobook), consolidating all other genres into the category of ‘other’. Left with a CSV dataset comprising of cleaned book description summaries, as well as the top 3 genres associated with each book entry. The data was loaded in, ///descriptions = pd.read_csv("cleanedData.csv”)///  Train test splits were created utilizing library sklearn’s function ///train, test = train_test_split(descriptions, test_size=0.2)/// 
Both training and testing data were then split into X and Y columns for model usage: 
///
train_X = train['description']
train_y1 = train['genre1']
train_y2 = train['genre2']
train_y3 = train['genre3’]test_X = test['description']
test_y1 = test['genre1']
test_y2 = test['genre2']
test_y3 = test['genre3']
///

NAIVE BAYES MODEL: 
The NBLangIDModel class implements a Naive Bayes language identification model for classifying sentences into different languages. The core functionality of the NBLangIDModel class revolves around two key methods: fit and predict. All calculations were performed in log space to avoid underflow issues. After implementation, the model was trained via commands 
///
naiveBayes = NBLangIDModel() 
naiveBayes.fit(train_X.tolist(), train_y1.tolist())
///

Predictions on the test set made by the trained Naive Bayes model, were then obtained by commands: 
///
predictions = test_X.apply(lambda sentence: naiveBayes.predict_one_log_proba(sentence))
argmaxPreds = [argmaxThree(dictionary) for dictionary in predictions]
/// 

The predictions made my the Naive Bayes Model were then analyzed:
///
accuracyPos, genre1Acc, genre2Acc, genre3Acc = AccPosDepend(argmaxPreds, test_y1, test_y2, test_y3)
accuracyNonPos, pred1Acc, pred2Acc, pred3Acc = AccPosNonDepend(argmaxPreds, test_y1, test_y2, test_y3)
accuracyLevels = AccLevels(argmaxPreds, test_y1, test_y2, test_y3)
///
Exploring different measures of accuracy, including genre position-dependent and genre position-independent accuracy. Genre position-dependent accuracy indicates the model's ability to accurately assign genres based on their position within the prediction.



BERT MODEL: 
To reproduce our BERT model's results, the user will have to use a HPC cluster, such as the Middlebury Ada cluster. Since the user will have to run a Python file using an sbatch command, they will have to utilize a remote connection in VSCode, and utilize the `nlp-project.py`, which is the Python file equivalent to our Jupyter Notebook. Within this file, there are three areas of our code that have the labels **for top 3** and **for top 1**. For our 3-label classification that returned a 34% accuracy rate, make sure that the **for top 3** sections are uncommented, and that the **for top 1** sections are commented out. For our 1-label classification that returned a 15% accuracy rate, make sure that the opposite is true: that the **for top 1** sections are uncommented, and that the **for top 3** sections are commented out. After ensuring these conditions dependent on which performance the user is trying to emulate, open the remote connection terminal and run `sbatch nlp-project.sbatch` (all of the BERT training parameters are consistent between the two run conditions). Included in the `nlp-project.py` file are classification reports that will allow the user to see the accuracy rates of the BERT classification model.