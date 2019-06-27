import time
t0 = time.time()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer   # “Term Frequency times Inverse Document Frequency”
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB       # Multinomial Naive Bayes, supposedly goes well with the data from the transformer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, confusion_matrix

print("Reading the data into a working variable...")        # READ DATA IN
df = pd.read_csv('train_data.csv')      # to read the training data into working memory

# DATA SPLITING
print("Splitting Data into subsets...")
X_train, X_test, y_train, y_test = train_test_split(df['review'],   # Test-Train Split function
                                                    df['target'],
                                                    train_size = 0.8,
                                                    test_size = 0.2,
                                                    random_state=14
                                                    )
print("Training shape is: ", X_train.shape)

# FEATURE EXTRACTION
print("Extracting features...")
vect = CountVectorizer(min_df=2,          # Minimum Document Frequency
                       ngram_range=(1,4),   # unigrams to trigrams
                       # stop_words='english'   # stopwords
                       ).fit(X_train)
X_train_vectorized = vect.transform(X_train)
X_test_vectorized = vect.transform(X_test)
print(len(vect.get_feature_names()), "features found.")     # Feature check

# TFIDF Transformer
print("Transforming for tfidf...")
X_train_vectorized_tfidf = TfidfTransformer().fit_transform(X_train_vectorized)     # or X_train_dense
X_test_vectorized_tfidf = TfidfTransformer().fit_transform(X_test_vectorized)       # or X_test_dense

# # Numpy Dense Array Transformation
# print("Converting to Numpy Dense Array...")
# X_train_dense = X_train_vectorized.toarray()   # Convert to Dense Numpy Array
# X_test_dense = X_test_vectorized.toarray()

# # Classifier Comparison

# # Gaussian Naive Bayes model
# print("Training/Evaluating Gaussian Naive Bayes model...")
# gnb_model = GaussianNB().fit((X_train_dense), y_train)
# print("GaussianNB accuracy, on validation data: ", gnb_model.score((X_test_dense), y_test))


# Bernoulli Naive Bayes model
print("Training/Evaluating Bernoulli Naive Bayes model...")
bnb_model = BernoulliNB().fit(X_train_vectorized, y_train)
print("BernoulliNB accuracy, on validation data: ", bnb_model.score(X_test_vectorized, y_test))


# Mulitnomial Naive Bayes model
print("Training/Evaluating Multinomial Naive Bayes model...")
mnb_model = MultinomialNB().fit(X_train_vectorized, y_train)
print("MultinomialNB accuracy, on validation data: ", mnb_model.score(X_test_vectorized, y_test))


# Linear SVC Model
print("Training/Evaluating Linear SVC model...")
lsvc_model = LinearSVC().fit(X_train_vectorized_tfidf, y_train)
print("Linear SVC accuracy, on validation data: ", lsvc_model.score(X_test_vectorized_tfidf, y_test))


# Logistic Regression Model
print("Training/Evaluating Logistic Regression model...")
lr_model = LogisticRegression().fit(X_train_vectorized, y_train)
print("Logistic Regression accuracy, on validation data: ", lr_model.score(X_test_vectorized, y_test))


# # svm.SVC model
# clfrSVM = svm.SVC(kernel='linear', C=0.1)             # SVM Classifier training without tfidf
# clfrSVM.fit(X_train_vectorized, y_train)
# predicted_labels = clfrSVM.predict(X_test_vectorized)
# print("SVM, Accuracy on validation data:", accuracy_score(y_test, predicted_labels))
#
# clfrSVMtfidf = svm.SVC(kernel='linear', C=0.1)          # SVM Classifier training with tfidf
# clfrSVMtfidf.fit(X_train_vectorized_tfidf, y_train)
# predicted_labels_tfidf = clfrSVMtfidf.predict(X_test_vectorized_tfidf)
# print("SVM, Accuracy on validation data with tfidf:", accuracy_score(y_test, predicted_labels_tfidf))


# Final statistics

          # This is the actual testing data being read into a new variable
test_doc = pd.read_csv('test_data.csv')     # reading in the testing data file for final testing

X_train2, X_test2, y_train2, y_test2 = train_test_split(test_doc['review'], # creating a similar variable with test data
                                                        test_doc['target'],
                                                        test_size=.999999,
                                                        random_state=0
                                                        )

X_test2_vect = vect.transform(X_test2)          # Test Data feature extraction
# print('\n\nX_train shape: ', X_test2_vect.shape)      # Are we on the right track?
X_test2_final = TfidfTransformer().fit_transform(X_test2_vect)      # Tfidf transformation of Test Data
# X_testFinalDense = X_test2_final.toarray()        # Numpy vector smoothing

# Elicitation of LinearSVC Stats
weighted_prediction = lsvc_model.predict(X_test2_final)
print("\nLinearSVC Model statistics, on Test data (with tfidf)")    # statistics for the LinearSVC model with tfidf
print('Accuracy:', accuracy_score(y_test2, weighted_prediction))
print('F1 score:', f1_score(y_test2, weighted_prediction,average='weighted'))
print('Recall:', recall_score(y_test2, weighted_prediction,
                              average='weighted'))
print('Precision:', precision_score(y_test2, weighted_prediction,
                                    average='weighted'))
print('\n clasification report:\n', classification_report(y_test2, weighted_prediction))
print('\n confussion matrix:\n',confusion_matrix(y_test2, weighted_prediction))

# Elicitation of Logistic Regression Stats
lrweighted_prediction = lr_model.predict(X_test2_final)
print("\nLogistic Regression Model statistics, on Test data (with tfidf)")    # statistics for the LogisticRegression model with tfidf
print('Accuracy:', accuracy_score(y_test2, lrweighted_prediction))
print('F1 score:', f1_score(y_test2, lrweighted_prediction,average='weighted'))
print('Recall:', recall_score(y_test2, lrweighted_prediction,
                              average='weighted'))
print('Precision:', precision_score(y_test2, lrweighted_prediction,
                                    average='weighted'))
print('\n clasification report:\n', classification_report(y_test2, lrweighted_prediction))
print('\n confussion matrix:\n',confusion_matrix(y_test2, lrweighted_prediction))

# Elicitation of Multinomial Naive Bayes Stats
mnbweighted_prediction = mnb_model.predict(X_test2_final)
print("\nMultinomial Naive Bayes Model statistics, on Test data (with tfidf)")    # statistics for the MultinomialNB model with tfidf
print('Accuracy:', accuracy_score(y_test2, mnbweighted_prediction))
print('F1 score:', f1_score(y_test2, mnbweighted_prediction,average='weighted'))
print('Recall:', recall_score(y_test2, mnbweighted_prediction,
                              average='weighted'))
print('Precision:', precision_score(y_test2, mnbweighted_prediction,
                                    average='weighted'))
print('\n clasification report:\n', classification_report(y_test2, mnbweighted_prediction))
print('\n confussion matrix:\n',confusion_matrix(y_test2, mnbweighted_prediction))

# Elicitation of Bernoulli Naive Bayes Stats
bnbweighted_prediction = bnb_model.predict(X_test2_final)
print("\nBernoulli Naive Bayes Model statistics, on Test data (with tfidf)")    # statistics for the BernoulliNB model with tfidf
print('Accuracy:', accuracy_score(y_test2, bnbweighted_prediction))
print('F1 score:', f1_score(y_test2, bnbweighted_prediction,average='weighted'))
print('Recall:', recall_score(y_test2, bnbweighted_prediction,
                              average='weighted'))
print('Precision:', precision_score(y_test2, bnbweighted_prediction,
                                    average='weighted'))
print('\n clasification report:\n', classification_report(y_test2, bnbweighted_prediction))
print('\n confussion matrix:\n',confusion_matrix(y_test2, bnbweighted_prediction))

t1 = time.time()

print("time to run: ", t1-t0)