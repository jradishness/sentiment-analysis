import time
t0 = time.time()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
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
vect = CountVectorizer(min_df=4,          # Minimum Document Frequency
                       ngram_range=(1,4),   # unigrams to trigrams
                       # stop_words='english'   # stopwords
                       ).fit(X_train)
X_train_vectorized = vect.transform(X_train)
X_test_vectorized = vect.transform(X_test)
print(len(vect.get_feature_names()), "features found.")     # Feature check

# Numpy Dense Array Transformation
print("Converting to Numpy Dense Array...")
X_train_dense = X_train_vectorized.toarray()   # Convert to Dense Numpy Array
X_test_dense = X_test_vectorized.toarray()

# # Classifier Comparison

# Gaussian Naive Bayes model
print("Training/Evaluating Gaussian Naive Bayes model...")
gnb_model = GaussianNB().fit((X_train_dense), y_train)
print("GaussianNB accuracy, on validation data: ", gnb_model.score((X_test_dense), y_test))



# Final statistics

          # This is the actual testing data being read into a new variable
test_doc = pd.read_csv('test_data.csv')     # reading in the testing data file for final testing

X_train2, X_test2, y_train2, y_test2 = train_test_split(test_doc['review'], # creating a similar variable with test data
                                                        test_doc['target'],
                                                        test_size=.999999,  # to set every document to the test variable
                                                        random_state=0
                                                        )

X_test2_vect = vect.transform(X_test2)          # Test Data feature extraction
# print('\n\nX_train shape: ', X_test2_vect.shape)      # Are we on the right track?
X_testFinalDense = X_test2_vect.toarray()

weighted_prediction = gnb_model.predict(X_testFinalDense)

print("\nGaussian Naive Bayes Model statistics, on Test data")    # statistics for the Gaussian model
print('F1 score:', f1_score(y_test2, weighted_prediction,average='weighted'))
print('Recall:', recall_score(y_test2, weighted_prediction,
                              average='weighted'))
print('Precision:', precision_score(y_test2, weighted_prediction,
                                    average='weighted'))
print('\n clasification report:\n', classification_report(y_test2, weighted_prediction))
print('\n confussion matrix:\n',confusion_matrix(y_test2, weighted_prediction))


t1 = time.time()

print("time to run: ", t1-t0)