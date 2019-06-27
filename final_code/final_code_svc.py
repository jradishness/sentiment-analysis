import time
t0 = time.time()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
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


# # Classifier Comparison

# svm.SVC model
clfrSVM = svm.SVC(kernel='linear', C=0.1)             # SVM Classifier training without tfidf
clfrSVM.fit(X_train_vectorized, y_train)
predicted_labels = clfrSVM.predict(X_test_vectorized)
print("SVM, Accuracy on validation data:", accuracy_score(y_test, predicted_labels))

# Begin commenting here to remove tfidf step from classifier training

# TFIDF Transformer
from sklearn.feature_extraction.text import TfidfTransformer
print("Transforming for tfidf...")
X_train_vectorized_tfidf = TfidfTransformer().fit_transform(X_train_vectorized)
X_test_vectorized_tfidf = TfidfTransformer().fit_transform(X_test_vectorized)
clfrSVMtfidf = svm.SVC(kernel='linear', C=0.1)          # SVM Classifier training with tfidf
clfrSVMtfidf.fit(X_train_vectorized_tfidf, y_train)
predicted_labels_tfidf = clfrSVMtfidf.predict(X_test_vectorized_tfidf)
print("SVM, Accuracy on validation data with tfidf:", accuracy_score(y_test, predicted_labels_tfidf))
# Stop commenting here for removing tfidf


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

weighted_prediction = clfrSVM.predict(X_test2)

print("\nSVM.SVC Model statistics, on Test data")             # statistics for the svm.svc model with no transformation
print('Accuracy:', accuracy_score(y_test2, weighted_prediction))
print('F1 score:', f1_score(y_test2, weighted_prediction,average='weighted'))
print('Recall:', recall_score(y_test2, weighted_prediction,
                              average='weighted'))
print('Precision:', precision_score(y_test2, weighted_prediction,
                                    average='weighted'))
print('\n clasification report:\n', classification_report(y_test2, weighted_prediction))
print('\n confussion matrix:\n',confusion_matrix(y_test2, weighted_prediction))


# Everything between here and line 98 can be commented out to skip tfidf

weighted_prediction_tfidf = clfrSVMtfidf.predict(X_test2_final)

print("\nSVM.SVC Model statistics, with Tfidf Transformation on Test data") # statistics for the svm.svc model with tfidf
print('Accuracy:', accuracy_score(y_test2, weighted_prediction_tfidf))
print('F1 score:', f1_score(y_test2, weighted_prediction_tfidf,average='weighted'))
print('Recall:', recall_score(y_test2, weighted_prediction_tfidf,
                              average='weighted'))
print('Precision:', precision_score(y_test2, weighted_prediction_tfidf,
                                    average='weighted'))
print('\n clasification report:\n', classification_report(y_test2, weighted_prediction_tfidf))
print('\n confussion matrix:\n',confusion_matrix(y_test2, weighted_prediction_tfidf))

# End of commenting to cancel tfidf transformation of statistics section


t1 = time.time()

print("time to run: ", t1-t0)