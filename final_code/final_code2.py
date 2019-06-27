import time
t0 = time.time()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

df = pd.read_csv('train_data.csv')      # to read the training data into working memory

X_train, X_test, y_train, y_test = train_test_split(df['review'],   # Test-Train Split function
                                                    df['target'],
                                                    train_size = 0.8,
                                                    random_state=14
                                                    )
vect = CountVectorizer(min_df=2,          # Minimum Document Frequency
                       ngram_range=(1,4),   # unigrams to trigrams
                       # stop_words='english'   # stopwords
                       ).fit(X_train)
X_train_vectorized = vect.transform(X_train)
X_test_vectorized = vect.transform(X_test)

lr_model = LogisticRegression().fit(X_train_vectorized, y_train)
print("Logistic Regression accuracy, on validation data: ", lr_model.score(X_test_vectorized, y_test))
