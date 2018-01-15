#########################################################################################################

"""
                                                NEWS AGGREGATOR
"""

#########################################################################################################


####Load libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from normalize_text import norm_text

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB

####Load dataset

filename = "uci-news-aggregator.csv"
df = pd.read_csv(filename)

df = df[['TITLE', 'CATEGORY']]

df['TEXT'] = [norm_text(s) for s in df["TITLE"]]

X = df['TEXT']

####Encoding the labels
encoder = LabelEncoder()
y = encoder.fit_transform(df['CATEGORY'])

####Preprocessing the dataset
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33, random_state=30,shuffle=True)

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

####Spot-checking classification algorithms to find best performer
models = []
models.append(('LR',LogisticRegression(class_weight='balanced', multi_class='multinomial', solver='newton-cg')))
models.append(('DTC',DecisionTreeClassifier()))
models.append(('NBC',MultinomialNB()))

for name, clf in models:
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    cv_results = model_selection.cross_val_score(clf, X_train, y_train, cv=kfold, scoring='accuracy')
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


###OUTPUT -: LR: 0.939467 (0.01644)
###          DTC: 0.906540 (0.002461)
###          NBC: 0.921235 (0.002376)

### Best performer is Logistic Regression. Parameter tuning is necessary for better performance.
