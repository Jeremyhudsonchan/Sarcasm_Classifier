import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from nltk.stem.wordnet import WordNetLemmatizer

df = pd.read_csv('Data/train-balanced-sarcasm.csv')
df = df.dropna()
df = df.sample(frac=0.01)
df['comment'] = df['comment'].apply(lambda s: re.sub('[^a-zA-Z]', ' ', s))

lemmatizer = WordNetLemmatizer()
df['comment'] = df['comment'].apply(lambda s: ' '.join(
    [lemmatizer.lemmatize(word) for word in s.split()]))

tv = TfidfVectorizer(max_features=5000)
features = list(df['comment'])
features = tv.fit_transform(features).toarray()

X_train, X_test, y_train, y_test = train_test_split(
    features, df['label'], test_size=0.2, random_state=42)

# model 1:-
# Linear support vector classifier
lsvc = LinearSVC()
lsvc.fit(X_train, y_train)
print(lsvc.score(X_train, y_train))
print(lsvc.score(X_test, y_test))

lsvc_scores = cross_val_score(lsvc, X_train, y_train, cv=10)
print("Linear Support Vector Classifier Cross Validation Score:", lsvc_scores.mean())

lsvc_y_pred = lsvc.predict(X_test)
lsvc_cm = confusion_matrix(y_test, lsvc_y_pred)
print(lsvc_cm)

# model 2:-
# Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
print(mnb.score(X_train, y_train))
print(mnb.score(X_test, y_test))

mnb_scores = cross_val_score(mnb, X_train, y_train, cv=10)
print("Multinomial Naive Bayes Cross Validation Score:", mnb_scores.mean())

mnb_y_pred = mnb.predict(X_test)
mnb_cm = confusion_matrix(y_test, mnb_y_pred)
print(mnb_cm)

# model 3:-
# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))

lr_scores = cross_val_score(lr, X_train, y_train, cv=10)
print("Logistic Regression Cross Validation Score:", lr_scores.mean())

lr_y_pred = lr.predict(X_test)
lr_cm = confusion_matrix(y_test, lr_y_pred)
print(lr_cm)

# model 4:-
# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=10, random_state=0)
rfc.fit(X_train, y_train)
print(rfc.score(X_train, y_train))
print(rfc.score(X_test, y_test))

rfc_scores = cross_val_score(rfc, X_train, y_train, cv=10)
print("Random Forest Classifier Cross Validation Score:", rfc_scores.mean())

rfc_y_pred = rfc.predict(X_test)
rfc_cm = confusion_matrix(y_test, rfc_y_pred)
print(rfc_cm)
