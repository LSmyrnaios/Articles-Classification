# from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn import preprocessing
import numpy as np
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.preprocessing import Normalizer

if __name__ == '__main__':
    train_data = pd.read_csv('train_set.csv', sep="\t")
    testdata = pd.read_csv('test_set.csv', sep="\t")

    train_data = train_data[0:25]
    testdata = testdata[0:25]

    le = preprocessing.LabelEncoder()
    le.fit(train_data["Category"])
    y = le.transform(train_data["Category"])

    # print y
    count_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
    tfidf = TfidfTransformer()
    clf = RandomForestClassifier(n_estimators=50)

    # vX = count_vectorizer.fit_transform(train_data)
    # tfidfX = tfidf.fit_transform(vX)
    # predicted = clf.predict(tfidfX)

    # vX = count_vectorizer.fit_transform(testdata)
    # tfidfX = tfidf.fit_transform(vX)
    # predicted = clf.predict(tfidfX)

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier()),
    ])

    vectorTrain = count_vectorizer.fit_transform(train_data['Content'], train_data['Title'])
    vectorTest = count_vectorizer.fit_transform(testdata['Content'], testdata['Title'])
    transformer = TfidfTransformer(smooth_idf=False)

    print vectorTrain.shape
    print vectorTest.shape
    # print vectorTest.toarray()

    lsa = TruncatedSVD(n_components=100)

    vectorTrain = lsa.fit_transform(vectorTrain)
    vectorTrain = Normalizer(copy=False).fit_transform(vectorTrain)

    vectorTest = lsa.fit_transform(vectorTest)
    vectorTest = Normalizer(copy=False).fit_transform(vectorTest)

    print vectorTrain.shape
    print vectorTest.shape

    clf.fit(vectorTrain, y)

    y_pred = clf.predict(vectorTest)

    # print y_pred
    # print(clf.feature_importances_)

    # y_pred = pipeline.fit(vectorTrain, y).predict(vectorTrain)
    # Now evaluate all steps on test set
    # y_pred = pipeline.predict(vectorTest)

    predicted_categories = le.inverse_transform(y_pred)
    print predicted_categories
    print classification_report(y, y_pred, target_names=list(le.classes_))

    print "Train Accuracy :: ", accuracy_score(y, clf.predict(vectorTrain))
    print "Test Accuracy  :: ", accuracy_score(y, y_pred)

    # train_test_split()
