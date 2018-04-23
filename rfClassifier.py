# from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.preprocessing import Normalizer


stop_words = set(ENGLISH_STOP_WORDS)


def addPreDefinedStopWords():
    stop_words.add('said')
    stop_words.add('he')
    stop_words.add('He')
    stop_words.add('it')
    stop_words.add('It')
    stop_words.add('got')
    stop_words.add("don't")
    stop_words.add('like')
    stop_words.add("didn't")
    stop_words.add('ago')
    stop_words.add('went')
    stop_words.add('did')
    stop_words.add('day')
    stop_words.add('just')
    stop_words.add('thing')
    stop_words.add('think')
    stop_words.add('say')
    stop_words.add('says')
    stop_words.add('know')
    stop_words.add('clear')
    stop_words.add('despite')
    stop_words.add('going')
    stop_words.add('time')
    stop_words.add('people')
    stop_words.add('way')
    # TODO - Add more stopWords...


def split_dataset(dataset, train_percentage, feature_headers, target_header):
    """
    Split the dataset with train_percentage
    :param dataset:
    :param train_percentage:
    :param feature_headers:
    :param target_header:
    :return: train_x, test_x, train_y, test_y
    """

    # Split dataset into train and test dataset
    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header],
                                                        train_size=train_percentage)
    return train_x, test_x, train_y, test_y



if __name__ == '__main__':

    headers = ['RowNum', 'Id', 'Title', 'Content', 'Category']

    train_data = pd.read_csv('train_set.csv', sep="\t")
    test_data = pd.read_csv('test_set.csv', sep="\t")

    train_x, test_x, train_y, test_y = split_dataset(train_data, 0.7, headers[2:-1], headers[-1])


   # train_data = train_data[0:250]
   # test_data = test_data[0:250]

    le = preprocessing.LabelEncoder()
    le.fit(train_data["Category"])
    y = le.transform(train_data["Category"])

    # print y
    addPreDefinedStopWords()
    count_vectorizer = CountVectorizer(stop_words)
    #tfidf = TfidfTransformer()
    clf = RandomForestClassifier(n_estimators=50)

    # vX = count_vectorizer.fit_transform(train_data)
    # tfidfX = tfidf.fit_transform(vX)
    # predicted = clf.predict(tfidfX)

    # vX = count_vectorizer.transform(test_data)
    # tfidfX = tfidf.transform(vX)
    # predicted = clf.predict(tfidfX)

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier()),
    ])

    print train_data.shape

    for row in train_data:
        train_data['Content'] += 5 * train_data['Title']

    for row in test_data:
        test_data['Content'] += 5 * test_data['Title']

    vectorTrain = count_vectorizer.fit_transform(train_data['Content'])
    vectorTest = count_vectorizer.transform(test_data['Content'])
    # transformer = TfidfTransformer(smooth_idf=False)

    print vectorTrain.shape
    print vectorTest.shape
    # print vectorTest.toarray()

    lsa = TruncatedSVD(n_components=100)

    #vectorTrain = lsa.fit_transform(vectorTrain)
    # vectorTrain = Normalizer(copy=False).fit_transform(vectorTrain)

    #vectorTest = lsa.fit_transform(vectorTest)
    # vectorTest = Normalizer(copy=False).transform(vectorTest)

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

