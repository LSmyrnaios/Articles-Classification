# from sklearn.cross_validation import train_test_split
import time

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
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
    # TODO - Add more stopWords..


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
                                                        train_size=train_percentage, test_size=0.3)
    return train_x, test_x, train_y, test_y


def svm_classifier():

    print 'Running svmClassifier...\n'

    headers = ['RowNum', 'Id', 'Title', 'Content', 'Category']

    train_data = pd.read_csv('Resources/train_set.csv', sep="\t")
    test_data = pd.read_csv('Resources/test_set.csv', sep="\t")

    # print(headers[2:4])

    # Split dataSet to 70-30.
    train_x, test_x, train_y, test_y = split_dataset(train_data, 0.7, headers[2:4], headers[-1])

    # LE
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(train_data["Category"])

    # print 'y : ', set(y)

    # Train and Test dataset size details
    print "Train_x Shape :: ", train_x.shape
    print "Train_y Shape :: ", train_y.shape
    print "Test_x Shape :: ", test_x.shape
    print "Test_y Shape :: ", test_y.shape
    print "Train_x colums ::", train_x.columns

    for row in train_data:
        train_x['Content'] += 5 * train_x['Title']

    for row in test_data:
        test_x['Content'] += 5 * test_x['Title']

    addPreDefinedStopWords()

    # print train_x['Content'][1]

    start_time_diadox = time.time()

    # Count Vectorizer
    count_vectorizer = CountVectorizer(stop_words)
    vectorTrain = count_vectorizer.fit_transform(train_x['Content'])
    vectorTest = count_vectorizer.transform(test_x['Content'])
    print "VectorTrain shape::", vectorTrain.shape
    print "VectorTest shape::", vectorTest.shape

    # TfidfTransformer
    tfidf = TfidfTransformer()
    vectorTrain = tfidf.fit_transform(vectorTrain)
    vectorTest = tfidf.transform(vectorTest)

    # TfidfVectorizer (it does the job of CountVectorizer & TfidfTransformer together)
    # tfidf_v = TfidfVectorizer(stop_words)
    # vectorTrain = tfidf_v.fit_transform(train_x['Content'])
    # vectorTest = tfidf_v.transform(test_x['Content'])

    # LSA
    lsa = TruncatedSVD(n_components=100)
    vectorTrain = lsa.fit_transform(vectorTrain)
    vectorTest = lsa.transform(vectorTest)

    print "VectorTrain shape after LSA::", vectorTrain.shape
    print "VectorTest shape after LSA::", vectorTest.shape

    # Normalizer
    norm = Normalizer(norm="l2", copy=True)
    vectorTrain = norm.fit_transform(vectorTrain)
    vectorTest = norm.transform(vectorTest)

    # CLF
    clf = svm.SVC(kernel='linear', C=1.0)
    # clf = svm.SVC(kernel='rbf', C=1.0, gamma='auto')

    # GridSearch (find the best parameters)
    # parameters = {'kernel': ('linear', 'rbf'), 'C': [1.5, 10], 'gamma': [0, 'auto']}
    # svr = svm.SVC()
    # clf = GridSearchCV(svr, parameters)

    clf.fit(vectorTrain, train_y)
    y_pred = clf.predict(vectorTest)

    # Best GridSearch params
    #print clf.best_params_

    print "Train Accuracy :: ", accuracy_score(train_y, clf.predict(vectorTrain))
    print "Test Accuracy  :: ", accuracy_score(test_y, y_pred)

    print "Elapsed time of diadoxika: ", time.time() - start_time_diadox

    # PipeLine.
    start_time_pipeline = time.time()

    pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words)),
        ('tfidf', TfidfTransformer()),
        # ('tfidf_v', TfidfVectorizer(stop_words)),
        ('lsa', TruncatedSVD(n_components=100)),
        ('norm', Normalizer(norm="l2", copy=True)),
        ('clf', svm.SVC(kernel='linear', C=1.0))
        # ('clf', svm.SVC(kernel='rbf', C=1.0, gamma='auto'))
    ])

    predicted_train = pipeline.fit(train_x['Content'], train_y).predict(train_x['Content'])
    # Now evaluate all steps on test set
    predicted_test = pipeline.predict(test_x['Content'])

    print "Train Accuracy :: ", accuracy_score(train_y, predicted_train)
    print "Test Accuracy  :: ", accuracy_score(test_y, predicted_test)

    print "Elapsed time of pipeline: ", time.time() - start_time_pipeline

    print 'svmClassifier finished!\n'


if __name__ == '__main__':
    svm_classifier()
