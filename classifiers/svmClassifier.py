# from sklearn.cross_validation import train_test_split
import time

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.preprocessing import Normalizer
from supportFuncs import stopWords, splitDataSet, crossValidation


def svm_classifier(use_pipeline):

    print 'Running svmClassifier...\n'

    headers = ['RowNum', 'Id', 'Title', 'Content', 'Category']

    train_data = pd.read_csv('Resources/csv/train_set.csv', sep="\t")
    test_data = pd.read_csv('Resources/csv/test_set.csv', sep="\t")

    # print(headers[2:4])

    # Split dataSet to 70-30.
    train_x, test_x, train_y, test_y = splitDataSet.split_dataset(train_data, 0.7, headers[2:4], headers[-1])

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

    stop_words = stopWords.get_stop_words()

    # print train_x['Content'][1]

    # List to be returned later.
    scores = []

    if use_pipeline:
        print '\nRunning pipeline-version of svmClassifier...'

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

    else:
        print '\nRunning successional-version of svmClassifier...'

        start_time_successional = time.time()

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
        # tfidf_v = TfidfVectorizer(stopWords)
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

        scores = crossValidation.get_scores_from_cross_validation(clf, vectorTrain, train_y)

        # GridSearch (find the best parameters)
        # parameters = {'kernel': ('linear', 'rbf'), 'C': [1.5, 10], 'gamma': [0, 'auto']}
        # svr = svm.SVC()
        # clf = GridSearchCV(svr, parameters)

        # clf.fit(vectorTrain, train_y)
        # y_pred = clf.predict(vectorTest)
        #
        # print "Train Accuracy :: ", accuracy_score(train_y, clf.predict(vectorTrain))
        # print "Test Accuracy :: ", accuracy_score(test_y, y_pred)

        # Best GridSearch params
        #print clf.best_params_

        print "Elapsed time of successional-run: ", time.time() - start_time_successional

    print 'svmClassifier finished!\n'
    return scores


# Run svmClassifier directly:
if __name__ == '__main__':
    usePipeline = False
    svm_classifier(usePipeline)
