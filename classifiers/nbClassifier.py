# from sklearn.cross_validation import train_test_split
import time

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.preprocessing import Normalizer
from supportFuncs import stopWords, splitDataSet, crossValidation


def nb_classifier(use_pipeline):

    print 'Running nbClassifier...\n'

    headers = ['RowNum', 'Id', 'Title', 'Content', 'Category']

    train_data = pd.read_csv('Resources/csv/train_set.csv', sep="\t")
    test_data = pd.read_csv('Resources/csv/test_set.csv', sep="\t")

    #print(headers[2:4])

    train_x, test_x, train_y, test_y = splitDataSet.split_dataset(train_data, 0.7, headers[2:4], headers[-1])

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(train_data["Category"])

    #print 'y : ', set(y)

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
        print '\nRunning pipeline-version of nbClassifier...'

        # PipeLine.
        start_time_pipeline = time.time()

        pipeline = Pipeline([
            ('vect', CountVectorizer(stop_words)),
            ('tfidf', TfidfTransformer()),
            # ('tfidf_v', TfidfVectorizer(stop_words)),
            ('norm', Normalizer(norm="l2", copy=True)),
            ('clf', MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True))
        ])

        predicted_train = pipeline.fit(train_x['Content'], train_y).predict(train_x['Content'])
        # Now evaluate all steps on test set
        predicted_test = pipeline.predict(test_x['Content'])

        print "Train Accuracy :: ", accuracy_score(train_y, predicted_train)
        print "Test Accuracy  :: ", accuracy_score(test_y, predicted_test)

        print "Elapsed time of pipeline: ", time.time() - start_time_pipeline

    else:
        print '\nRunning successional-version of nbClassifier...'

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
        # tfidf_v = TfidfVectorizer(stop_words)
        # vectorTrain = tfidf_v.fit_transform(train_x['Content'])
        # vectorTest = tfidf_v.transform(test_x['Content'])

        # Here we don't use LSA, as it has some issues (negative numbers).

        # Normalizer
        norm = Normalizer(norm="l2", copy=True)
        vectorTrain = norm.fit_transform(vectorTrain)
        vectorTest = norm.transform(vectorTest)

        # CLF
        clf = MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)

        scores = crossValidation.get_scores_from_cross_validation(clf, vectorTrain, train_y)

        # GridSearch
        # parameters = {'alpha': [10, 2, 1, 0.5, 0.1, 0.01, 0.001, 0.0001]}
        # svr = MultinomialNB()
        # clf = GridSearchCV(svr, parameters)

        # clf.fit(vectorTrain, train_y)
        # y_pred = clf.predict(vectorTest)
        #
        # print "Train Accuracy :: ", accuracy_score(train_y, clf.predict(vectorTrain))
        # print "Test Accuracy :: ", accuracy_score(test_y, y_pred)

        # Best GridSearch params
        # print clf.best_params_

        print "Elapsed time of successional-run: ", time.time() - start_time_successional


    print 'nbClassifier finished!\n'
    return scores


# Run nbClassifier directly:
if __name__ == '__main__':
    usePipeline = False
    nb_classifier(usePipeline)
