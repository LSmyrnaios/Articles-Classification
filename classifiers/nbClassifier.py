import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import numpy as np
# from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from supportFuncs import stopWords, readDatasets, appendTitleToContentXtimes, crossValidation


def nb_classifier(stop_words, train_data, test_data, use_pipeline):
    print('Running nbClassifier...\n')

    headers = ['RowNum', 'Id', 'Title', 'Content', 'Category']
    # print(headers[2:4]) #DEBUG!

    # Split train_dataset into 0.7% train and 0.3% test.
    train_x, test_x, train_y, test_y = train_test_split(train_data[headers[2:4]], train_data[headers[-1]], train_size=0.7, test_size=0.3)

    # Train and Test dataset size details
    print("Train_x Shape :: ", train_x.shape)
    print("Train_y Shape :: ", train_y.shape)
    print("Test_x Shape :: ", test_x.shape)
    print("Test_y Shape :: ", test_y.shape)
    print("Train_x colums ::", train_x.columns)

    train_x, test_x = appendTitleToContentXtimes.append_title_to_content_x_times(train_x, test_x, 1)

    # LE
    # le = preprocessing.LabelEncoder()
    # train_x = le.fit_transform(train_x)
    # test_x = le.fit(test_x)

    # print train_x['Content'][1] #DEBUG!

    # List to be returned later.
    scores = []

    if use_pipeline:
        print('\nRunning pipeline-version of nbClassifier...')

        # PipeLine.
        start_time_pipeline = time.time()

        pipeline = Pipeline([
            ('vect', CountVectorizer(input=stop_words)),
            # ('tfidf', TfidfTransformer()),
            # ('tfidf_v', TfidfVectorizer(stop_words)),
            # ('norm', Normalizer(norm="l2", copy=True)),
            ('clf', MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True))
        ])

        pipeline = pipeline.fit(train_x['Content'], train_y)
        # accuracies = []
        # for i in range(10):
        predicted_train = pipeline.predict(train_x['Content'])
        # Now evaluate all steps on test set
        predicted_test = pipeline.predict(test_x['Content'])
        print("Train Accuracy :: ", accuracy_score(train_y, predicted_train))
        print("Test Accuracy  :: ", accuracy_score(test_y, predicted_test))
        # accuracies.append(accuracy_score(test_y, predicted_test))

        # print 'CrossValidation mean accuracy: ', np.mean(accuracies)

        print("Elapsed time of pipeline: ", time.time() - start_time_pipeline)

    else:
        print('\nRunning successional-version of nbClassifier...')

        start_time_successional = time.time()

        # Count Vectorizer
        count_vectorizer = CountVectorizer(input=stop_words)
        vectorTrain = count_vectorizer.fit_transform(train_x['Content'])
        vectorTest = count_vectorizer.transform(test_x['Content'])
        print("VectorTrain shape::", vectorTrain.shape)
        print("VectorTest shape::", vectorTest.shape)

        # TfidfTransformer
        # tfidf = TfidfTransformer()
        # vectorTrain = tfidf.fit_transform(vectorTrain)
        # vectorTest = tfidf.transform(vectorTest)

        # TfidfVectorizer (it does the job of CountVectorizer & TfidfTransformer together)
        # tfidf_v = TfidfVectorizer(stop_words)
        # vectorTrain = tfidf_v.fit_transform(train_x['Content'])
        # vectorTest = tfidf_v.transform(test_x['Content'])

        # Here we don't use LSA, as it has some issues (negative numbers).

        # Normalizer
        # norm = Normalizer(norm="l2", copy=True)
        # vectorTrain = norm.fit_transform(vectorTrain)
        # vectorTest = norm.transform(vectorTest)

        # CLF
        clf = MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)

        print('Running crossValidation on NaiveBayes...')
        scores = crossValidation.get_scores_from_cross_validation(clf, vectorTrain, train_y)

        # GridSearch
        # parameters = {'alpha': [10, 2, 1, 0.5, 0.1, 0.01, 0.001, 0.0001]}
        # svr = MultinomialNB()
        # clf = GridSearchCV(svr, parameters)

        # clf.fit(vectorTrain, train_y)
        # y_pred = clf.predict(vectorTest)

        # print "Train Accuracy :: ", accuracy_score(train_y, clf.predict(vectorTrain))
        # print "Test Accuracy :: ", accuracy_score(test_y, y_pred)

        # Best GridSearch params
        # print clf.best_params_

        print("Elapsed time of successional-run: ", time.time() - start_time_successional)

    print('nbClassifier finished!\n')
    return scores


# Run nbClassifier directly:
if __name__ == '__main__':
    dynamic_datasets_path = '..'
    data = readDatasets.read_dataset(dynamic_datasets_path)
    trainData = data[0]
    testData = data[1]
    usePipeline = False

    nb_classifier(stopWords.get_stop_words(), trainData, testData, usePipeline)
    exit()
