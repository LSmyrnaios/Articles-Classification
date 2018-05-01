import time
import multiprocessing
from sklearn import svm
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import cross_val_predict
from supportFuncs import stopWords, readDatasets, appendTitleToContentXtimes, crossValidation, writePredictionsToCsv


# This method is the optimized SVM-Classifer.
def my_method_classifier(stop_words, train_data, test_data):
    print 'Running myMethodClassifier...\n'

    # headers = ['RowNum', 'Id', 'Title', 'Content', 'Category']
    # print(headers[2:4]) #DEBUG!

    # Split train_dataset into 0.7% train and 0.3% test.
    # train_data, test_data, train_y, test_y = train_test_split(train_data[headers[2:4]], train_data[headers[-1]], train_size=1, test_size=0)

    # Train and Test dataset size details
    # print "Train_x Shape :: ", train_data.shape
    # print "Train_y Shape :: ", train_y.shape
    # print "Test_x Shape :: ", test_data.shape
    # print "Test_y Shape :: ", test_y.shape
    # print "Train_x colums ::", train_data.columns

    train_data, test_data = appendTitleToContentXtimes.append_title_to_content_x_times(train_data, test_data, 1)

    # LE
    # le = preprocessing.LabelEncoder()
    # train_data = le.fit_transform(train_data)
    # test_data = le.fit(test_data)

    # print train_data['Content'][1] #DEBUG!

    # List to be returned later.
    scores = []

    print 'Running successional-version of myMethodClassifier...'

    start_time_successional = time.time()

    # Count Vectorizer
    count_vectorizer = CountVectorizer(stop_words)
    vectorTrain = count_vectorizer.fit_transform(train_data['Content'])
    vectorTest = count_vectorizer.transform(test_data['Content'])
    print "VectorTrain shape::", vectorTrain.shape
    print "VectorTest shape::", vectorTest.shape

    # TfidfTransformer
    tfidf = TfidfTransformer()
    vectorTrain = tfidf.fit_transform(vectorTrain)
    vectorTest = tfidf.transform(vectorTest)

    # TfidfVectorizer (it does the job of CountVectorizer & TfidfTransformer together)
    # tfidf_v = TfidfVectorizer(stopWords)
    # vectorTrain = tfidf_v.fit_transform(train_data['Content'])
    # vectorTest = tfidf_v.transform(test_data['Content'])

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

    print 'Running crossValidation on MyMethod...'
    scores = crossValidation.get_scores_from_cross_validation(clf, vectorTrain, train_data['Category'])

    # GridSearch (find the best parameters)
    # parameters = {'kernel': ('linear', 'rbf'), 'C': [1.5, 10], 'gamma': [0, 'auto']}
    # svr = svm.SVC()
    # clf = GridSearchCV(svr, parameters)

    clf.fit(vectorTrain, train_data['Category'])
    y_pred = clf.predict(vectorTest)

    # print "Train Accuracy :: ", accuracy_score(train_data['Category'], clf.predict(vectorTrain))
    # print "Test Accuracy :: ", accuracy_score(train_data['Category'], y_pred)

    #y_pred = cross_val_predict(clf, X=vectorTrain, y=vectorTest, cv=10, n_jobs=multiprocessing.cpu_count())
    writePredictionsToCsv.write_predictions_to_csv(y_pred, test_data)

    # Best GridSearch params
    # print clf.best_params_

    print "Elapsed time of successional-run: ", time.time() - start_time_successional

    print 'MyMethodClassifier finished!\n'
    return scores


# Run myMethodClassifier directly:
if __name__ == '__main__':
    data = readDatasets.read_dataset()
    trainData = data[0]
    testData = data[1]

    my_method_classifier(stopWords.get_stop_words(), trainData, testData)
