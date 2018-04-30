import math
import operator
import time
import numpy as np
from random import randint
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from supportFuncs import stopWords, readDatasets, appendTitleToContentXtimes, crossValidation


def euklidianDist(x, xi):
    dist = 0.0
    for i in range(len(x)-1):
        dist += pow((float(x[i])-float(xi[i])), 2)
    dist = math.sqrt(dist)
    return dist


def knn_predict(train_data, test_data, k_value):

    for i in test_data:
        eu_Distance = []
        knn = []
        good = 0
        bad = 0
        for j in train_data:
            eu_dist = euklidianDist(i, j)
            eu_Distance.append((j[5], eu_dist))
            eu_Distance.sort(key=operator.itemgetter(1))
            knn = eu_Distance[:k_value]
            for k in knn:
                if k[0] == 'g':
                    good += 1
                else:
                    bad += 1
        if good > bad:
            i.append('g')
        elif good < bad:
            i.append('b')
        else:
            i.append('NaN')


def accuracy(test_data):
    correct = 0
    for i in test_data:
        if i[5] == i[6]:
            correct += 1

    accuracy = float(correct) / len(test_data) * 100
    return accuracy


def knn_classifier(stop_words, train_data, test_data):  # It's uncertain if we will implement pipeline for knn..

    print 'Running knnClassifier...\n'

    headers = ['RowNum', 'Id', 'Title', 'Content', 'Category']
    # print(headers[2:4]) #DEBUG!

    # Split train_dataset into 0.7% train and .03% test.
    train_x, test_x, train_y, test_y = train_test_split(train_data[headers[2:4]], train_data[headers[-1]], train_size=0.7, test_size=0.3)

    # Train and Test dataset size details
    print "Train_x Shape :: ", train_x.shape
    print "Train_y Shape :: ", train_y.shape
    print "Test_x Shape :: ", test_x.shape
    print "Test_y Shape :: ", test_y.shape
    print "Train_x colums ::", train_x.columns

    train_x, test_x = appendTitleToContentXtimes.append_title_to_content_x_times(train_x, test_x, 1)

    start_time_successional = time.time()

    # LE
    # le = preprocessing.LabelEncoder()
    # train_x = le.fit_transform(train_x)
    # test_x = le.fit(test_x)

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

    # TODO - Implement the KNN.
    kValue = randint(1, 10)  # Assumed K value
    knn_predict(train_x, test_x, kValue)
    print 'Accuracy: ', accuracy(test_data)

    print "Elapsed time of successional-run: ", time.time() - start_time_successional


# Run knnClassifier directly:
if __name__ == '__main__':

    data = readDatasets.read_dataset()
    trainData = data[0]
    testData = data[1]

    knn_classifier(stopWords.get_stop_words(), trainData, testData)
