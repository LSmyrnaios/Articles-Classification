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


def euclideanDistance(instance1, instance2, length):
    distance = 0
    #print length
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    distance = math.sqrt(distance)
    #print(distance)
    return  distance

def manhattan_distance(start, end):
    return sum(abs(e - s) for s,e in zip(start, end))


def getNeighbors(trainingSet, testInstance, k, train_data):
    distances = []
    length = len(testInstance)
    for x in range(len(trainingSet)):
        #dist = euclideanDistance(testInstance, trainingSet[x], length)
        dist = manhattan_distance(testInstance, trainingSet[x])
        distances.append((trainingSet[x], dist, train_data['Category'][x]))
        #print distances[x][1]
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        #print '-------------------'
        #print distances[x][1], distances[x][2]
        neighbors.append((distances[x][0], distances[x][2]))
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet['Category'][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def knn_classifier(stop_words, train_data, test_data):  # It's uncertain if we will implement pipeline for knn..

    print 'Running knnClassifier...\n'

    headers = ['RowNum', 'Id', 'Title', 'Content', 'Category']
    print(headers[2:4]) #DEBUG!

    # Split train_dataset into 0.7% train and .03% test.
    train_x, test_x, train_y, test_y = train_test_split(train_data[headers[2:5]], train_data[headers[-1]],
                                                        train_size=0.7, test_size=0.3)

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
    # tfidf = TfidfTransformer()
    # vectorTrain = tfidf.fit_transform(vectorTrain)
    # vectorTest = tfidf.transform(vectorTest)

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
    # norm = Normalizer(norm="l2", copy=True)
    # vectorTrain = norm.fit_transform(vectorTrain)
    # vectorTest = norm.transform(vectorTest)

    predictions = []
    # accuracies = []
    # kValue = randint(1, 10)  # Assumed K value
    kValue = 10
    count = 0

    # for i in range(10):  # 10-fold-validation
    for x in range(100):
        neighbors = getNeighbors(vectorTrain, vectorTest[x], kValue, train_data)
        # print(neighbors)
        result = getResponse(neighbors)
        # print result
        predictions.append(result)
        # print('> predicted=' + repr(result) + ', actual=' + repr(train_data['Category'][8586+x]))
        if result == train_data['Category'][8586+x]:
            count += 1

    # print 'Test', test_x[1:2], 'Pred', predictions[0]
    # accuracies.append(getAccuracy(test_data, predictions))
    # print('Accuracy: ' + repr(accuracy) + '%')
    print("Got right", count, "out of", 100)

    # Final accuracy after crossValidation
    # print accuracies
    # print np.mean(accuracies)

    print "Elapsed time of successional-run: ", time.time() - start_time_successional


# Run knnClassifier directly:
if __name__ == '__main__':
    data = readDatasets.read_dataset()
    trainData = data[0]
    testData = data[1]

    knn_classifier(stopWords.get_stop_words(), trainData, testData)
