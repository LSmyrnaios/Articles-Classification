import time
from sklearn.model_selection import train_test_split
from supportFuncs import stopWords, readDatasets, appendTitleToContentXtimes, crossValidation



def knn_classifier(stop_words, train_data, test_data):  # It's uncertain if we will implement pipeline for knn..

    print 'Running knnClassifier...\n'

    headers = ['RowNum', 'Id', 'Title', 'Content', 'Category']
    # print(headers[2:4]) #DEBUG!

    # Split train_dataset into 0.7% train and .03% test.
    train_x, test_x, train_y, test_y = train_test_split(train_data[headers[2:4]], train_data[headers[-1]], train_size=0.7, test_size=0.3)

    # LE (currently not used..)
    # le = preprocessing.LabelEncoder()
    # y = le.fit_transform(train_data["Category"])
    # print 'y : ', set(y) #DEBUG!

    # Train and Test dataset size details
    print "Train_x Shape :: ", train_x.shape
    print "Train_y Shape :: ", train_y.shape
    print "Test_x Shape :: ", test_x.shape
    print "Test_y Shape :: ", test_y.shape
    print "Train_x colums ::", train_x.columns

    train_x, test_x = appendTitleToContentXtimes.append_title_to_content_x_times(train_x, test_x, 1)

    start_time_successional = time.time()

    # TODO - Implement the KNN.

    print "Elapsed time of successional-run: ", time.time() - start_time_successional


# Run knnClassifier directly:
if __name__ == '__main__':

    data = readDatasets.read_dataset()
    trainData = data[0]
    testData = data[1]

    knn_classifier(stopWords.get_stop_words(), trainData, testData)
