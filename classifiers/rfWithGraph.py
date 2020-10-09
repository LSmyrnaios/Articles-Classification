import os
import time
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from supportFuncs import stopWords, readDatasets, appendTitleToContentXtimes, crossValidation
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def rf_classifier_with_graph(stop_words, train_data, test_data, use_pipeline):

    print('Running rfClassifier...\n')

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
    accuraccies = []
    components = []

    if use_pipeline:
        print('\nRunning pipeline-version of rfClassifier...')

        # PipeLine-test.
        start_time_pipeline = time.time()

        pipeline = Pipeline([
            ('vect', CountVectorizer(input=stop_words)),
            ('tfidf', TfidfTransformer()),
            # ('tfidf_v', TfidfVectorizer(stopWords)),
            ('lsa', TruncatedSVD(n_components=100)),
            ('norm', Normalizer(norm="l2", copy=True)),
            ('clf', RandomForestClassifier(n_estimators=100))
        ])

        predicted_train = pipeline.fit(train_x['Content'], train_y).predict(train_x['Content'])
        # Now evaluate all steps on test set
        predicted_test = pipeline.predict(test_x['Content'])

        print("Train Accuracy :: ", accuracy_score(train_y, predicted_train))
        print("Test Accuracy  :: ", accuracy_score(test_y, predicted_test))

        print("Elapsed time of pipeline: ", time.time() - start_time_pipeline)

    else:
        print('\nRunning successional-version of rfClassifier...')

        # Count Vectorizer
        count_vectorizer = CountVectorizer(input=stop_words)

        # LSA
        for x in range(1, 101, 10):
            start_time_successional = time.time()

            vectorTrain = count_vectorizer.fit_transform(train_x['Content'])
            vectorTest = count_vectorizer.transform(test_x['Content'])
            print("VectorTrain shape::", vectorTrain.shape)
            print("VectorTest shape::", vectorTest.shape)

            # TfidfTransformer
            # tfidf = TfidfTransformer()
            # vectorTrain = tfidf.fit_transform(vectorTrain)
            # vectorTest = tfidf.transform(vectorTest)

            lsa = TruncatedSVD(n_components=x)
            vectorTrain = lsa.fit_transform(vectorTrain)
            vectorTest = lsa.transform(vectorTest)

            print(x, "- VectorTrain shape after LSA::", vectorTrain.shape)
            print(x, "- VectorTest shape after LSA::", vectorTest.shape)

            # CLF
            clf = RandomForestClassifier(n_estimators=100)

            print('Running crossValidation on RandomForest...')
            scores = crossValidation.get_scores_from_cross_validation(clf, vectorTrain, train_y)
            components.append(x)
            accuraccies.append(scores[0])

            print("Elapsed time of successional-run: ", time.time() - start_time_successional)

    x = np.array(components)
    y = np.array(accuraccies)

    x_new = np.linspace(x.min(), x.max(), 500)

    f = interp1d(x, y, kind='quadratic')
    y_smooth = f(x_new)
    plt.xlabel('n_components')
    plt.ylabel('Accuracy')
    plt.title('RandomForestClassifier Accuracy Graph')
    plt.plot(x_new, y_smooth, color='r')
    plt.scatter(x, y)
    plt.savefig(os.path.join(dynamic_datasets_path, 'Resources', 'images', 'RandomForestClassifier Accuracy Graph.png'))
    plt.show()
    print('rfClassifier finished!\n')
    return scores


# Run rfClassifier directly:
if __name__ == '__main__':
    dynamic_datasets_path = '..'
    data = readDatasets.read_dataset(dynamic_datasets_path)
    trainData = data[0]
    testData = data[1]
    usePipeline = False

    rf_classifier_with_graph(stopWords.get_stop_words(), trainData, testData, usePipeline)
    exit()
