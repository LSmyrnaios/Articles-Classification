from classifiers import rfClassifier, svmClassifier, nbClassifier
from supportFuncs import stopWords, readDatasets
import csv


def run_all_classifiers(stop_words, use_pipeline):

    data = readDatasets.read_dataset()
    train_data = data[0]
    test_data = data[1]

    print 'Running the classifiers...\n'

    nbScores = nbClassifier.nb_classifier(stop_words, train_data, test_data, use_pipeline)
    rfScores = rfClassifier.rf_classifier(stop_words, train_data, test_data, use_pipeline)
    svmScores = svmClassifier.svm_classifier(stop_words, train_data, test_data, use_pipeline)
    # knnScores = knnClassifier.knn_classifier(stop_words, train_data, test_data, use_pipeline)
    # mymethodScores = mymethodClassifier.mymethod_classifier(stop_words, train_data, test_data, use_pipeline)

    # Open an outputCsvFile and write the scores which we will recieve from the classifiers.

    print 'Writing classifiers\' scores to the outputCsvFile...\n'

    with open('Resources/csv/EvaluationMetric_10fold.csv', 'wb') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        # Write the Headers (first row & column).
        csvWriter.writerow(['StatisticMeasure'] + ['NaiveBayes'] + ['Random Forest'] + ['SVM'] + ['KNN'] + ['My Method'])
        csvWriter.writerow(['Accuracy'] + [nbScores[0]] + [rfScores[0]] + [svmScores[0]] )  # + [knnScores[0]] + [mymethodScores[0]])
        csvWriter.writerow(['Precision'] + [nbScores[1]] + [rfScores[1]] + [svmScores[1]] )  # + [knnScores[1]]] + [mymethodScores[1]])
        csvWriter.writerow(['Recall'] + [nbScores[2]] + [rfScores[2]] + [svmScores[2]] )  # + [knnScores[2]] + [mymethodScores[2]])
        csvWriter.writerow(['F-Measure'] + [nbScores[3]] + [rfScores[3]] + [svmScores[3]] )  # + [knnScores[3]]] + [mymethodScores[3]])

    print 'Finished writing to the outputCsvFile!'


# Run all classifiers:
if __name__ == '__main__':
    usePipeline = False
    run_all_classifiers(stopWords.get_stop_words(), usePipeline)
