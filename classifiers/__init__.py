from classifiers import rfClassifier, svmClassifier, nbClassifier, MyMethodClassifier
from supportFuncs import stopWords, readDatasets
import csv


def run_all_classifiers(stop_words, use_pipeline):
    data = readDatasets.read_dataset()
    train_data = data[0]
    test_data = data[1]

    print('Running the classifiers...\n')

    nbScores = nbClassifier.nb_classifier(stop_words, train_data, test_data, use_pipeline)
    rfScores = rfClassifier.rf_classifier(stop_words, train_data, test_data, use_pipeline)
    svmScores = svmClassifier.svm_classifier(stop_words, train_data, test_data, use_pipeline)
    # knnScores = knnClassifier.knn_classifier(stop_words, train_data, test_data)
    mymethodScores = MyMethodClassifier.my_method_classifier(stop_words, train_data, test_data)

    '{:06.2f}'.format(3.141592653589793)

    # Open an outputCsvFile and write the scores which we will receive from the classifiers.

    print('Writing classifiers\' scores to the outputCsvFile...\n')

    with open('Resources/csv/EvaluationMetric_10fold.csv', mode='w', encoding="utf8") as csvfile:
        csvWriter = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        # Write the Headers (first row & column).
        csvWriter.writerow(['Statistic_Measure'] + ['Naive_Bayes'] + ['Random_Forest'] + ['SVM'] + ['KNN']
                           + ['My_Method'])

        # Write the scores.
        csvWriter.writerow(['Accuracy'] + ['{:.3}'.format(nbScores[0])] + ['{:.3}'.format(rfScores[0])]
                           + ['{:.3}'.format(svmScores[0])] + ['knn'] + ['{:.3}'.format(mymethodScores[0])])
        # + [knnScores[0]] + [mymethodScores[0]])
        csvWriter.writerow(['Precision'] + ['{:.3}'.format(nbScores[1])] + ['{:.3}'.format(rfScores[1])]
                           + ['{:.3}'.format(svmScores[1])] + ['knn'] + ['{:.3}'.format(mymethodScores[1])])
        # + [knnScores[1]]] + [mymethodScores[1]])
        csvWriter.writerow(['Recall'] + ['{:.3}'.format(nbScores[2])] + ['{:.3}'.format(rfScores[2])]
                           + ['{:.3}'.format(svmScores[2])] + ['knn'] + ['{:.3}'.format(mymethodScores[2])])
        # + [knnScores[2]] + [mymethodScores[2]])
        csvWriter.writerow(['F-Measure'] + ['{:.3}'.format(nbScores[3])] + ['{:.3}'.format(rfScores[3])]
                           + ['{:.3}'.format(svmScores[3])] + ['knn'] + ['{:.3}'.format(mymethodScores[3])])
        # + [knnScores[3]]] + [mymethodScores[3]])

    print('Finished writing to the outputCsvFile!')


# Run all classifiers:
if __name__ == '__main__':
    usePipeline = False
    run_all_classifiers(stopWords.get_stop_words(), usePipeline)
