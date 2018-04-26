from classifiers import rfClassifier, svmClassifier, nbClassifier
import csv


def run_all_classifiers():
    print 'Running the classifiers...\n'
    usePipeline = False

    nbScores = nbClassifier.nb_classifier(usePipeline)
    rfScores = rfClassifier.rf_classifier(usePipeline)
    svmScores = svmClassifier.svm_classifier(usePipeline)
    #     knnScores = knnClassifier.knn_classifier(usePipeline)
    # mymethodScores = mymethodClassifier.mymethod_classifier(usePipeline)

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
    run_all_classifiers()
