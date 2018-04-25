from classifiers import rfClassifier, svmClassifier, nbClassifier


if __name__ == '__main__':

    # TODO - Open an outputCsvFile and write the scores which we will recieve from the classifiers.

    rfClassifier.rf_classifier()
    svmClassifier.svm_classifier()
    nbClassifier.nb_classifier()
