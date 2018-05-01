import csv


def write_predictions_to_csv(predicted_data, test_data):
    #print predicted_data
    #print test_data.shape

    with open('Resources/csv/testSet_categories.csv', 'wb') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvWriter.writerow(['ID'] + ['Predicted_Category'])    # Write headers.
        for x in range(len(test_data)):
            csvWriter.writerow([test_data['Id'][x]] + [predicted_data[x]]) # Write id-category pairs.
