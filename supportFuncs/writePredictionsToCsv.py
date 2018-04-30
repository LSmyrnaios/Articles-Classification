import csv


def write_predictions_to_csv(predicted_data):

    with open('Resources/csv/testSet_categories.csv', 'wb') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        csvWriter.writerow(['ID'] + ['Predicted_Category'])    # Write headers.
        for row in predicted_data:
            csvWriter.writerow([row[0]] + [row[1]]) # Write id-category pairs.
