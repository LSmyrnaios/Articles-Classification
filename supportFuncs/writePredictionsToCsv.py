import csv


def write_predictions_to_csv(predicted_data, test_data):
    # print predicted_data
    # print test_data.shape

    with open('Resources/csv/testSet_categories.csv', mode='w', encoding="utf8", newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['ID'] + ['Predicted_Category'])  # Write headers.
        for x in range(len(test_data)):
            csv_writer.writerow([test_data['Id'][x]] + [predicted_data[x]])  # Write id-category pairs.
