import pandas as pd


def read_dataset():
    train_data = pd.read_csv('Resources/csv/train_set.csv', sep="\t")
    test_data = pd.read_csv('Resources/csv/test_set.csv', sep="\t")

    return [train_data, test_data]
