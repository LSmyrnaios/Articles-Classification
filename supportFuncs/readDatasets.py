import os

import pandas as pd


def read_dataset(dynamic_datasets_path=''):

    location_train = os.path.join(dynamic_datasets_path, 'Resources', 'datasets', 'train_set.csv')
    train_data = pd.read_csv(location_train, sep="\t")

    location_test = os.path.join(dynamic_datasets_path, 'Resources', 'datasets', 'test_set.csv')
    test_data = pd.read_csv(location_test, sep="\t")

    return [train_data, test_data]
