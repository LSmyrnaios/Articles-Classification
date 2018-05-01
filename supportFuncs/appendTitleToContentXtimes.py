

def append_title_to_content_x_times(train_data, test_data, x_times):

    for row in train_data:
        train_data['Content'] += x_times * (' ' + train_data['Title'])

    for row in test_data:
        test_data['Content'] += x_times * (' ' + test_data['Title'])

    return train_data, test_data
