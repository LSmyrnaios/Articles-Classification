# coding=utf-8
import os

try:
    from wordcloud import WordCloud
except ImportError:
    import platform
    if platform.system() == 'Windows':
        print('WordCloud will not run, as Microsoft Visual C++ 14.0 is required to build it on Windows..!')
    exit(-1)

from supportFuncs import stopWords
import csv


def show_wordcloud(stop_words, data, title=None, dynamic_datasets_path=''):
    print('Creating WordCloud "' + title + '" image...')
    image_location = os.path.join(dynamic_datasets_path, 'Resources', 'images', title + '.png')
    WordCloud(
        background_color='black',
        stopwords=stop_words,
        max_words=200,
        max_font_size=40,
        scale=5,
        random_state=1
    ).generate(str(data)).to_file(image_location)


def my_wordcloud(stop_words, dynamic_datasets_path):
    print('Running myWordcloud...\n')

    # print 'StopWords ', stop_words

    businessStr = ''
    politicsStr = ''
    footballStr = ''
    filmStr = ''
    technologyStr = ''

    location_train = os.path.join(dynamic_datasets_path, 'Resources', 'datasets', 'train_set.csv')

    with open(location_train, mode='r', encoding="utf8") as csvfile:
        csvReader = csv.DictReader(csvfile, delimiter='\t', quotechar='|')

        for row in csvReader:
            category = row["Category"]
            if category == 'Business':
                businessStr += row["Content"]
            elif category == 'Politics':
                politicsStr += row["Content"]
            elif category == 'Football':
                footballStr += row["Content"]
            elif category == 'Film':
                filmStr += row["Content"]
            elif category == 'Technology':
                technologyStr += row["Content"]

    show_wordcloud(stop_words, businessStr, 'Business', dynamic_datasets_path)
    show_wordcloud(stop_words, politicsStr, 'Politics', dynamic_datasets_path)
    show_wordcloud(stop_words, footballStr, 'Football', dynamic_datasets_path)
    show_wordcloud(stop_words, filmStr, 'Film', dynamic_datasets_path)
    show_wordcloud(stop_words, technologyStr, 'Technology', dynamic_datasets_path)

    print('myWordcloud finished!\n')


# Run myWordcloud directly:
if __name__ == '__main__':
    dynamic_datasets_path = ''
    my_wordcloud(stopWords.get_stop_words(), dynamic_datasets_path)
    exit()
