# coding=utf-8
from wordcloud import WordCloud
from supportFuncs import stopWords
import csv


def show_wordcloud(stop_words, data, title=None):
    print("Creating wordcloud \"" + title + '\" img...')
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stop_words,
        max_words=200,
        max_font_size=40,
        scale=5,
        random_state=1
    ).generate(str(data)).to_file("Resources/csv/" + title + ".png")


def my_wordcloud():

    print 'Running myWordcloud...\n'

    #print 'StopWords ', stop_words

    businessStr = ''
    politicsStr = ''
    footballStr = ''
    filmStr = ''
    technologyStr = ''

    with open('Resources/csv/train_set.csv', 'rb') as csvfile:
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

    stop_words = stopWords.get_stop_words()

    show_wordcloud(stop_words, businessStr, 'Business')
    show_wordcloud(stop_words, politicsStr, 'Politics')
    show_wordcloud(stop_words, footballStr, 'Football')
    show_wordcloud(stop_words, filmStr, 'Film')
    show_wordcloud(stop_words, technologyStr, 'Technology')

    print 'myWordcloud finished!\n'


# Run myWordcloud directly:
if __name__ == '__main__':
    my_wordcloud()
