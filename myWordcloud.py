# coding=utf-8
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import csv

stop_words = set(ENGLISH_STOP_WORDS)


def show_wordcloud(data, title=None):
    print("Creating wordcloud " + title + ' img...')
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stop_words,
        max_words=200,
        max_font_size=40,
        scale=5,
        random_state=1
    ).generate(str(data)).to_file(title + ".png")


def addPreDefinedStopWords():
    stop_words.add('said')
    stop_words.add('he')
    stop_words.add('He')
    stop_words.add('it')
    stop_words.add('It')
    stop_words.add('got')
    stop_words.add("don't")
    stop_words.add('like')
    stop_words.add("didn't")
    stop_words.add('ago')
    stop_words.add('went')
    stop_words.add('did')
    stop_words.add('day')
    stop_words.add('just')
    stop_words.add('thing')
    stop_words.add('think')
    stop_words.add('say')
    stop_words.add('says')
    stop_words.add('know')
    stop_words.add('clear')
    stop_words.add('despite')
    stop_words.add('going')
    stop_words.add('time')
    stop_words.add('people')
    stop_words.add('way')
    # TODO - Add more stopWords..


if __name__ == '__main__':

    #print 'StopWords ', stop_words

    businessStr = ''
    politicsStr = ''
    footballStr = ''
    filmStr = ''
    technologyStr = ''

    with open('train_set.csv', 'rb') as csvfile:
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

    addPreDefinedStopWords()

    show_wordcloud(businessStr, 'Business')
    show_wordcloud(politicsStr, 'Politics')
    show_wordcloud(footballStr, 'Football')
    show_wordcloud(filmStr, 'Film')
    show_wordcloud(technologyStr, 'Technology')
