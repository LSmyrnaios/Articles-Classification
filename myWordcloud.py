# coding=utf-8
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import csv

stopwords = set(STOPWORDS)


def show_wordcloud(data, title=None):
    print("Creating wordcloud " + title + ' img...')
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40,
        scale=5,
        random_state=1  # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    # plt.imshow(wordcloud)
    # plt.show()
    plt.imsave(title, wordcloud)


if __name__ == '__main__':

    businessStr = ''
    politicsStr = ''
    footballStr = ''
    filmStr = ''
    technologyStr = ''

    with open('train_set.csv', 'rb') as csvfile:
        spamreader = csv.DictReader(csvfile, delimiter='\t', quotechar='|')

        for row in spamreader:
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

    stopwords.add('said')
    stopwords.add('told')
    stopwords.add('says')

    show_wordcloud(businessStr, 'Business')
    show_wordcloud(politicsStr, 'Politics')
    show_wordcloud(footballStr, 'Football')
    show_wordcloud(filmStr, 'Film')
    show_wordcloud(technologyStr, 'Technology')
