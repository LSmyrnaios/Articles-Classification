from myWordcloud import my_wordcloud
from classifiers.__init__ import run_all_classifiers
from supportFuncs import stopWords


# Run everything:
if __name__ == '__main__':

    stop_words = stopWords.get_stop_words()

    my_wordcloud(stop_words)
    run_all_classifiers(stop_words)
