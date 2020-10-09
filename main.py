wordCloudWindowsProblem = False

try:
    from myWordcloud import my_wordcloud
except:
    wordCloudWindowsProblem = True


from classifiers.__init__ import run_all_classifiers
from supportFuncs import stopWords


# Run everything:
if __name__ == '__main__':
    stop_words = stopWords.get_stop_words()
    usePipeline = False     # Pipeline currently not running crossValidation. Use the regular way.

    dynamic_datasets_path = ''

    if not wordCloudWindowsProblem:
        my_wordcloud(stop_words, dynamic_datasets_path)

    run_all_classifiers(stop_words, usePipeline, dynamic_datasets_path)
    exit()
