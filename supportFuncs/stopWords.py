from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


stop_words = set(ENGLISH_STOP_WORDS)


def add_and_return_stop_words():
    stop_words.add('said')
    stop_words.add('He')
    stop_words.add("He's")
    stop_words.add("he's")
    stop_words.add('It')
    stop_words.add("It's")
    stop_words.add("it's")
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
    stop_words.add("I'm")
    stop_words.add("I'd")
    stop_words.add('does')
    stop_words.add("doesn't")
    stop_words.add('week')
    stop_words.add('year')
    stop_words.add('Year')
    stop_words.add("Year's")
    stop_words.add('years')
    stop_words.add('want')
    stop_words.add('make')
    stop_words.add('come')
    stop_words.add('came')
    stop_words.add('new')
    # TODO - Add more stopWords..
    return stop_words


def get_stop_words():
    return add_and_return_stop_words()


# Run and print stopWords directly:
if __name__ == '__main__':
    print((get_stop_words()))
