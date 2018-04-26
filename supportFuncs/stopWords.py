from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


initial_stop_words = set(ENGLISH_STOP_WORDS)


def add_and_return_stop_words():
    initial_stop_words.add('said')
    initial_stop_words.add('He')
    initial_stop_words.add("He's")
    initial_stop_words.add("he's")
    initial_stop_words.add('It')
    initial_stop_words.add("It's")
    initial_stop_words.add("it's")
    initial_stop_words.add('got')
    initial_stop_words.add("don't")
    initial_stop_words.add('like')
    initial_stop_words.add("didn't")
    initial_stop_words.add('ago')
    initial_stop_words.add('went')
    initial_stop_words.add('did')
    initial_stop_words.add('day')
    initial_stop_words.add('just')
    initial_stop_words.add('thing')
    initial_stop_words.add('think')
    initial_stop_words.add('say')
    initial_stop_words.add('says')
    initial_stop_words.add('know')
    initial_stop_words.add('clear')
    initial_stop_words.add('despite')
    initial_stop_words.add('going')
    initial_stop_words.add('time')
    initial_stop_words.add('people')
    initial_stop_words.add('way')
    initial_stop_words.add("I'm")
    initial_stop_words.add("I'd")
    initial_stop_words.add('does')
    initial_stop_words.add("doesn't")
    # TODO - Add more stopWords..
    return initial_stop_words


def get_stop_words():
    return add_and_return_stop_words()


# Run and print stopWords directly:
if __name__ == '__main__':
    print get_stop_words()
