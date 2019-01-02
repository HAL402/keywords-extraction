#!/usr/bin/env python3
# coding=utf-8
import csv
import pickle

import maru

from noise_cleaner import normalize_string, deduplicate_spaces, punctuation_spaces


if __name__ == '__main__':
    NEW_JOKES_RAW = '../../data/new_jokes_raw.csv'  # в формате шутка;категория
    START_INDEX = 0  # максимальный индекс в текущем датасете + 1 (откуда начинать нумерацию)
    ITERATION = 0  # номер новых датасетов

    TRAIN = f'../../data/train{ITERATION}.csv'  # где весь юмор
    LEMMAS = f'../../data/lemmas_dump{ITERATION}'  # куда добавлять леммы

    assert START_INDEX != 0
    assert ITERATION != 0

    analyzer = maru.get_analyzer(tagger='rnn', lemmatizer='pymorphy')

    with open(NEW_JOKES_RAW, encoding='utf-8') as raw_file, \
         open(TRAIN, 'a', newline='', encoding='utf-8') as train, \
         open(LEMMAS, 'a+b') as lemmas:

        # reader = csv.reader(raw_file, delimiter=';')
        reader = (
            (x[0], int(x[1])) for x in (
            line.rsplit(';', maxsplit=1) for line in raw_file
        ))
        writer = csv.writer(train, delimiter=';', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        for i, (joke, cat) in enumerate(reader, start=START_INDEX):
            joke = punctuation_spaces(deduplicate_spaces(normalize_string(joke)))
            writer.writerow([i, joke, cat])

            lemma = tuple(analyzer.analyze(joke.split()))
            pickle.dump((i, lemma), lemmas)
