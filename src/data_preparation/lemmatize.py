#!/usr/bin/env python3
# coding=utf-8
import json
import pickle
import random
from typing import List, Sequence, Iterable

import maru
from maru.analyzer import Analyzer
from maru.morph import Morph

DUMP_FILE = '../../data/lemmas_dump'


def read_dataset(file_path: str) -> List[str]:
    with open(file_path, encoding='utf-8') as f:
        return json.load(f)


def lemmatize(analyzer: Analyzer, samples: Iterable[str]) -> Iterable[Sequence[Morph]]:
    return (
        tuple(analyzer.analyze(sample))
        for sample in tokenize(samples)
    )


def tokenize(samples: Iterable[str]) -> Iterable[Sequence[str]]:
    return map(str.split, samples)


def save_dump(morphs: Iterable[Sequence[Morph]], file_path: str) -> None:
    with open(file_path, 'a+b') as f:
        for morph in morphs:
            pickle.dump(morph, f)


def read_dump(file_path: str) -> Iterable[Sequence[Morph]]:
    with open(file_path, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


if __name__ == '__main__':
    dataset = read_dataset('../../prepared_data/jokes_cleaned.json')
    random.seed(42)
    random.shuffle(dataset)

    analyzer = maru.get_analyzer(tagger='rnn', lemmatizer='pymorphy')

    lemmas = lemmatize(analyzer, dataset[:30000])
    save_dump(lemmas, DUMP_FILE)
