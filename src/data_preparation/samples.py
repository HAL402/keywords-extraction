#!/usr/bin/env python3
# coding=utf-8
import pickle
from operator import attrgetter
from pprint import pprint
from collections import Counter
from typing import Sequence, Set, Tuple, Iterable, Dict, Any, Callable

import nltk
from maru.grammeme.pos import PartOfSpeech
from maru.morph import Morph
from nltk.corpus import stopwords

from lemmatize import read_dump, read_dataset, DUMP_FILE

MorphFilter = Callable[[Morph], bool]


class SampleMeta:
    __slots__ = 'sample', 'morphs', 'bigrams', 'lemmas'
    _get_lemma: Callable[[Morph], str] = attrgetter('lemma')

    def __init__(self, sample: str, morphs: Sequence[Morph], morph_filter: MorphFilter):
        self.sample = sample
        self.morphs = morphs

        self.lemmas = self._get_lemmas(morphs, morph_filter)
        self.bigrams = self._get_bigrams(morphs)

    @staticmethod
    def _get_bigrams(morphs: Sequence[Morph]) -> Set[Tuple[str, str]]:
        it = iter(morphs)
        next(it, None)
        return set(zip(
            map(SampleMeta._get_lemma, morphs),
            map(SampleMeta._get_lemma, it)
        ))

    @staticmethod
    def _get_lemmas(morphs: Sequence[Morph], morph_filter: MorphFilter) -> Set[str]:
        return set(SampleMeta._get_lemma(m) for m in filter(morph_filter, morphs))

    def __hash__(self) -> int:
        return hash(self.sample)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SampleMeta):
            return False
        return self.sample == other.sample


def min_jaccard_distance(selected: Set[SampleMeta], joke: SampleMeta, attr: Callable[[SampleMeta], bool]) -> float:
    if not len(selected):
        return 1

    return min(
        nltk.jaccard_distance(attr(sample), attr(joke))
        for sample in selected
    )


def words_counter(morphs: Iterable[Sequence[Morph]]) -> Dict[str, int]:
    return Counter(
        morph.lemma
        for line in morphs
        for morph in line
    )


def compose_selection(
       dataset: Iterable[str], morphs: Iterable[Sequence[Morph]], morph_filter: MorphFilter,
       selection_size: int, distance_threshold: float, bigrams_distance_threshold: float,
       selected_meta=None
) -> Iterable[Tuple[int, str]]:
    selected_meta = selected_meta or set()
    lemmas_getter = attrgetter('lemmas')
    bigrams_getter = attrgetter('bigrams')

    for i, (sample, sample_morphs) in enumerate(zip(dataset, morphs)):
        if len(selected_meta) >= selection_size:
            break
        try:
            meta = SampleMeta(sample, sample_morphs, morph_filter)
            distance = min_jaccard_distance(selected_meta, meta, lemmas_getter)
            bigrams_distance = min_jaccard_distance(selected_meta, meta, bigrams_getter)
        except Exception as e:
            print(e)
            continue

        if distance >= distance_threshold and bigrams_distance >= bigrams_distance_threshold:
            selected_meta.add(meta)
            yield i, sample

        if i % 100 == 0:
            print(f'Drop ratio: {i/len(selected_meta)}, seen {i}, picked {len(selected_meta)}')

    else:
        print('Exhausted')
        yield from compose_selection(
            dataset, morphs, morph_filter, selection_size - len(selected_meta),
            distance_threshold - 0.2, bigrams_distance_threshold - 0.1,
            selected_meta
        )

    print(f'Total processed strings: {i}')
    return selected_meta


def cycle(cb):
    while True:
        yield from cb()


if __name__ == '__main__':
    SELECTION_SIZE = 300
    DISTANCE_THRESHOLD = 0.96
    BIGRAMS_THRESHOLD = 0.98
    WORD_FREQUENCY_THRESHOLD = 3
    STOPWORDS = set(stopwords.words('russian'))
    FILTERED_POS = {
        PartOfSpeech.UNKNOWN,
        PartOfSpeech.PUNCTUATION,
        PartOfSpeech.PARTICLE,
        PartOfSpeech.CONJUNCTION,
        PartOfSpeech.ADPOSITION,
        PartOfSpeech.INTERJECTION,
        PartOfSpeech.DETERMINANT,
        PartOfSpeech.NUMERICAL,
        PartOfSpeech.PRONOUN
    }

    morphs = read_dump(DUMP_FILE)
    frequencies = words_counter(morphs)

    # pprint(frequencies)

    def morph_filter(morph: Morph) -> bool:
        return (
            morph.tag.pos not in FILTERED_POS and
            morph.lemma not in STOPWORDS and
            frequencies.get(morph.lemma, 0) > WORD_FREQUENCY_THRESHOLD
        )

    dataset = read_dataset('../../prepared_data/jokes_cleaned.json')
    morphs = cycle(lambda: read_dump('../../data/lemmas_dump'))

    with open('../data/samples', 'w+b') as f:
        for sample in compose_selection(dataset, morphs, morph_filter, SELECTION_SIZE, DISTANCE_THRESHOLD, BIGRAMS_THRESHOLD):
            print(sample)
            pickle.dump(sample, f)

