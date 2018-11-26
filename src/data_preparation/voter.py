#!/usr/bin/env python3
# coding=utf-8
import csv
from itertools import chain
from typing import Dict, Tuple, NamedTuple


class Sample(NamedTuple):
    index: int
    joke: str
    category: int


class SamplesSet(NamedTuple):
    filename: str
    label: str


class LabelledSample(NamedTuple):
    sample: Sample
    label: str


Diff = Tuple[LabelledSample, ...]


class Result(NamedTuple):
    samples: Dict[int, Sample]
    inconclusive: Dict[int, Diff]
    diff: Dict[int, Diff]
    insufficient: Dict[int, Diff]


def split_line(line: str) -> Tuple[int, str, int]:
    try:
        n, rest = line.split(';', maxsplit=1)
        joke, cat = rest.rsplit(';', maxsplit=1)
        return int(n), joke.strip('"'), int(cat)
    except Exception:
        return -1, '', -1


def read_samples(filename: str) -> Dict[int, Sample]:
    with open(filename, encoding='utf-8') as f:
        return {
            n: Sample(n, joke, cat)
            for n, joke, cat in map(split_line, f)
        }


def group_by_cat(samples: Tuple[LabelledSample, ...]) -> Dict[int, Tuple[LabelledSample, ...]]:
    result = {}
    for sample in samples:
        result.setdefault(sample.sample.category, []).append(sample)
    return result


def compare_samples(set1: SamplesSet, set2: SamplesSet, set3: SamplesSet) -> Result:
    samp1 = read_samples(set1.filename)
    samp2 = read_samples(set2.filename)
    samp3 = read_samples(set3.filename)
    samples = samp1, samp2, samp3
    sets = set1, set2, set3

    keys = set(chain(samp1.keys(), samp2.keys(), samp3.keys()))
    result = Result({}, {}, {}, {})

    def vote(key: int):
        labelled_samples = (
            LabelledSample(samp.get(key), set.label)
            for samp, set in zip(samples, sets)
        )
        not_nones = tuple(
            samp
            for samp in labelled_samples if samp.sample is not None
        )
        if len(not_nones) < 2:
            result.insufficient[key] = not_nones
            return

        groups = group_by_cat(not_nones)

        # all samples have same category
        if len(groups) == 1:
            sample = groups.popitem()[1][0]
            result.samples[key] = sample.sample
            return

        result.diff[key] = not_nones

        # all samples differ in category
        if len(groups) == len(not_nones):
            result.inconclusive[key] = not_nones
            return

        result.samples[key] = max(groups.items(), key=lambda i: len(i[1]))[1][0].sample  # :/

    for key in keys:
        vote(key)

    return result


if __name__ == '__main__':
    result = compare_samples(SamplesSet('../../data/samples_an.csv', 'Anton'), SamplesSet('../../data/samples_art.csv', 'Artyom'),
                  SamplesSet('../../data/samples_i.csv', 'Ivan'))

    with open('samples_result.csv', 'w+', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        for sample in result.samples.values():
            writer.writerow([sample.index, sample.joke, sample.category])

    with open('diff.txt', 'w+', encoding='utf-8') as f:
        f.write(repr(result.diff))

    with open('inconclusive.txt', 'w+', encoding='utf-8') as f:
        f.write(repr(result.inconclusive))

    with open('insufficient.txt', 'w+', encoding='utf-8') as f:
        f.write(repr(result.insufficient))
