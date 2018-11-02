#!/usr/bin/env python3
# coding=utf-8
import string
import re
import unicodedata
from typing import List, Sequence, Dict

from src.common.utilities import read_dataset, save_dataset


def normalize_string(string: str) -> str:
    return unicodedata.normalize('NFC', string)


def deduplicate_spaces(string: str) -> str:
    """
    >>> deduplicate_spaces('')
    ''
    >>> deduplicate_spaces('abc')
    'abc'
    >>> deduplicate_spaces('abc def')
    'abc def'
    >>> deduplicate_spaces('abc  def')
    'abc def'
    >>> deduplicate_spaces('abc\\t def')
    'abc def'
    >>> deduplicate_spaces('abc\\ndef')
    'abc def'
    >>> deduplicate_spaces('abc\\n\\tdef')
    'abc def'
    """
    return ' '.join(string.split())


_PUNCTUATION_PATTERN = re.compile(
    rf'(?P<before>.+?)(?P<punctuation>[{string.punctuation}])(?P<after>[^{string.punctuation}]*)')


def punctuation_spaces(s: str) -> str:
    """
    >>> punctuation_spaces('')
    ''
    >>> punctuation_spaces('abc')
    'abc'
    >>> punctuation_spaces('abc123')
    'abc123'
    >>> punctuation_spaces('abc - def')
    'abc - def'
    >>> punctuation_spaces('abc- def')
    'abc - def'
    >>> punctuation_spaces('abc -def')
    'abc - def'
    >>> punctuation_spaces('abc-def')
    'abc - def'
    >>> punctuation_spaces('abc.')
    'abc .'
    >>> punctuation_spaces('abc -def-ghi.')
    'abc - def - ghi .'
    """

    result: List[str] = []
    offset = 0
    tail = ''

    while offset < len(s):
        match = _PUNCTUATION_PATTERN.search(s, offset)
        if not match:
            break
        groups = match.groupdict()
        before, punctuation, after = (groups[key] for key in ('before', 'punctuation', 'after'))
        offset += len(before) + len(punctuation)
        result.extend((before, punctuation))
        tail = after

    if len(result):
        result.append(tail)
        return ' '.join(filter(len, map(str.strip, result)))

    return s


def remove_urls(string: str) -> str:
    return re.sub(r'^https?:\/\/.*[\r\n]*', '', string, flags=re.MULTILINE)


def filter_translit(string: str) -> bool:
    """
    >>> filter_translit('')
    True
    >>> filter_translit('абв')
    True
    >>> filter_translit('абв cde')
    True
    >>> filter_translit('ALLO YOBA')
    False
    """
    return bool(not len(string) or re.search(r'[\u0400-\u04FF]', string))


def delete_duplicates(dataset: Sequence[str]) -> Sequence[str]:
    hashdict: Dict[int, List[int]] = {}
    for i, line in enumerate(dataset):
        hashdict.setdefault(hash(line), []).append(i)

    return [
        dataset[indices[0]]
        for indices in hashdict.values()
    ]


if __name__ == '__main__':
    DATASET_PATH = '../../datasets/jokes.json'
    NEW_DATASET_PATH = '../../datasets/jokes_cleaned.json'

    dataset = read_dataset(DATASET_PATH)

    translit_filtered = filter(filter_translit, dataset)
    urls_deleted = map(remove_urls, translit_filtered)
    punctuation_cleaned = map(punctuation_spaces, urls_deleted)
    spaces_deduplicated = map(deduplicate_spaces, punctuation_cleaned)

    cleaned = delete_duplicates(tuple(spaces_deduplicated))

    save_dataset(NEW_DATASET_PATH, cleaned)
