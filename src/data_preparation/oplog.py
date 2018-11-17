#!/usr/bin/env python3
# coding=utf-8
import pickle
import json
from dataclasses import dataclass
from typing import Tuple, Callable, Set, List, Dict

from categories import Categories


@dataclass
class OplogEntry:
    index: int
    entry: str
    value: Tuple[str, ...]

    def __reduce__(self) -> Tuple[Callable, Tuple]:
        return OplogEntry, (self.index, self.entry, self.value)

    def serialize(self, file_path: str):
        with open(file_path, 'ab') as f:
            pickle.dump(self, f)

    def to_dict(self) -> Dict:
        return {
            'index': self.index,
            'values': [Categories[v.upper()].value for v in self.value]
        }

    def __hash__(self) -> int:
        return self.index

    def __eq__(self, other) -> bool:
        if not isinstance(other, OplogEntry):
            return False
        return self.index == other.index


def read_oplog(oplog_path: str) -> List[OplogEntry]:
    entries = list()

    with open(oplog_path, 'rb') as f:
        while True:
            try:
                entries.append(pickle.load(f))
            except EOFError:
                break

    return entries


def get_samples_set(entries: List[OplogEntry]) -> Set[OplogEntry]:
    return set(reversed(entries))


def save_training_meta(entries: Set[OplogEntry], file_name: str = '../../training_meta/meta.txt') -> None:
    dicts = list(map(OplogEntry.to_dict, entries))
    with open(file_name, 'a') as f:
        json.dump({'entries': dicts}, f)
