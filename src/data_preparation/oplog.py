#!/usr/bin/env python3
# coding=utf-8
import pickle
from dataclasses import dataclass
from typing import Tuple, Callable, Set


@dataclass
class OplogEntry:
    index: int
    entry: str
    value: Tuple[int, ...]

    def __reduce__(self) -> Tuple[Callable, Tuple]:
        return OplogEntry, (self.index, self.entry, self.value)

    def serialize(self, file_path: str):
        with open(file_path, 'ab') as f:
            pickle.dump(self, f)

    def __hash__(self) -> int:
        return hash(self.entry)

    def __eq__(self, other) -> bool:
        if not isinstance(other, OplogEntry):
            return False
        return self.entry == other.entry


def read_oplog(oplog_path: str) -> Set[OplogEntry]:
    entries = set()

    with open(oplog_path, 'rb') as f:
        while True:
            try:
                entries.add(pickle.load(f))
            except EOFError:
                break

    return entries
