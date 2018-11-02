#!/usr/bin/env python3
# coding=utf-8
import json
from typing import List, Sequence


def read_dataset(file_path: str) -> List[str]:
    with open(file_path, encoding='utf-8') as f:
        return json.load(f)


def save_dataset(file_path: str, dataset: Sequence[str], allow_rewrite=False) -> None:
    mode = allow_rewrite and 'w' or 'a'
    with open(file_path, mode, encoding='utf-8') as f:
        json.dump(dataset, f)
