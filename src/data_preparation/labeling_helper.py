#!/usr/bin/env python3
# coding=utf-8
"""
    Usage: in jupyter run
        `import sys`
        `sys.path.append(r'{path}\src\data_preparation')`
        `from labeling_helper import setup`
        `setup('{path}/jokes_cleaned.json', START_OF_RANGE, END_OF_RANGE)`
"""
from typing import List, Callable, Set, Optional

from ipyannotate import annotate
from ipyannotate.annotation import Annotation
from ipyannotate.tasks import Task, MultiTask
from ipyannotate.buttons import (
    ValueButton, NextButton, BackButton
)

from src.common.categories import Categories
from src.common.utilities import read_dataset
from oplog import OplogEntry


def check_category_correctness(selected_categories: Set[Categories]) -> Optional[str]:
    if (Categories.NONSENSE.name.capitalize() in selected_categories and
            Categories.INCONGRUITY.name.capitalize() in selected_categories):
        return '!!! NON cannot be combined with INC-RES'
    return None


class SavingButton(ValueButton):
    def __init__(self, oplog_path: str, index_offset: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._oplog_path = oplog_path
        self._index_offset = index_offset

    def handle_click(self) -> None:
        current_task: MultiTask = self.annotation.tasks.current
        previous_value = set(current_task.value)

        index = self.annotation.tasks.index
        super().handle_click()

        incorrectness = check_category_correctness(current_task.value)
        if incorrectness is not None:
            print(incorrectness)
            current_task.value = previous_value
            return

        self._write_oplog(index, current_task)

    def _write_oplog(self, index: int, task: Task) -> None:
        entry = OplogEntry(
            index + self._index_offset,
            task.output,
            tuple(task.value)
        )
        entry.serialize(self._oplog_path)


def get_batch(dataset: List[str], start: int, end: int) -> List[str]:
    return dataset[start:end]


def get_annotaion(data: List[str], button_factory: Callable) -> Annotation:
    colors = 'red', 'blue', 'green', 'gray'
    buttons = [
        button_factory(category.name.capitalize(), color=colors[(i - 1) % 4], shortcut=str(i))
        for i, category in enumerate(Categories, start=1)
    ]
    controls = [
        BackButton(shortcut='j'),
        NextButton(shortcut='k')
    ]
    return annotate(data, buttons=(buttons + controls), multi=True)


def setup(dataset_file: str, start: int = -1, end: int = -1, oplog_path: str = 'oplog.txt') -> Annotation:
    if start < 0 or end < 0:
        raise ValueError('Specify correct data range')

    dataset = read_dataset(dataset_file)
    batch = get_batch(dataset, start, end)
    button_factory = lambda *args, **kwargs: SavingButton(oplog_path, start, *args, **kwargs)
    return get_annotaion(batch, button_factory)
