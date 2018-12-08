#!/usr/bin/env python3
# coding=utf-8
import math
import pickle
from functools import partial
from typing import Iterable, Sequence

import pandas as pd
import numpy as np
from maru.grammeme.pos import PartOfSpeech
from maru.morph import Morph
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

Morphs = Sequence[Sequence[Morph]]


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


MEANINGFUL_POS = {

}


def read_dump(file_path: str) -> Iterable[Sequence['Morph']]:
    with open(file_path, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def select_text_index(x: pd.Series):
    return x[['index', 'text']]


def select_text(x: pd.Series):
    return x.text


def gain_train_data():
    df = pd.read_csv('../data/train.csv', delimiter=';').dropna()
    return train_test_split(select_text_index(df), df.category, test_size=0.3)


def get_text_length(x: pd.Series):
    return np.array([len(t) for t in x]).reshape(-1, 1)


def get_morphs(morphs: Morphs, x: pd.Series):
    return np.array([
        morphs[t] for t in x.index
    ]).reshape(-1, 1)


def get_lemmas(x: Morphs):
    return np.array([
        ' '.join(morph.lemma for morph in line[0])
        for line in x
    ])


def create_model(morphs: Morphs):
    # forest = RandomForestClassifier(n_estimators=42, n_jobs=-1, random_state=17)
    # forest_params = {
    #     'max_depth': range(1, 10),
    #     'max_features': range(1, 10)
    # }
    # grid = GridSearchCV(
    #     forest, forest_params,
    #     cv=3, n_jobs=-1,
    #     verbose=True
    # )

    param_grid = {'C': [0.001, 0.01, 0.1, 0.2, 0.3, 0.35, 0.38, 0.39, 0.4, 0.41, 0.42, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}

    svc_grid = GridSearchCV(
        LinearSVC(), param_grid,
        cv=3, n_jobs=-1, verbose=True
    )

    tfidf_pipeline = Pipeline([
        ('column', FunctionTransformer(select_text, validate=False)),
        ('vect', CountVectorizer(ngram_range=(1, 5))),
        ('tfidf', TfidfTransformer())
    ])

    lemma_tfidf_pipeline = Pipeline([
        ('morphs', FunctionTransformer(partial(get_morphs, morphs), validate=False)),
        ('lemmas', FunctionTransformer(get_lemmas, validate=False)),
        ('vect', CountVectorizer(ngram_range=(1, 5))),
        ('tfidf', TfidfTransformer())
    ])

    length_pipeline = Pipeline([
        ('column', FunctionTransformer(select_text, validate=False)),
        ('length', FunctionTransformer(get_text_length, validate=False))
    ])

    count_vec_lemma_pipeline = Pipeline([
        ('morphs', FunctionTransformer(partial(get_morphs, morphs), validate=False)),
        ('lemmas', FunctionTransformer(get_lemmas, validate=False)),
        ('vectorizer', CountVectorizer(ngram_range=(1, 3)))
    ])

    count_vec_pipeline = Pipeline([
        ('column', FunctionTransformer(select_text, validate=False)),
        ('vectorizer', CountVectorizer(ngram_range=(1, 3)))
    ])

    return Pipeline([
        ('features', FeatureUnion([
            ('tfidf', tfidf_pipeline),
            ('lemma_tfidf', lemma_tfidf_pipeline),
            ('count_vec_lemma', count_vec_lemma_pipeline),
            ('count_vec', count_vec_pipeline),
            ('length', length_pipeline)
        ])),
        ('svc_grid', svc_grid)
    ]), svc_grid


# region validation

def validate(model, data_test, answer_test):
    test_df = pd.concat([data_test, answer_test], axis=1)

    sex_df = test_df[test_df.category == 0]
    non_df = test_df[test_df.category == 1]
    incres_df = test_df[test_df.category == 2]

    predicted_sex = model.predict(select_text_index(sex_df))
    predicted_non = model.predict(select_text_index(non_df))
    predicted_incres = model.predict(select_text_index(incres_df))

    sex_accuracy = np.mean(predicted_sex == sex_df.category)
    non_accuracy = np.mean(predicted_non == non_df.category)
    incres_accuracy = np.mean(predicted_incres == incres_df.category)

    overall_accuracy = math.sqrt(sex_accuracy * non_accuracy * incres_accuracy)
    print(pd.DataFrame({
        'Sex': [sex_accuracy],
        'Non': [non_accuracy],
        'Inc-Res': [incres_accuracy],
        'Overall': [overall_accuracy]
    }))

# endregion


if __name__ == '__main__':
    lemmas = list(read_dump('../data/lemmas_dump'))
    data_train, data_test, answer_train, answer_test = gain_train_data()

    model, grid = create_model(lemmas)
    model.fit(data_train, answer_train)
    print(grid.best_params_)

    validate(model, data_test, answer_test)
