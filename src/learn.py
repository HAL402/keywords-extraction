#!/usr/bin/env python3
# coding=utf-8
import math
import pickle
from functools import partial
from typing import Iterable, Sequence

import pandas as pd
import numpy as np
import pymorphy2 as pm
from src.maru.grammeme.pos import PartOfSpeech
from src.maru.morph import Morph
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
from wiki_ru_wordnet import WikiWordnet
from  itertools import combinations

Morphs = Sequence[Sequence[Morph]]
wikiwordnet = WikiWordnet()
morph = pm.MorphAnalyzer()

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
    PartOfSpeech.ADJECTIVE,
    PartOfSpeech.VERB,
    PartOfSpeech.ADVERB,
    PartOfSpeech.NOUN,
}


def get_normalized_words(text):
    words = []
    for word in text.split():
        res = morph.parse(word)
        if res[0].tag.POS in MEANINGFUL_POS:
            words.append(res[0].normal_form)
    return words


def get_words_synset(word):
    wordsSet = set()
    wikiwordnet.get_synsets(word)
    synsets = wikiwordnet.get_synsets(word)
    if len(synsets) != 0:
        for w in synsets[0].get_words():
            wordsSet.add(w.lemma())
        wordsSet.remove(word)
    return wordsSet


def get_syn_count(joke):
    words = set(get_normalized_words(joke))
    synsets = [(x, get_words_synset(x)) for x in words]
    words_with_syn = [w for w in synsets if len(w[1]) > 0]
    if len(words_with_syn) <= 1:
        return 0
    kek = [a for a in combinations(words_with_syn, 2) if a[0][0] != a[1][0] and len(a[0][1].intersection(a[1][1])) > 0]
    return len(kek)


def get_unknown_words(text):
    words = []
    for word in text.split():
        res = morph.parse(morph.normal_forms(word)[0])
        if res[0].tag.POS in MEANINGFUL_POS and len(wikiwordnet.get_synsets(morph.normal_forms(word)[0])) == 0:
            words.append(word)
    return len(words)


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
    df = pd.read_csv('../data/train2.csv', delimiter=';').dropna()
    df = df[df.category != 0]
    text_index = select_text_index(df)
    return train_test_split(text_index, df.category, test_size=0.3)


def get_text_length(x: pd.Series):
    return np.array([len(t) for t in x]).reshape(-1, 1)


def contains_question(x: pd.Series):
    return np.array([1 if "?" in t else 0 for t in x]).reshape(-1, 1)


def syn_count(x: pd.Series):
    return np.array([get_syn_count(t) for t in x]).reshape(-1, 1)


def unknown_words_count(x: pd.Series):
    return np.array([get_unknown_words(t) for t in x]).reshape(-1, 1)


def get_morphs(morphs: Morphs, x: pd.Series):
    return np.array([
        morphs[t] for t in x.index
    ]).reshape(-1, 1)


def get_lemmas(x: Morphs):
    return np.array([
        ' '.join(morph.lemma for morph in line[0])
        for line in x
    ])


def get_unique_count(x: Sequence[Morph]):
    lemmas = [m.lemma for m in x]
    return len(set(lemmas))/len(lemmas)


def unique_words(x: pd.Series):
    return np.array([
        get_unique_count(m) for line in x for m in line
    ]).reshape(-1, 1)


def pos_ratio(x: Sequence[Morph], pos: PartOfSpeech):
    pos_filtered = list(filter(lambda m: m.tag.pos == pos, x))
    return len(pos_filtered)/len(x)


def get_part_of_speech_ratio(pos: PartOfSpeech, x: pd.Series):
    return np.array([
        pos_ratio(m, pos) for line in x for m in line
    ]).reshape(-1, 1)


def meaningful_pos(x: Sequence[Morph]):
    pos_filtered = list(filter(lambda m: m.tag.pos in MEANINGFUL_POS, x))
    return len(pos_filtered)


def get_meaningful_pos(x: pd.Series):
    return np.array([
        meaningful_pos(m) for line in x for m in line
    ]).reshape(-1, 1)


def create_model(morphs: Morphs):
    forest = RandomForestClassifier(n_estimators=42, n_jobs=-1, random_state=17)
    forest_params = {
        'max_depth': range(1, 11),
        'max_features': range(1, 11)
    }
    grid = GridSearchCV(
        forest, forest_params,
        cv=3, n_jobs=-1,
        verbose=True
    )

    param_grid = {'C': list(i/20 for i in range(1, 21))}

    svc_grid = GridSearchCV(
        LinearSVC(), param_grid,
        cv=3, n_jobs=-1, verbose=True
    )

    tfidf_pipeline = Pipeline([
        ('column', FunctionTransformer(select_text, validate=False)),
        ('vect', CountVectorizer(ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer())
    ])

    lemma_tfidf_pipeline = Pipeline([
        ('morphs', FunctionTransformer(partial(get_morphs, morphs), validate=False)),
        ('lemmas', FunctionTransformer(get_lemmas, validate=False)),
        ('vect', CountVectorizer(ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer())
    ])

    length_pipeline = Pipeline([
        ('column', FunctionTransformer(select_text, validate=False)),
        ('length', FunctionTransformer(get_text_length, validate=False))
    ])

    count_vec_lemma_pipeline = Pipeline([
        ('morphs', FunctionTransformer(partial(get_morphs, morphs), validate=False)),
        ('lemmas', FunctionTransformer(get_lemmas, validate=False)),
        ('vectorizer', CountVectorizer(ngram_range=(1, 2)))
    ])

    count_vec_pipeline = Pipeline([
        ('column', FunctionTransformer(select_text, validate=False)),
        ('vectorizer', CountVectorizer(ngram_range=(1, 2)))
    ])

    unique_words_pipeline = Pipeline([
        ('morphs', FunctionTransformer(partial(get_morphs, morphs), validate=False)),
        ('counts', FunctionTransformer(unique_words, validate=False))
    ])

    verb_pipeline = Pipeline([
        ('morphs', FunctionTransformer(partial(get_morphs, morphs), validate=False)),
        ('pos_ratio', FunctionTransformer(partial(get_part_of_speech_ratio, PartOfSpeech.VERB), validate=False))
    ])

    noun_pipeline = Pipeline([
        ('morphs', FunctionTransformer(partial(get_morphs, morphs), validate=False)),
        ('pos_ratio', FunctionTransformer(partial(get_part_of_speech_ratio, PartOfSpeech.NOUN), validate=False))
    ])

    adj_pipeline = Pipeline([
        ('morphs', FunctionTransformer(partial(get_morphs, morphs), validate=False)),
        ('pos_ratio', FunctionTransformer(partial(get_part_of_speech_ratio, PartOfSpeech.ADJECTIVE), validate=False))
    ])

    pos_pipeline = Pipeline([
        ('morphs', FunctionTransformer(partial(get_morphs, morphs), validate=False)),
        ('pos', FunctionTransformer(get_meaningful_pos, validate=False))
    ])

    quest_pipeline = Pipeline([
        ('column', FunctionTransformer(select_text, validate=False)),
        ('quest', FunctionTransformer(contains_question, validate=False))
    ])

    syn_pipeline = Pipeline([
        ('column', FunctionTransformer(select_text, validate=False)),
        ('quest', FunctionTransformer(syn_count, validate=False))
    ])

    unknow_words_pipeline = Pipeline([
        ('column', FunctionTransformer(select_text, validate=False)),
        ('unkn', FunctionTransformer(unknown_words_count, validate=False))
    ])

    return Pipeline([
        ('features', FeatureUnion([
            ('tfidf', tfidf_pipeline),
            ('lemma_tfidf', lemma_tfidf_pipeline),
            ('count_vec_lemma', count_vec_lemma_pipeline),
            ('count_vec', count_vec_pipeline),
            ('length', length_pipeline),
            ('unique_words', unique_words_pipeline),
            ('adj', adj_pipeline),
            ('noun', noun_pipeline),
            ('verb', verb_pipeline),
            ('meaningful_pos', pos_pipeline),
            ('quest', quest_pipeline),
            ('syn', syn_pipeline),
            ('unkn', unknow_words_pipeline)
        ])),
        ('svc_grid', svc_grid)
    ]), svc_grid


# region validation

def validate(model, data_test, answer_test):
    test_df = pd.concat([data_test, answer_test], axis=1)

    # sex_df = test_df[test_df.category == 0]
    non_df = test_df[test_df.category == 1]
    incres_df = test_df[test_df.category == 2]

    # predicted_sex = model.predict(select_text_index(sex_df))
    predicted_non = model.predict(select_text_index(non_df))
    predicted_incres = model.predict(select_text_index(incres_df))

    # sex_accuracy = np.mean(predicted_sex == sex_df.category)
    non_accuracy = np.mean(predicted_non == non_df.category)
    incres_accuracy = np.mean(predicted_incres == incres_df.category)

    overall_accuracy = math.sqrt(non_accuracy * incres_accuracy)
    print(pd.DataFrame({
        # 'Sex': [sex_accuracy],
        'Non': [non_accuracy],
        'Inc-Res': [incres_accuracy],
        'Overall': [overall_accuracy]
    }))

# endregion


if __name__ == '__main__':
    for x in range(0, 10):
        lemmas = list(read_dump('../data/lemmas_dump2'))
        data_train, data_test, answer_train, answer_test = gain_train_data()

        model, grid = create_model(lemmas)
        model.fit(data_train, answer_train)
        # print(grid.best_params_)

        validate(model, data_test, answer_test)
