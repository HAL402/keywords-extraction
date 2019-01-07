#!/usr/bin/env python3
# coding=utf-8
import csv
import json
import pickle
import random
from functools import partial
from itertools import combinations
from typing import Iterable, Sequence, List

import numpy as np
import pandas as pd
import pymorphy2 as pm
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from wiki_ru_wordnet import WikiWordnet

from src.maru.grammeme import Gender, Animacy
from src.maru.grammeme.pos import PartOfSpeech
from src.maru.morph import Morph
import gensim.downloader as api

Morphs = Sequence[Sequence[Morph]]
wikiwordnet = WikiWordnet()
morph = pm.MorphAnalyzer()
word_vectors = api.load("word2vec-ruscorpora-300")

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


def get_words_synset(word):
    wordsSet = set()
    wikiwordnet.get_synsets(word)
    synsets = wikiwordnet.get_synsets(word)
    if len(synsets) != 0:
        for w in synsets[0].get_words():
            wordsSet.add(w.lemma())
        wordsSet.remove(word)
    return wordsSet


def get_syn_count(x: Sequence[Morph]):
    words = set([m.lemma for m in x if m.tag.pos in MEANINGFUL_POS])
    synsets = [(x, get_words_synset(x)) for x in words]
    words_with_syn = [w for w in synsets if len(w[1]) > 0]
    if len(words_with_syn) <= 1:
        return 0
    kek = [a for a in combinations(words_with_syn, 2) if a[0][0] != a[1][0] and len(a[0][1].intersection(a[1][1])) > 0]
    return len(kek)


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


def gain_train_data(source_path, no_test=False):
    df = pd.read_csv(source_path, delimiter=';').dropna()
    text_index = select_text_index(df)

    if no_test:
        return select_text_index(df), (), df.category, ()
    return train_test_split(text_index, df.category, test_size=0.3, shuffle=True, random_state=16)


def get_text_length(x: pd.Series):
    return np.array([len(t) for t in x]).reshape(-1, 1)


def contains_question(x: pd.Series):
    return np.array([1 if t[-1] == "?" else 0 for t in x]).reshape(-1, 1)


def contains_question1(x: pd.Series):
    return np.array([1 if t[-1] != "?" and "?" in t else 0 for t in x]).reshape(-1, 1)


def syn_count(x: pd.Series):
    return np.array([
        get_syn_count(m) for line in x for m in line
    ]).reshape(-1, 1)


def get_sex_keywords_count(x: Sequence[Morph]):
    words = [m.lemma for m in x if m.tag.pos in MEANINGFUL_POS]
    sex_keywords = ["секс", "член", "трах", "жена", "муж", "парень", "любовь", "оргазм",
                    "девушка", "трах", "постель", "измена", "женщина", "мужчина", "порно", "минет", "анал", "дроч"]
    res = len([word for word in words if word in sex_keywords or any([sex_word in word for sex_word in sex_keywords])])

    return res


def sex_topic_words_count(x: pd.Series):
    return np.array([
        get_sex_keywords_count(m) for line in x for m in line
    ]).reshape(-1, 1)


def get_mean_w2v(x: Sequence[Morph]):
    words = [f"{m.lemma}_{m.tag.pos}" for m in x if f"{m.lemma}_{m.tag.pos}" in word_vectors]
    return np.mean([word_vectors[word] for word in words], axis=0)


def mean_w2v(x: pd.Series):
    return np.array([
        get_mean_w2v(m) for line in x for m in line
    ]).reshape(-1, 1)


def get_morphs(morphs: Morphs, x: pd.Series):
    return np.array([
        morphs[t] for t in x["index"]
    ]).reshape(-1, 1)


def get_lemmas(x: Morphs):
    return np.array([
        ' '.join(morph.lemma for morph in line[0])
        for line in x
    ])


def get_unique_count(x: Sequence[Morph]):
    lemmas = [m.lemma for m in x]
    return len(set(lemmas)) / len(lemmas)


def unique_words(x: pd.Series):
    return np.array([
        get_unique_count(m) for line in x for m in line
    ]).reshape(-1, 1)


def get_unknown_words_count(x: Sequence[Morph]):
    return len([m.lemma for m in x if m.tag.pos in MEANINGFUL_POS and f"{m.lemma}_{m.tag.pos}" not in word_vectors])


def unknown_words_count(x: pd.Series):
    return np.array([
        get_unknown_words_count(m) for line in x for m in line
    ]).reshape(-1, 1)


def get_gender_words_count(x: Sequence[Morph]):
    return len([m for m in x if m.tag.gender == Gender.FEMININE and m.tag.animacy == Animacy.ANIMATE])


def gender_words_count(x: pd.Series):
    return np.array([
        get_gender_words_count(m) for line in x for m in line
    ]).reshape(-1, 1)


def pos_ratio(x: Sequence[Morph], pos: PartOfSpeech):
    pos_filtered = list(filter(lambda m: m.tag.pos == pos, x))
    return len(pos_filtered) / len(x)


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
    russian_stopwords = stopwords.words("russian")
    lemma_tfidf_pipeline = Pipeline([
        ('morphs', FunctionTransformer(partial(get_morphs, morphs), validate=False)),
        ('lemmas', FunctionTransformer(get_lemmas, validate=False)),
        ('vect', TfidfVectorizer(stop_words=russian_stopwords, ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer())
    ])

    count_vec_lemma_pipeline = Pipeline([
        ('morphs', FunctionTransformer(partial(get_morphs, morphs), validate=False)),
        ('lemmas', FunctionTransformer(get_lemmas, validate=False)),
        ('vectorizer', TfidfVectorizer(stop_words=russian_stopwords, ngram_range=(1, 2)))
    ])

    count_vec_pipeline = Pipeline([
        ('column', FunctionTransformer(select_text, validate=False)),
        ('vectorizer', TfidfVectorizer(stop_words=russian_stopwords, ngram_range=(1, 2)))
    ])

    unique_words_pipeline = Pipeline([
        ('morphs', FunctionTransformer(partial(get_morphs, morphs), validate=False)),
        ('counts', FunctionTransformer(unique_words, validate=False))
    ])

    unknown_words_pipeline = Pipeline([
        ('morphs', FunctionTransformer(partial(get_morphs, morphs), validate=False)),
        ('counts', FunctionTransformer(unknown_words_count, validate=False))
    ])

    gender_pipeline = Pipeline([
        ('morphs', FunctionTransformer(partial(get_morphs, morphs), validate=False)),
        ('lol_counts', FunctionTransformer(gender_words_count, validate=False))
    ])

    pos_pipeline = Pipeline([
        ('morphs', FunctionTransformer(partial(get_morphs, morphs), validate=False)),
        ('pos', FunctionTransformer(get_meaningful_pos, validate=False))
    ])

    quest_pipeline = Pipeline([
        ('column', FunctionTransformer(select_text, validate=False)),
        ('quest', FunctionTransformer(contains_question, validate=False))
    ])
    quest_pipeline1 = Pipeline([
        ('column', FunctionTransformer(select_text, validate=False)),
        ('quest1', FunctionTransformer(contains_question1, validate=False))
    ])
    syn_pipeline = Pipeline([
        ('morphs', FunctionTransformer(partial(get_morphs, morphs), validate=False)),
        ('syn', FunctionTransformer(syn_count, validate=False))
    ])

    sex_topic_words_count_pipeline = Pipeline([
        ('morphs', FunctionTransformer(partial(get_morphs, morphs), validate=False)),
        ('quest', FunctionTransformer(sex_topic_words_count, validate=False))
    ])

    mean_w2v_pipeline = Pipeline([
        ('morphs', FunctionTransformer(partial(get_morphs, morphs), validate=False)),
        ('counts', FunctionTransformer(mean_w2v, validate=False))
    ])

    return Pipeline([
        ('features', FeatureUnion([
            ('lemma_tfidf', lemma_tfidf_pipeline),
            ('count_vec_lemma', count_vec_lemma_pipeline),
            ('count_vec', count_vec_pipeline),
            ('unique_words', unique_words_pipeline),
            ('meaningful_pos', pos_pipeline),
            ('quest', quest_pipeline),
            ("quest1", quest_pipeline1),
            ('syn', syn_pipeline),
            ('unkn', unknown_words_pipeline),
            ('gen', gender_pipeline),
            ('sex_keywords', sex_topic_words_count_pipeline),
            #('mean_w2v', mean_w2v_pipeline),
        ])),
        ('lr', LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=400))
    ])


def print_report(prediction, answer):
    print(classification_report(answer, prediction, target_names=["sex", "non", "inc-res"]))

    matrix = confusion_matrix(answer, prediction)
    total_accuracy = accuracy_score(answer, prediction)
    accuracy_report = np.append(matrix.diagonal() / matrix.sum(axis=1), total_accuracy)
    print(pd.DataFrame(accuracy_report, index=["sex", "non", "inc-res", "total"], columns=["Accuracy"]))


def read_json_dataset(file_path: str) -> List[str]:
    with open(file_path, encoding='utf-8') as f:
        return json.load(f)


def get_predict_data() -> pd.DataFrame:
    dataset = read_json_dataset('../data/jokes_cleaned.json')
    random.seed(42)
    random.shuffle(dataset)
    return pd.DataFrame(list(enumerate(dataset)), columns=['index', 'text'])


def active_learning(model, data_train, save_path1, save_path2):
    NUM_TOP = 250
    NUM_BOTTOM = 150

    learn_data_indices = set(data_train['index'])
    predict_data = get_predict_data()

    predicted = model.predict_proba(predict_data)

    top = list(sorted(enumerate(predicted), key=lambda x: min_distance(x[1])))

    np.set_printoptions(suppress=True)
    text = predict_data['text']

    with open(save_path1, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        c = 0
        p = 0
        while c < NUM_TOP:
            i, score = top[p]
            p += 1
            if i in learn_data_indices:
                continue
            c += 1

            writer.writerow([i, text[i], score])

    TARGET_CLASS = 2
    with open(save_path2, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        c = 0
        p = 1
        while c < NUM_BOTTOM:
            i, score = top[len(top) - p]
            p += 1
            if max(enumerate(score), key=lambda x: x[1])[0] != TARGET_CLASS:
                continue
            if 'тирлиц' in text[i]:
                continue
            if i in learn_data_indices:
                continue
            c += 1

            writer.writerow([i, text[i], TARGET_CLASS])


def min_distance(distances) -> float:
    distances = list(enumerate(distances))
    first = max(distances, key=lambda x: x[1])
    others = [(i, d) for i, d in distances if i != first[0]]
    return min(
        (first[1] - others[0][1]) ** 2,
        (first[1] - others[1][1]) ** 2
    )


def test(model, data_test, answer_test):
    predicted = model.predict(data_test)
    print_report(predicted, answer_test)


if __name__ == '__main__':
    TEST = False

    lemmas = dict(read_dump(f'../data/lemmas_dump4'))

    data_train, data_test, answer_train, answer_test = gain_train_data(
        f'../data/train6.csv', no_test=not TEST)
    full_table = data_train.join(pd.DataFrame(answer_train))
    model = create_model(lemmas)
    model.fit(data_train, answer_train)

    if TEST:
        test(model, data_test, answer_test)
    else:
        active_learning(model, data_train, '../data/top3.csv', '../data/bottom3.csv')
