import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem.snowball import RussianStemmer


def phrase_to_bag_of_words(phrase, stopwords=set(stopwords.words('russian'))):
    vectorizer = CountVectorizer()
    vectorizer.fit_transform([phrase])

    features = vectorizer.get_feature_names()

    return [word for word in features if not word in stopwords]
    
def jaccard_similarity(a, b):
    a_features = phrase_to_bag_of_words(a)
    b_features = phrase_to_bag_of_words(b)

    a_set = set(a_features)
    b_set = set(b_features)

    intersection = a_set.intersection(b_set)

    return len(intersection) / (len(a_set) + len(b_set) - len(intersection))


def gain_all_data():
    return pd.read_csv('./data/all.csv')


if __name__ == "__main__":
    data = gain_all_data().sample(frac=1)

    sample = []

    for phrase in data.text:
        if len(sample) == 600:
            break

        distances = [jaccard_similarity(phrase, text) < 0.04 for text in sample]

        if all(distances):
            sample.append(phrase)

            print(len(sample))
    
    series = pd.Series(data=sample)
    series.to_csv('./data/sample_600_96%.csv', index=False)
