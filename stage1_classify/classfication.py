import pandas as pd
import numpy as np
import pickle
import itertools
import functools
import collections
import random

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
from gensim.utils import simple_preprocess


params = {'vector_size': [400],
         'min_count': [1, 2],
         'epochs': [20, 50, 100],
         'window': [5, 10, 15],
         'steps': [20, 30, 40],
         'dm': [0, 1],
          'threshold': [1e-2, 1e-5],
          'negative': [5, 15]
         }


def process(df):
    df.dropna(subset=['body', 'headline', 'summary'], thresh=3, inplace=True)

    df['categories'] = df['categories'].apply(lambda x: '. '.join(x))

    df['train'] = [t + '. ' + h + '. ' + s + ' ' + b  for t, h, s, b in
                            zip(list(df['categories']), list(df['headline']), list(df['summary']), list(df['body']))]
    
    return df


def strat_test_train(X, y, test_size):
    strat = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    
    for train_index, test_index in strat.split(X, y):
        X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
        y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
    return X_train, y_train, X_test, y_test


def read_corpus(data):
    for i, line in enumerate(data):
        yield TaggedDocument(simple_preprocess(line), tags=[i])


def doc2vec(data, vector_size, dm, threshold, negative, min_count, epochs, window):
    model = Doc2Vec(vector_size=vector_size,
                    dm=dm,
                    min_count=min_count,
                    window=window,
                    threshold=threshold,
                    negative=negative,
                    epocphs=epochs,
                    workers=8)
    
    model.build_vocab(data)
    model.train(data, total_examples=model.corpus_count, epochs=model.epochs)
    
    return model

def embeddings(model, X, steps):
    z = [model.infer_vector(X[doc_id].words, steps=steps) for doc_id in range(len(X))]
    return z

def classifier(X_train, y_train):
    clf = svm.SVC()
    clf.fit(X_train, y_train)


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()

    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def flatten(x):
    if isinstance(x, collections.Iterable) and not isinstance(x, tuple) and not isinstance(x, str) and not isinstance(x, dict):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

def average(l):
    return functools.reduce(lambda x, y: x + y, l) / len(l)

def extract_pos(X_tr, y_tr):
    return np.array([v for v, l in zip(X_tr, y_tr) if l==1])


def unpack_kwargs(**kwargs):
    vector_size = kwargs.pop('vector_size')
    min_count = kwargs.pop('min_count')
    epochs = kwargs.pop('epochs')
    window = kwargs.pop('window')
    steps = kwargs.pop('steps')
    dm = kwargs.pop('dm')
    threshold = kwargs.pop('threshold')
    negative = kwargs.pop('negative')

    return vector_size, min_count, epochs, window, steps, dm, threshold, negative

def full_pipeline(scores, X_tr, y_tr, all_data, **kwargs):

    vector_size, min_count, epochs, window, steps, dm, threshold, negative = unpack_kwargs(**kwargs)
    
    print('Training doc2vec.. this will take some time')
    d2v = doc2vec(all_data, vector_size=vector_size, dm=dm, threshold=threshold, negative=negative, min_count=min_count, epochs=epochs, window=window)

    X_tr = embeddings(d2v, X_tr, steps=steps)

    skf = StratifiedKFold(n_splits=5, random_state=42)

    temp, i = [], 0

    print('Cross-validating SVM')

    for train_index, test_index in skf.split(X_tr, y_tr):

        print('Split %r...' % i)

        X_tr_cv, X_te_cv = [X_tr[i] for i in train_index], [X_tr[i] for i in test_index]
        
        y_tr_cv, y_te_cv = [y_tr[i] for i in train_index], [y_tr[i] for i in test_index]

        clf = svm.SVC()
        clf.fit(X_tr_cv, y_tr_cv)

        y_pr_cv = clf.predict(X_te_cv)

        c = confusion_matrix(y_te_cv, y_pr_cv)
        print(c)

        p = precision_score(y_te_cv, y_pr_cv)
        r = recall_score(y_te_cv, y_pr_cv)
        f1 = f1_score(y_te_cv, y_pr_cv)
        a = accuracy_score(y_te_cv, y_pr_cv)

        temp.append([p, r, f1, a])

        i+=1

    scores.append(temp)

    print('----------------------------------------------------')

    return scores
    


if __name__ == '__main__':
    
    df_eq = pd.read_pickle('../dataframes/df_eq_label.pkl')
    
    with open('../dataframes/all_data.pkl', 'rb') as f:
        all_data = pickle.load(f)

    df_eq = process(df_eq)

    X = list(df_eq['train'])
    y = list(df_eq['label'])

    X_train, y_train, X_test, y_test = strat_test_train(X, y, 0.2)

    X_train = list(read_corpus(X_train))

    results = {}
    scores = []

    for i, param in enumerate(list(product_dict(**params))):
        print('Checking set %r of parameters...' % i)

        scores = full_pipeline(scores, X_train, y_train, all_data, **param)

        results[i] = flatten( [param, list(zip(*scores[i]))] )

    data = [[key] + [val for val in vals] for key, vals in results.items()]

    pr = pd.DataFrame(data, columns=['Model #', 'Parameters', 'Precision', 'Recall', 'F1', 'Accuracy'])

    pr['Precision'] = pr['Precision'].apply(average)
    pr['Recall'] = pr['Recall'].apply(average)
    pr['F1'] = pr['F1'].apply(average)
    pr['Accuracy'] = pr['Accuracy'].apply(average)

    pd.to_pickle(pr, '../dataframes/grid_search_results.pkl')












