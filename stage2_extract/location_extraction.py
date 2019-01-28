import pandas as pd
import numpy as np
import spacy as spacy
from spacy import displacy
from collections import Counter
import collections
import pickle

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")


POS_MAP = {'ADJ' : 0,
          'ADP' : 1,
          'ADV': 2,
          'AUX': 3,
          'CONJ': 4,
          'CCONJ': 5,
          'DET': 6,
          'INTJ': 7,
          'NOUN': 8,
          'NUM': 9,
          'PART': 10,
          'PRON': 11,
          'PROPN': 12,
          'PUNCT': 13,
          'SCONJ': 14,
          'SYM': 15,
          'VERB': 16,
          'X': 17,
          'SPACE': 18}

ENT_MAP = {'PERSON': 0,
          'NORP': 1,
          'FAC': 2,
          'ORG': 3,
          'GPE': 4,
          'LOC': 5,
          'PRODUCT': 6,
          'EVENT': 7,
          'WORK_OF_ART': 8,
          'LAW': 9,
          'LANGUAGE': 10,
          'DATE': 11,
          'TIME': 12,
          'PERCENT': 13,
          'MONEY': 14,
          'QUANTITY': 15,
          'ORDINAL': 16,
          'CARDINAL': 17}


def location_features(docs):
    """
    Returns _id, X, y as three nparrays.
    """

    features = []

    num_ents = len(ENT_MAP)
    num_pos = len(POS_MAP)
    
    for _id, doc, t in docs:

        names = [ent.text for ent in doc.ents if ent.label_ == 'GPE' or ent.label_=='LOC']
        loc_ents = [ent for ent in doc.ents if ent.label_=='GPE' or ent.label_=='LOC']
        num_examples = len(names)
        doc_num = [_id for _ in range(num_examples)]
        target = [t for _ in range(num_examples)]

        # Feature 1: Sentence vector in which the ent appears in.
        f1 = np.array([ent.sent.vector for ent in doc.ents if ent.label_ == 'GPE' or ent.label_=='LOC'])

        # Feature 2: Check surrounidng entity types
        f2 = np.zeros((num_examples, num_ents))

        for i, e1 in enumerate(loc_ents):
            for e2 in e1.sent.ents:
                f2[i, ENT_MAP[e2.label_]] = 1

        # Feature 3: Check surrounding part of speech tags
        f3 = np.zeros((num_examples, num_pos))

        for i, e1 in enumerate(loc_ents):
            for e2 in e1.sent:
                f3[i, POS_MAP[e2.pos_]] = 1

        # Feature 4: Token offset for each ent.
        f4 = np.array([e.start for e in loc_ents]).reshape(-1, 1)

        # Feature 5: How many times that particular ent appears in the entire document.
        loc_counts = Counter(names)
        f5 = np.array([loc_counts[e.text] for e in loc_ents]).reshape(-1, 1)
        
        # Feature 6: The word vectors themselves.
        f6 = ([ent.vector for ent in doc.ents if ent.label_ =='GPE' or ent.label_=='LOC'])

        feature = np.hstack((np.array(doc_num).reshape(-1, 1), np.array(names).reshape(-1, 1), f1, f2, f3, f4, f5, f6, np.array(target).reshape(-1, 1)))
        features.append(feature)
    
    num_features = feature.shape[1]
    ret = []
    
    for f in features:
        for r in f:
            ret.append(r)
            
    ret = np.array(ret).reshape(-1, num_features)

    _id = ret[:, 0]
    name = ret[:, 1]
    X = ret[:, 2:-1]
    y = ret[:, -1] 
    
    y = list(map(lambda x, y: int(x.lower() in y.lower()), y, name))  

    return _id, names, X, y


def eq_only(df):
    df = df[df['label'] == 1]
    df.dropna(subset=['body', 'headline', 'summary'], thresh=3, inplace=True)
    
    return df.reset_index()


if __name__ == '__main__':
    df = pd.read_pickle('../dataframes/df_eq_label.pkl')
    df = eq_only(df)

    df = df[['_id', 'body', 'headline', 'summary', 'categories', 'T0', 'T1', 'T2']]
    df['categories'] = df['categories'].apply(lambda x: '. '.join(x))

    combined = [t + '. ' + h + '. ' + s + ' ' + b  for t, h, s, b in
                        zip(list(df['categories']), list(df['headline']), list(df['summary']), list(df['body']))]

    target_locs = df[['T0', 'T1', 'T2']].fillna(method='bfill', axis='columns')['T0']

    print('Passing data to SpaCy')

    nlp = spacy.load('en_core_web_lg')
    docs = [(_id, nlp(doc), t) for doc, _id, t in list(zip(combined, list(df['_id']), target_locs))]

    print('Extracting features')
    _ids, names, X, y = location_features(docs)

    print('Training model')
    sc = StandardScaler()
    pca = PCA(n_components=500)
    svm = SVC()

    pipe = Pipeline( [('sc', sc), ('pca', pca), ('svm', svm)] )
    pipe.fit(X, y)

    print('Saving model to disk...')

    with open('../models/location_extraction_clf.pkl', 'wb') as f:
        pickle.dump(pipe, f)

    print('Done.')






