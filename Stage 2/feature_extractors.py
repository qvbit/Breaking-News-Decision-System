import pandas as pd
import numpy as np
import spacy as spacy
from spacy import displacy
from collections import Counter



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


def location_features(df, save=False):
    """
    Returns two arrays: the first contains the document number and the name of the named location (for identification purposes). The second contains the feature vectors that we pass to the model.
    """

    nlp = spacy.load('en_core_web_lg')
    docs = [nlp(doc) for doc in df['body']]


    def drop_noGPE(docs):
        drop = []
        for i, doc in enumerate(docs):
            if len([ent for ent in doc.ents if ent.label_ == 'GPE' or ent.label_=='LOC']) == 0:
                drop.append(i)

        for index in sorted(drop, reverse=True):
            del docs[index]

        return docs

    docs = drop_noGPE(docs)        

    features = []

    num_ents = len(ENT_MAP)
    num_pos = len(POS_MAP)
    
    for i, doc in enumerate(docs):

        names = [ent.text for ent in doc.ents if ent.label_ == 'GPE' or ent.label_=='LOC']
        loc_ents = [ent for ent in doc.ents if ent.label_=='GPE' or ent.label_=='LOC']
        num_examples = len(names)
        doc_num = [i for _ in range(num_examples)]

        # Feature 1
        f1 = np.array([ent.sent.vector for ent in doc.ents if ent.label_ == 'GPE' or ent.label_=='LOC'])

        # Feature 2
        f2 = np.zeros((num_examples, num_ents))

        for i, e1 in enumerate(loc_ents):
            for e2 in e1.sent.ents:
                f2[i, ENT_MAP[e2.label_]] = 1

        # Feature 3
        f3 = np.zeros((num_examples, num_pos))

        for i, e1 in enumerate(loc_ents):
            for e2 in e1.sent:
                f3[i, POS_MAP[e2.pos_]] = 1

        # Feature 4
        f4 = np.array([e.start for e in loc_ents]).reshape(-1, 1)

        # Feature 5
        loc_counts = Counter(names)
        f5 = np.array([loc_counts[e.text] for e in loc_ents]).reshape(-1, 1)

        feature = np.hstack((np.array(doc_num).reshape(-1, 1), np.array(names).reshape(-1, 1), f1, f2, f3, f4, f5))
        features.append(feature)
    
    num_features = feature.shape[1]
    ret = []
    
    for f in features:
        for r in f:
            ret.append(r)
            
    ret = np.array(ret).reshape(-1, num_features)
        
    return ret[:, :2], ret[:, 2:]    