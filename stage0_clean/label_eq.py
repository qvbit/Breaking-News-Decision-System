import pandas as pd
import numpy as np
import pickle


def group_events(unique):
    i, j = 0, 0
    group = 0
    res = []
    while i < len(unique):
        while j < len(unique)-1 and unique[j+1] == '0':
            j += 1
            
        res += ((str(group)+ ',') * (j-i+1)).split(',')[:-1]

        group += 1
        j += 1
        i = j     
    return res


if __name__ == '__main__':

    df = pd.read_pickle('../dataframes/df_clean.pkl')

    df_eq = pd.read_csv('../data/quakes-ddupe.csv')

    df_eq['_id'] = df_eq['_id'].apply(lambda x: str(x))
    df['_id'] = df['_id'].apply(lambda x: str(x))

    df = pd.merge(df, df_eq[['_id', 'Earthquake (NaturalDisaster)', 'T0', 'T1', 'T2', 'Unique']], on='_id')
    df.rename(columns={'Earthquake (NaturalDisaster)': 'label'}, inplace=True)
    df['label'].fillna(0, inplace=True)

    df = df[df['label'] == 1]
    df.dropna(subset=['body', 'headline', 'summary'], thresh=3, inplace=True)

    df = df[['_id', 'body', 'headline', 'summary', 'categories', 'T0', 'T1', 'T2', 'Unique']]

    df['categories'] = df['categories'].apply(lambda x: '. '.join(x))

    unique = list(df['Unique'])

    df['group'] = group_events(unique)

    df = df.drop(columns=['Unique'])

    pd.to_pickle(df, '../dataframes/df_eq_label.pkl')