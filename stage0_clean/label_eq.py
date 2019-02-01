import pandas as pd
import numpy as np
import pickle



df = pd.read_pickle('../dataframes/df_clean.pkl')

df_eq = pd.read_csv('../data/quakes - quakes.csv')

df_eq['_id'] = df_eq['_id'].apply(lambda x: str(x))
df['_id'] = df['_id'].apply(lambda x: str(x))

res = pd.merge(df, df_eq[['_id', 'Earthquake (NaturalDisaster)', 'T0', 'T1', 'T2']], on='_id')
res.rename(columns={'Earthquake (NaturalDisaster)': 'label'}, inplace=True)
res['label'].fillna(0, inplace=True)

pd.to_pickle(res, '../dataframes/df_eq_label.pkl')