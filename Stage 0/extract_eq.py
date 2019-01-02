import pandas as pd
import numpy as np


if __name__ == '__main__':
    df = pd.read_pickle('/Users/naumanw1/Documents/SoT/dataframes/df_clean.pkl')
    df_eq = pd.read_csv('/Users/naumanw1/Documents/SoT/csvs/quakes - quakes.csv')
    
    df_eq['_id'] = df_eq['_id'].apply(lambda x: str(x))
    df['_id'] = df['_id'].apply(lambda x: str(x))
    
    res = pd.merge(df, df_eq[['_id', 'Earthquake (NaturalDisaster)']], on='_id', how='left')
    
    res.rename(columns={'Earthquake (NaturalDisaster)': 'label'}, inplace=True)
    res['label'].fillna(0, inplace=True)
    
    pd.to_pickle(res, '/Users/naumanw1/Documents/SoT/dataframes/df_eq_label.pkl')
   
    
    
   