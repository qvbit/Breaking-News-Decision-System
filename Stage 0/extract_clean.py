import pandas as pd
import numpy as np
import pymongo
from pymongo import MongoClient
import pprint
import collections
from html.parser import HTMLParser

def dic_only(series):
    """ 
    Function to unnest basic dict
    """
    list_series = list(series)
    
    for i, x in enumerate(list_series):
        if (type(x) != dict) or not x:
            list_series[i] = {'':''}
            
    return [list(dic.values())[0] for dic in list_series]

def extract_labels(series):
    """
    Function to extract the heavily nested labels
    """
    list_series = list(series)
    ret = [[] for _ in range(len(list_series))]
    
    for i, x in enumerate(list_series):
        if type(x) != list or not x:
            list_series[i] = [{}]
            
    for i, doc in enumerate(list_series):
        if 'about' in doc[0].keys():
            list_temp = []
            for dictionary in doc[0]['about']:
                if type(dictionary) == dict:
                    for key, value in dictionary.items():
                        if key == 'preferredLabel':
                            list_temp.append(value)
                else:
                    ret[i] = []
            ret[i] = list_temp   
        else:
            ret[i] = []         
    return ret

def flatten(l):
    """
    Flatten an irregular nested list
    """
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, str):
            for sub in flatten(el):
                yield sub
        else:
            yield el
            
def make_datetime(dates):
    """
    Standardizes time and also turns into datetime object
    """
    
    import operator
    from datetime import datetime
    from datetime import timedelta
    
    dates = list(dates)
    
    sep = 'T'
    
    just_dates = [date.split(sep, 1)[0] if type(date) == str else np.nan for date in dates]
    just_times = [date.split(sep, 1)[1] if type(date) == str else np.nan for date in dates]
    
    # Datetime object for just the dates:
    just_dates = [datetime.strptime(date, '%Y-%m-%d') if type(date) == str else np.nan for date in just_dates]
    
    # Convert times to datetime and also universalize offsets.
    ops = {'+' : operator.sub, '-' : operator.add}
    
    for i, time in enumerate(just_times):
        if type(time) == str:
            time, op, zone = time[:8], time[8], time[9:]
            time = datetime.strptime(time,"%H:%M:%S")
            res1 = ops[op](time, timedelta(hours = int(zone[:2])))
            just_times[i] = ops[op](res1, timedelta(minutes = int(zone[3:])))
    
    # Combine results into datetime object:
    
    return [datetime.combine(datetime.date(one), datetime.time(two)) if type(one) == datetime else np.nan
            for one, two in zip(just_dates, just_times)]

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ' '.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

if __name__ == '__main__':
    
    print('Connecting to db...')
    client = MongoClient('localhost', 27017)
    bbc_news = client['bbc_news']
    cps_withtags = bbc_news['cps_withtags']
    
    cps_withtags_proj = list(cps_withtags.find({}, {
                                                'summary': 1,
                                                'title': 1,
                                                'firstPublished': 1,
                                                'lastPublished': 1,
                                                'changeQueueId': 1,
                                                'changeQueueTimestamp': 1,
                                                'body': 1,
                                                'options.isBreakingNews': 1,
                                                'headline': 1,
                                                'shortHeadline': 1,
                                                'section.name': 1,
                                                'site.name': 1,
                                                'linkData.about': 1,
                                                'linkData.@type': 1,
                                                'linkData.createdBy.preferredLabel': 1
                                               }))
    
    df = pd.DataFrame(cps_withtags_proj)
    
    print('Extraction complete, now commencing data cleaning')
    
    df['site'] = dic_only(df['site'])
    df['options'] = dic_only(df['options'])
    df['section'] = dic_only(df['section'])
    labels = extract_labels(df['linkData'])
    labels = [list(flatten(l)) for l in labels]
    df['categories'] = labels
    df.drop(['linkData'], axis=1, inplace=True)
    df['firstPublished'] = make_datetime(df['firstPublished'])
    df['lastPublished'] = make_datetime(df['lastPublished'])
    df['changeQueueTimestamp'] = make_datetime(df['changeQueueTimestamp'])
    
    print('Stripping HTML tags, this may take a while')
    body = list(df['body'])
    body_stripped = [strip_tags(v) if type(v) == str else np.nan for v in body]
    df['body'] = body_stripped
    
    
    print('Cleaning done, saving dataframe to disk, now labelling earthquakes and their locations')
    pd.to_pickle(df, '../dataframes/df_clean.pkl')

    df_eq = pd.read_csv('../data/quakes - quakes.csv')

    df_eq['_id'] = df_eq['_id'].apply(lambda x: str(x))
    df['_id'] = df['_id'].apply(lambda x: str(x))

    res = pd.merge(df, df_eq[['_id', 'Earthquake (NaturalDisaster)', 'T0', 'T1', 'T2']], on='_id')
    res.rename(columns={'Earthquake (NaturalDisaster)': 'label'}, inplace=True)
    res['label'].fillna(0, inplace=True)

    pd.to_pickle(res, '../dataframes/df_eq_label.pkl')
    
    
    
    
    
    
    
    
    
    
    
    