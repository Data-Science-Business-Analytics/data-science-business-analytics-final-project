import pandas as pd
import numpy as np

years = [
#     '1996_97',
#     '1997_98',
#     '1998_99',
#     '1999_00',
#     '2000_01',
#     '2001_02',
#     '2002_03',
#     '2003_04',
#     '2004_05',
#     '2005_06',
#     '2006_07',
#     '2007_08',
#     '2008_09',
#     '2009_10',
#     '2010_11',
#     '2011_12',
    '2012_13',
    '2013_14',
    '2014_15',
    '2015_16',
    '2016_17',
    '2017_18'
]

def clean_df(df):
    df_copy = df.copy()

    df_copy = df_copy.replace('PrivacySuppressed',np.NaN).dropna(subset=['DEBT_MDN'])
    
    rows = df_copy.shape[0]
    df_copy = df_copy.dropna(axis=1, thresh=rows*0.7)
    
    df_copy = df_copy.dropna(axis=0) 

    df_copy = df_copy.apply(pd.to_numeric, errors='ignore')

    return df_copy

    

def create_data_dict(datadir):
    data_dict = {}

    for year in years:
        data_dict[year] = pd.read_csv(f'{datadir}/MERGED{year}_PP.csv', low_memory=False)
    
    return data_dict
