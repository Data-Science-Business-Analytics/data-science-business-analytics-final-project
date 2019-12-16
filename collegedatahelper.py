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
    # '2014_15',
    # '2015_16',
    # '2016_17',
    '2017_18'
]

train_feature_categories = [
    'admissions',
    'academics',
    'student',
    'cost'
]

train_blacklist = [
    'CIPTITLE1',
    'CIPTITLE2',
    'CIPTITLE3',
    'CIPTITLE4',
    'CIPTITLE5',
    'CIPTITLE6',
    'INC_PCT_LO',
    'DEP_STAT_PCT_IND',
    'IND_INC_PCT_LO',
    'DEP_INC_PCT_LO',
    'PAR_ED_PCT_1STGEN',
    'INC_PCT_M1',
    'INC_PCT_M2',
    'INC_PCT_H1',
    'INC_PCT_H2',
    'DEP_INC_PCT_M1',
    'DEP_INC_PCT_M2',
    'DEP_INC_PCT_H1',
    'DEP_INC_PCT_H2',
    'IND_INC_PCT_M1',
    'IND_INC_PCT_M2',
    'IND_INC_PCT_H1',
    'IND_INC_PCT_H2',
    'PAR_ED_PCT_MS',
    'PAR_ED_PCT_HS',
    'PAR_ED_PCT_PS',
    'APPL_SCH_PCT_GE2',
    'APPL_SCH_PCT_GE3',
    'APPL_SCH_PCT_GE4',
    'APPL_SCH_PCT_GE5',
    'DEP_INC_AVG',
    'IND_INC_AVG',
    'INC_N',
    'DEP_INC_N',
    'IND_INC_N',
    'DEP_STAT_N',
    'PAR_ED_N',
    'APPL_SCH_N',
    'PELL_EVER',
    'AGE_ENTRY',
    'FEMALE',
    'MARRIED',
    'DEPENDENT',
    'VETERAN',
    'FIRST_GEN',
    'FAMINC',
    'MD_FAMINC',
    'FAMINC_IND'
]

def format_df(df, isTrain):
    df_copy = df.copy()

    # Replace all Privacy Supressed with NaN
    df_copy = df_copy.replace('PrivacySuppressed',np.NaN).dropna(subset=['DEBT_MDN'])

    # Make everything numeric
    df_copy = df_copy.apply(pd.to_numeric, errors='ignore')

    # If isTrain - delete all rows that don't contain DEBT_MDN or MD_EARN_WNE_P6
    if isTrain:
        df_copy = df_copy.dropna(subset=['DEBT_MDN', 'MD_EARN_WNE_P6'])
        df_copy['debt_to_income'] = df_copy['DEBT_MDN'] / df_copy['MD_EARN_WNE_P6']

    return df_copy

def clean_df(df):
    df_copy = df.copy()

    df_copy = df_copy.replace('PrivacySuppressed',np.NaN).dropna(subset=['DEBT_MDN'])
    
    rows = df_copy.shape[0]
    df_copy = df_copy.dropna(axis=1, thresh=rows*0.7)
    
    df_copy = df_copy.dropna(axis=0) 

    df_copy = df_copy.apply(pd.to_numeric, errors='ignore')

    return df_copy


def clean_train_df(df, train_features):
    df_copy = df.copy()

    # Replace all instances of "PrivacySuppressed" with NaN
    df_copy = df_copy.replace('PrivacySuppressed',np.NaN)

    # Remove all unused features
    df_copy = df_copy[train_features]
    
    # rows = df_copy.shape[0]
    # df_copy = df_copy.dropna(axis=1, thresh=rows*0.7)
    
    # df_copy = df_copy.dropna(axis=0) 

    # df_copy = df_copy.apply(pd.to_numeric, errors='ignore')

    return df_copy


def get_features_dictionary(features_dir):
    features_df = pd.read_csv(f'{features_dir}/dictionary.csv', low_memory=False)[['VARIABLE NAME', 'dev-category', 'API data type', 'NAME OF DATA ELEMENT']]
    return features_df    

def getTrainFeatures(features_df):
    train_features = []

    for train_feature_category in train_feature_categories:
        features = features_df.loc[features_df['dev-category'] == train_feature_category]['VARIABLE NAME'].unique()
        train_features = np.concatenate((train_features, features), axis=None)
        
    # Remove Blacklist
    train_features = [train_features for train_features in train_features if train_features not in train_blacklist]
    # train_features = np.concatenate((train_features, ), axis=None)

    return train_features

def create_data_dict(datadir):
    data_dict = {}

    for year in years:
        data_dict[year] = pd.read_csv(f'{datadir}/MERGED{year}_PP.csv', low_memory=False)
    
    return data_dict
