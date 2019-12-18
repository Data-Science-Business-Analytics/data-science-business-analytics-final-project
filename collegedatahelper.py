import pandas as pd
import numpy as np

years_train_all = [
    '1996_97',
    '1997_98',
    '1998_99',
    '1999_00',
    '2000_01',
    '2001_02',
    '2002_03',
    '2003_04',
    '2004_05',
    '2005_06',
    '2006_07',
    '2007_08',
    '2008_09',
    '2009_10',
    '2010_11',
    '2011_12',
    '2012_13',
    '2013_14',
    '2014_15'
]

years_train = [
    '2003_04',
    '2005_06',
    '2007_08',
    '2009_10',
    '2011_12',
    '2012_13',
    '2013_14',
    '2014_15'
]

years_test =[
    '1996_97',
    '1997_98',
    '1998_99',
    '1999_00',
    '2000_01',
    '2001_02',
    '2002_03',
    '2004_05',
    '2006_07',
    '2008_09',
    '2010_11',
    '2015_16',
    '2016_17',
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

train_whitelist = [
    'CONTROL',
    'debt_to_income',
    'UNITID', 
    'year'
]

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 


def format_df(df, isTrain, train_features=[]):
    df_copy = df.copy()

    # Replace Privacy Suppressed with NaN
    # Get rid of all nan for DEBT_MDN and MD_EARN_WNE_P6 (for train)
    if isTrain:
        df_copy = df_copy.replace('PrivacySuppressed',np.NaN).dropna(subset=['DEBT_MDN', 'MD_EARN_WNE_P6'])
        
        # Make everything numeric
        df_copy = df_copy.apply(pd.to_numeric, errors='ignore')
    
        # Set debt to income ratio column
        df_copy['debt_to_income'] = df_copy['DEBT_MDN'] / df_copy['MD_EARN_WNE_P6']


        train_features_true = intersection(df_copy.columns, train_features)
        df_copy = df_copy[train_features_true].fillna(0)

    else:
        df_copy = df_copy.replace('PrivacySuppressed',np.NaN)
        # Make everything numeric
        df_copy = df_copy.apply(pd.to_numeric, errors='ignore')


    pub, priv, priv_prof = clean_df_nan(df_copy, isTrain)

    return {
        'public': pub,
        'private': priv,
        'private_for_profit': priv_prof
    }


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

    # Add whitelist
    train_features.extend(train_whitelist)

    return train_features

def create_data_dict_full(datadir):
    data_dict = {}

    for year in years_train_all:
        df_copy = pd.read_csv(f'{datadir}/MERGED{year}_PP.csv', low_memory=False)
        df_copy = df_copy.replace('PrivacySuppressed',np.NaN).dropna(subset=['DEBT_MDN', 'MD_EARN_WNE_P6'])
        
        # Make everything numeric
        df_copy = df_copy.apply(pd.to_numeric, errors='ignore')
    
        # Set debt to income ratio column
        df_copy['debt_to_income'] = df_copy['DEBT_MDN'] / df_copy['MD_EARN_WNE_P6']

        data_dict[year] = df_copy

    for year in years_test:
        df_copy = pd.read_csv(f'{datadir}/MERGED{year}_PP.csv', low_memory=False)
        df_copy = df_copy.replace('PrivacySuppressed',np.NaN)
        
        # Make everything numeric
        df_copy = df_copy.apply(pd.to_numeric, errors='ignore')

        data_dict[year] = df_copy

    return data_dict

def create_data_dict(datadir, featuresdir, whitelist=[]):
    # Get Features
    features_df = get_features_dictionary(featuresdir)
    train_features = getTrainFeatures(features_df)

    # Add whitelist to train_features
    train_features.extend(whitelist)

    data_dict = {}

    for year in years_train:
        data_dict[year] = pd.read_csv(f'{datadir}/MERGED{year}_PP.csv', low_memory=False)
        # Format DF (train)
        data_dict[year] = format_df(data_dict[year], True, train_features)
        
    for year in years_test:
        data_dict[year] = pd.read_csv(f'{datadir}/MERGED{year}_PP.csv', low_memory=False)
        # Format DF (test)
        data_dict[year] = format_df(data_dict[year], False, train_features)
    
    return data_dict



def dropbadvalues(df):
    df_copy = df.copy()
    rows = df_copy.shape[0]
    df_copy = df_copy.dropna(axis=1, thresh=rows*0.7)
    
#     df_copy = df_copy.dropna(axis=0) 
    
    return df_copy


def clean_df_nan(df, isTrain):
    df_copy = df.copy()

    df_copy_pub = df_copy.loc[df_copy['CONTROL'] == 1]
    df_copy_priv = df_copy.loc[df_copy['CONTROL'] == 2]
    df_copy_priv_prof = df_copy.loc[df_copy['CONTROL'] == 3]
    
    if isTrain:
        df_copy_pub = dropbadvalues(df_copy_pub)
        df_copy_priv = dropbadvalues(df_copy_priv)
        df_copy_priv_prof = dropbadvalues(df_copy_priv_prof)
        
    return df_copy_pub, df_copy_priv, df_copy_priv_prof



# Data from Random Forest
important_features_public = [
    'SAT_AVG_ALL', 'RET_FT4', 'ADM_RATE_ALL', 'UGDS_BLACK', 'PPTUG_EF',
    'TUITIONFEE_IN', 'UGDS_ASIAN', 'NPT4_PUB', 'PCIP50', 'UGDS_WHITE',
    'COSTT4_A', 'UGDS_HISP', 'UG25ABV', 'UGDS_NHPI', 'TUITIONFEE_OUT',
    'UGDS_AIAN', 'NPT45_PUB', 'PCIP14', 'NPT4_3075_PUB', 'GRADS'
]

important_features_private = [
    'PCIP50', 'TUITIONFEE_IN', 'ADM_RATE_ALL', 'PCIP23', 'RET_FT4',
    'PPTUG_EF', 'GRADS', 'UGDS_BLACK', 'NPT4_3075_PRIV', 'NPT42_PRIV',
    'PFTFTUG1_EF', 'UG25ABV', 'UGDS_WHITE', 'PCIP52', 'UGDS_HISP',
    'SAT_AVG_ALL', 'UGDS', 'NPT41_PRIV', 'PCIP51', 'D_PCTPELL_PCTFLOAN'
]

important_features_private_profit = [
    'TUITIONFEE_IN', 'PCIP50', 'PPTUG_EF', 'CIPTFBS1', 'GRADS',
    'UGDS_WOMEN', 'PFTFTUG1_EF', 'UG25ABV', 'UGDS_WHITE', 'TUITIONFEE_OUT',
    'UGDS_HISP', 'UGDS_ASIAN', 'RET_FTL4', 'TUITIONFEE_PROG', 'UGDS_BLACK',
    'MTHCMP1', 'UGDS', 'NUM4_PRIV', 'NUM42_PRIV', 'NUM41_PRIV'
]