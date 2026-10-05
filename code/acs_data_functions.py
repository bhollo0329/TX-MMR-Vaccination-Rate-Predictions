import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import statsmodels.api as sm
import random

from sklearn.linear_model import Ridge, RidgeCV, Lasso, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import resample
from sklearn.decomposition import PCA



def create_predictor_datasets(data_df, var_df, county_list):
    '''
    Create datasets for predictor variables.
        data_df (DataFrame): variable dataset downloaded from ACS5
        var_df (DataFrame): data frame containing a mapping of variable codes to their descriptions
        county_list (list): a list of counties in the vaccination dataset to filter the predictor dataset
    '''
    ### replace ACS codes (i.e. B01001_###) with their string descriptions
    labels = (var_df['label']+'_'+var_df['concept']).unique()
    varID = var_df['name'].unique()
    varMap = dict(zip(varID, labels))
    colDict = {'NAME': 'County', 'estimate': 'Population'}
    data_with_var = data_df.replace(varMap).drop(columns=['Unnamed: 0', 'moe', 'GEOID']).rename(columns=colDict)

    ### 
    cleanMap = {':!!': '_', '!!': '_', ' to ': '_', ' years': '', ' County, Texas': '', ':': '', ' ': '_'}
    data_with_var = data_with_var.replace(cleanMap, regex=True)

    ### filter data frame for only counties with vaccination rate data
    data_with_var = data_with_var[data_with_var['County'].isin(county_list)]

    ### remove gender based variables
    data_with_var_no_gender = data_with_var[~data_with_var['variable'].str.contains('Male|Female')]

    ### pivot df; retain county row assignment and make column names=variables, value=population
    ### drop highest level of column in order to merge later
    data_with_var_pivot_all = data_with_var.pivot(index=['County'], columns=['variable'], values=['Population']).droplevel(level=0, axis=1)
    data_with_var_pivot_no_gender = data_with_var_no_gender.pivot(index=['County'], columns=['variable'], values=['Population']).droplevel(level=0, axis=1)

    return data_with_var_pivot_all, data_with_var_pivot_no_gender




def create_predictor_percentages(predictor_df_all, predictor_df_no_gender):
        ''' 
        Convert predictor data frames from "Estimate Totals" to percentages
            predictor_df_all (DataFrame): 
            predictor_df_no_gender (DataFrame): 
        '''

        predictor_df_all_percents = predictor_df_all.copy()
        predictor_df_no_gender_percents = predictor_df_no_gender.copy()

        ### Converting "no gender" data set to percentages via divinding by Estimate Total
        for ca in predictor_df_no_gender_percents.drop(columns=['Estimate_Total']).columns:
            predictor_df_no_gender_percents[ca] = predictor_df_no_gender_percents[ca]/predictor_df_no_gender_percents['Estimate_Total']

        ### Convert perecentages based on whether the predictor is for Male, Female or Total populations
        if 'Estimate_Total_Male' in predictor_df_all_percents.columns:
            for cg in predictor_df_all_percents.drop(columns=['Estimate_Total', 'Estimate_Total_Male', 'Estimate_Total_Female']).columns:
                if 'Male_' in cg:
                    predictor_df_all_percents[cg] = predictor_df_all_percents[cg]/predictor_df_all_percents['Estimate_Total_Male']
                elif 'Female_' in cg:
                    predictor_df_all_percents[cg] = predictor_df_all_percents[cg]/predictor_df_all_percents['Estimate_Total_Female']
                else:
                    predictor_df_all_percents[cg] = predictor_df_all_percents[cg]/predictor_df_all_percents['Estimate_Total']     
        else:
             for c in predictor_df_all_percents.drop(columns=['Estimate_Total']).columns:
                predictor_df_all_percents[c] = predictor_df_all_percents[c]/predictor_df_all_percents['Estimate_Total']
    
        predictor_df_all_percents.columns = predictor_df_all_percents.columns.str.replace({'Estimate_Total_': 'Percent_'}, regex=True)
        predictor_df_no_gender_percents.columns = predictor_df_no_gender_percents.columns.str.replace({'Estimate_Total_': 'Percent_'}, regex=True)

        return predictor_df_all_percents, predictor_df_no_gender_percents



def combine_all_acs_variables(edu_df, race_df, enroll_df, age_df, poverty_df, percent=False, gender=False):
    ''' 
    '''

    if percent == True & gender==True:
        edu_race_df = pd.merge(edu_df.drop(columns=['Estimate_Total', 'B15003_025', 'Percent_Male', 'Percent_Female']), race_df.drop(columns=['Estimate_Total']), on=['County'])
        enroll_age_df = pd.merge(enroll_df.drop(columns=['Estimate_Total']), age_df.drop(columns=['Estimate_Total', 'Percent_Male', 'Percent_Female']), on=['County'])
        all_acs_variables = pd.merge(pd.merge(edu_race_df, enroll_age_df, on=['County']), poverty_df, on=['County'])
    else:
        edu_race_df = pd.merge(edu_df.drop(columns=['Estimate_Total', 'B15003_025']), race_df.drop(columns='Estimate_Total'), on=['County'])
        enroll_age_df = pd.merge(enroll_df.drop(columns='Estimate_Total'), age_df.drop(columns='Estimate_Total'), on=['County'])
        all_acs_variables = pd.merge(pd.merge(edu_race_df, enroll_age_df, on=['County']), poverty_df, on=['County'])

    return all_acs_variables


def combine_acs_variables(df_list):
    ''' 
    Combine ACS variable data by County
        df_list (arr(DataFrames)): list of DataFrames with ACS data indexed by county name
    '''
    if len(df_list) == 2:
        all_acs_variables = pd.merge(df_list[0], df_list[1], on=['County'])
    elif len(df_list) == 3:
        all_acs_variables = pd.merge(pd.merge(df_list[0], df_list[1], on=['County']), df_list[2], on=['County'])
    elif len(df_list) == 4:
        all_acs_variables = pd.merge(pd.merge(df_list[0], df_list[1], on=['County']), pd.merge(df_list[2], df_list[3], on=['County']), on=['County'])
    elif len(df_list) == 5:
        all_acs_variables_i = pd.merge(pd.merge(df_list[0], df_list[1], on=['County']), pd.merge(df_list[2], df_list[3], on=['County']), on=['County'])
        all_acs_variables = pd.merge(all_acs_variables_i, df_list[4], on=['County'])
    elif len(df_list) == 6:
        all_acs_variables_i = pd.merge(pd.merge(df_list[0], df_list[1], on=['County']), pd.merge(df_list[2], df_list[3], on=['County']), on=['County'])
        all_acs_variables = pd.merge(all_acs_variables_i, pd.merge(df_list[4], df_list[5], on=['County']), on=['County'])
    elif len(df_list) == 7:
        all_acs_variables_i = pd.merge(pd.merge(df_list[0], df_list[1], on=['County']), pd.merge(df_list[2], df_list[3], on=['County']), on=['County'])
        all_acs_variables = pd.merge(pd.merge(all_acs_variables_i, df_list[4], on=['County']), pd.merge(df_list[5], df_list[6], on=['County']), on=['County'])
    elif len(df_list) == 8:
            all_acs_variables_i = pd.merge(pd.merge(df_list[0], df_list[1], on=['County']), pd.merge(df_list[2], df_list[3], on=['County']), on=['County'])
            all_acs_variables_ii = pd.merge(pd.merge(all_acs_variables_i, df_list[4], on=['County']), pd.merge(df_list[5], df_list[6], on=['County']), on=['County'])
            all_acs_variables = pd.merge(all_acs_variables_ii, df_list[7], on=['County'])
    elif len(df_list) == 9:
            all_acs_variables_i = pd.merge(pd.merge(df_list[0], df_list[1], on=['County']), pd.merge(df_list[2], df_list[3], on=['County']), on=['County'])
            all_acs_variables_ii = pd.merge(pd.merge(all_acs_variables_i, df_list[4], on=['County']), pd.merge(df_list[5], df_list[6], on=['County']), on=['County'])
            all_acs_variables = pd.merge(pd.merge(all_acs_variables_ii, df_list[7], on=['County']), df_list[8], on=['County'])
    elif len(df_list) == 10:
            all_acs_variables_i = pd.merge(pd.merge(df_list[0], df_list[1], on=['County']), pd.merge(df_list[2], df_list[3], on=['County']), on=['County'])
            all_acs_variables_ii = pd.merge(pd.merge(all_acs_variables_i, df_list[4], on=['County']), pd.merge(df_list[5], df_list[6], on=['County']), on=['County'])
            all_acs_variables = pd.merge(pd.merge(all_acs_variables_ii, df_list[7], on=['County']), pd.merge(df_list[8], df_list[9], on=['County']), on=['County'])
    elif len(df_list) == 11:
            all_acs_variables_i = pd.merge(pd.merge(df_list[0], df_list[1], on=['County']), pd.merge(df_list[2], df_list[3], on=['County']), on=['County'])
            all_acs_variables_ii = pd.merge(pd.merge(all_acs_variables_i, df_list[4], on=['County']), pd.merge(df_list[5], df_list[6], on=['County']), on=['County'])
            all_acs_variables_iii = pd.merge(pd.merge(all_acs_variables_ii, df_list[7], on=['County']), pd.merge(df_list[8], df_list[9], on=['County']), on=['County'])
            all_acs_variables = pd.merge(all_acs_variables_ii, df_list[10], on=['County'])

    return all_acs_variables


def seperate_level_of_school_enrollment(absolute_df):
    '''
    Seperate absolute dataframe into the three school levels (i.e. high school, middle school and elemntary) if data is split into single grades
    '''
    high_school_df = pd.DataFrame()
    middle_school_df = pd.DataFrame()
    elementary_school_df = pd.DataFrame()
    for c in absolute_df.columns:
        if ("grade_10" in c)|("grade_11" in c)|("grade_9"in c)|("grade_12" in c):
            high_school_df[c] = absolute_df[c]
        elif ("grade_8" in c)|("grade_7" in c)|("grade_6"in c):
            middle_school_df[c] = absolute_df[c]
        elif ("grade_5" in c)|("grade_4" in c)|("grade_3" in c)|("grade_2"in c)|("grade_1_" in c)|("kindergarten" in c):
            elementary_school_df[c] = absolute_df[c]
    
    return high_school_df, middle_school_df, elementary_school_df


def calculate_race_percents_by_school_level_enrollment(school_level_df, percent_df, school_level):
    ''' 
    Calculate the racial percentage at each school level (i.e. high school, middle school and elementary school)
        school_level_df (DataFrame): DataFrame containing total and racial estimate at the school level
        percent_df (DataFrame): DataFrame where the percentages are stored
        school_level (str): which school level data is being used ("high", "middle", "elementary")
    '''

    total_level = pd.Series([0]*len(school_level_df))
    total_asian_alone = pd.Series([0]*len(school_level_df))
    total_black_alone = pd.Series([0]*len(school_level_df))
    total_islander_alone = pd.Series([0]*len(school_level_df))
    total_multi_alone = pd.Series([0]*len(school_level_df))
    total_other_race_alone = pd.Series([0]*len(school_level_df))
    total_white_alone = pd.Series([0]*len(school_level_df))
    total_hispanic_alone = pd.Series([0]*len(school_level_df))
    total_white_nonHispanic_alone = pd.Series([0]*len(school_level_df))

    for col in school_level_df.columns:
        if "(" not in col:
            total_level.index = school_level_df[col].index
            total_level += school_level_df[col]
        elif "Asian" in col:
            total_asian_alone.index = school_level_df[col].index
            total_asian_alone += school_level_df[col]
        elif "Black" in col:
            total_black_alone.index = school_level_df[col].index
            total_black_alone += school_level_df[col]
        elif "Native" in col:
            total_islander_alone.index = school_level_df[col].index
            total_islander_alone += school_level_df[col]
        elif "Two" in col:
            total_multi_alone.index = school_level_df[col].index
            total_multi_alone += school_level_df[col]
        elif "Other_Race" in col:
            total_other_race_alone.index = school_level_df[col].index
            total_other_race_alone += school_level_df[col]
        elif "White_Alone)" in col:
            total_white_alone.index = school_level_df[col].index
            total_white_alone += school_level_df[col]
        elif "(Hispanic" in col:
            total_hispanic_alone.index = school_level_df[col].index
            total_hispanic_alone += school_level_df[col]
        elif "Not_Hispanic" in col:
            total_white_nonHispanic_alone.index = school_level_df[col].index
            total_white_nonHispanic_alone += school_level_df[col]

    if school_level == "high":
        percent_df['Percent_High_School_Asian_Alone'] = total_asian_alone/total_level
        percent_df['Percent_High_School_Black_or_African_American_Alone'] = total_black_alone/total_level
        percent_df['Percent_High_School_Native_Hawaiian_and_Other_Pacific_Islander_Alone'] = total_islander_alone/total_level
        percent_df['Percent_High_School_Two_or_More_Races'] = total_multi_alone/total_level
        percent_df['Percent_High_School_Some_Other_Race_Alone'] = total_other_race_alone/total_level
        percent_df['Percent_High_School_White_Alone'] = total_white_alone/total_level
        percent_df['Percent_High_School_Hispanic_or_Latino_Alone'] = total_hispanic_alone/total_level
        percent_df['Percent_High_School_White_Alone_Not_Hispanic_or_Latino'] = total_white_nonHispanic_alone/total_level
    
    if school_level == "middle":
        percent_df['Percent_Middle_School_Asian_Alone'] = total_asian_alone/total_level
        percent_df['Percent_Middle_School_Black_or_African_American_Alone'] = total_black_alone/total_level
        percent_df['Percent_Middle_School_Native_Hawaiian_and_Other_Pacific_Islander_Alone'] = total_islander_alone/total_level
        percent_df['Percent_Middle_School_Two_or_More_Races'] = total_multi_alone/total_level
        percent_df['Percent_Middle_School_Some_Other_Race_Alone'] = total_other_race_alone/total_level
        percent_df['Percent_Middle_School_White_Alone'] = total_white_alone/total_level
        percent_df['Percent_Middle_School_Hispanic_or_Latino_Alone'] = total_hispanic_alone/total_level
        percent_df['Percent_Middle_School_White_Alone_Not_Hispanic_or_Latino'] = total_white_nonHispanic_alone/total_level
    
    if school_level == "elementary":
        percent_df['Percent_Elementary_School_Asian_Alone'] = total_asian_alone/total_level
        percent_df['Percent_Elementary_Black_or_African_American_Alone'] = total_black_alone/total_level
        percent_df['Percent_Elementary_Native_Hawaiian_and_Other_Pacific_Islander_Alone'] = total_islander_alone/total_level
        percent_df['Percent_Elementary_Two_or_More_Races'] = total_multi_alone/total_level
        percent_df['Percent_Elementaryl_Some_Other_Race_Alone'] = total_other_race_alone/total_level
        percent_df['Percent_Elementary_White_Alone'] = total_white_alone/total_level
        percent_df['Percent_Elementary_Hispanic_or_Latino_Alone'] = total_hispanic_alone/total_level
        percent_df['Percent_Elementary_White_Alone_Not_Hispanic_or_Latino'] = total_white_nonHispanic_alone/total_level