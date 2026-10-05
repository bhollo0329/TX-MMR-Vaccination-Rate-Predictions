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



def pivot_enrollment_dataset(df_enrollment, enrollment_type, drop_columns, index_columns):
    '''  
    Pivot school enrollment datasets; move enrollment value under column reflecting description
        df_enrollment (DataFrame): original dataframe obtained from TEA
        enrollment_type (str): type of enrollment 
    '''
    df_clean = df_enrollment.drop(columns=drop_columns).dropna()

    rename_dict_enrollment = {'COUNTY NAME': 'County_Name', 'DISTRICT NAME': 'District_Name'}
    df_pivot = df_clean.pivot(index=index_columns, columns=enrollment_type, values='ENROLLMENT').reset_index().rename(columns=rename_dict_enrollment).fillna(0)

    return df_clean, df_pivot

    
def convert_enrollment_percentage(df_absolute, df_full_counts):
    ''' 
    Replace unknonw values (<#) by half a value in order to calculate percentages of enrollment
        df_pivot (): 
        df_full_counts ():
    '''

    unknown_dict = {'<10': '5', '<20': '18', '<30': '15', '<40': '20','<50': '25', '<60': '30', '<70': '35', '<80': '40', '<90': '45',
                        '<100': '50', '<110': '55', '<120': '60', '<130': '65', '<140': '70', '<150': '75', '<160': '80', '<170': '85',
                        '<180': '90', '<190': '95', '<200': '100', '<240': '120', '<270': '135', '<320': '160', '<340': '170', '<350': '175',
                        '<370': '185', '<400': '200', '<460': '230', '<470': '235', '<510': '255', '<570': '285', '<710': '355', '<730': '365',
                        '<830': '415', '<1,010': '505', '<1,260': '630'}
    
    df_half = df_absolute.replace(unknown_dict)
    for c in  df_half.drop(columns=['County_Name', 'District_Name']).columns:
        df_half[c] = df_half[c].astype(int)


    df_full_counts['County_Name'] = df_full_counts['County_Name'].str.strip()
    df_full_counts['District_Name'] = df_full_counts['District_Name'].str.strip()

    df_percent = pd.merge(df_half, df_full_counts[['County_Name', 'District_Name', 'Total_Count']])

    for c in df_percent.drop(columns=['County_Name', 'District_Name', 'Total_Count']).columns:
        df_percent[c] = round(df_percent[c]/df_percent['Total_Count'], 3)

    return df_percent


def replace_unknown_random_percent(df_absolute, df_full_counts):
    ''' 
    Replace unknonw values (<#) by a randome geneartor within that range in order to calculate percentages of enrollment
        df_absolute (DataFrame): 
        df_full_counts (DataFrame):
    '''
    unknown_values = list(range(10, 2000, 10))
    rand_gen_dict = {}
    for i in unknown_values:
        rand_gen = random.randint(1, i-1)
        rand_gen_dict[f'{i}'] = rand_gen

    unknown_rand_dict = {'<10': rand_gen_dict['10'], '<20': rand_gen_dict['20'], '<30': rand_gen_dict['30'], '<40': rand_gen_dict['40'],
                        '<50': rand_gen_dict['50'], '<60': rand_gen_dict['60'], '<70': rand_gen_dict['70'], '<80': rand_gen_dict['80'], 
                        '<90': rand_gen_dict['90'], '<100': rand_gen_dict['100'], '<110': rand_gen_dict['110'], '<120': rand_gen_dict['120'], 
                        '<130': rand_gen_dict['130'], '<140': rand_gen_dict['140'], '<150': rand_gen_dict['150'], '<160': rand_gen_dict['160'], '<170': rand_gen_dict['170'],
                        '<180': rand_gen_dict['180'], '<190': rand_gen_dict['190'], '<200': rand_gen_dict['200'], '<210': rand_gen_dict['210'], '<220': rand_gen_dict['220'], 
                        '<230': rand_gen_dict['230'], '<240': rand_gen_dict['240'], '<250': rand_gen_dict['250'], '<260': rand_gen_dict['260'], '<270': rand_gen_dict['270'],
                        '<280': rand_gen_dict['280'], '<290': rand_gen_dict['290'], '<300': rand_gen_dict['300'], '<310': rand_gen_dict['310'], '<320': rand_gen_dict['320'],
                        '<330': rand_gen_dict['330'], '<340': rand_gen_dict['340'], '<350': rand_gen_dict['350'], '<360': rand_gen_dict['360'], '<370': rand_gen_dict['370'],
                        '<380': rand_gen_dict['380'], '<390': rand_gen_dict['390'], '<400': rand_gen_dict['400'], '<410': rand_gen_dict['410'], '<420': rand_gen_dict['420'], 
                        '<430': rand_gen_dict['430'], '<440': rand_gen_dict['440'], '<450': rand_gen_dict['450'], '<460': rand_gen_dict['460'], '<470': rand_gen_dict['470'],
                        '<480': rand_gen_dict['480'], '<490': rand_gen_dict['490'], '<500': rand_gen_dict['500'], '<510': rand_gen_dict['510'], '<520': rand_gen_dict['520'], 
                        '<530': rand_gen_dict['530'], '<540': rand_gen_dict['540'], '<550': rand_gen_dict['550'], '<560': rand_gen_dict['560'], '<570': rand_gen_dict['570'],
                        '<580': rand_gen_dict['580'], '<390': rand_gen_dict['590'], '<600': rand_gen_dict['600'], '<610': rand_gen_dict['610'], '<620': rand_gen_dict['620'], 
                        '<630': rand_gen_dict['630'], '<640': rand_gen_dict['640'], '<650': rand_gen_dict['650'], '<660': rand_gen_dict['660'], '<670': rand_gen_dict['670'],
                        '<680': rand_gen_dict['680'], '<690': rand_gen_dict['690'], '<710': rand_gen_dict['710'], '<720': rand_gen_dict['720'], '<730': rand_gen_dict['730'],
                        '<740': rand_gen_dict['740'], '<760': rand_gen_dict['760'], '<770': rand_gen_dict['770'], '<780': rand_gen_dict['780'], '<790': rand_gen_dict['790'],
                        '<800': rand_gen_dict['800'], '<820': rand_gen_dict['820'], '<830': rand_gen_dict['830'], '<870': rand_gen_dict['870'], '<880': rand_gen_dict['880'],
                        '<890': rand_gen_dict['890'], '<930': rand_gen_dict['930'], '<1,010': rand_gen_dict['1010'], '<1,110': rand_gen_dict['1110'], '<1,120': rand_gen_dict['1120'],
                        '<1,130': rand_gen_dict['1130'], '<1,180': rand_gen_dict['1180'], '<1,260': rand_gen_dict['1260'], '<1,300': rand_gen_dict['1300'], '<1,310': rand_gen_dict['1310'], '<1,380': rand_gen_dict['1380'],
                        '<1,690': rand_gen_dict['1690'], '<1,780': rand_gen_dict['1780'], '<1,960': rand_gen_dict['1960'], '<1,970': rand_gen_dict['1970'], '<1,980': rand_gen_dict['1980'],
                        }
    
    df_rand_fill = df_absolute.replace(unknown_rand_dict)
    for c in  df_rand_fill.drop(columns=['County_Name', 'District_Name']).columns:
        df_rand_fill[c] = df_rand_fill[c].astype(int)


    df_full_counts['County_Name'] = df_full_counts['County_Name'].str.strip()
    df_full_counts['District_Name'] = df_full_counts['District_Name'].str.strip()

    df_percent = pd.merge(df_rand_fill, df_full_counts[['County_Name', 'District_Name', 'Total_Count']])

    for c in df_percent.drop(columns=['County_Name', 'District_Name', 'Total_Count']).columns:
        df_percent[c] = round(df_percent[c]/df_percent['Total_Count'], 3)

    return df_percent







