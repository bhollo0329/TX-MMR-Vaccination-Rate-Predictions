import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import statsmodels.api as sm
import random
import xgboost as xgb
import scipy.stats as stats

from sklearn.linear_model import Ridge, RidgeCV, Lasso, LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_recall_curve, roc_curve, auc
from sklearn.utils import resample
from sklearn.decomposition import PCA




def vac_predictor_merge(vac_df, predictor_df):
    ''' 
    Merge predictor variable data set with vaccination data set.
        vac_df (DataFrame): data set containing vaccination rates
        predictor_df_all (DataFrame): data set containing all predictor variable estimates
        predictor_df_no_gender (DataFrame): data set containing all non-gender based predictor variable estimates
    '''
    vac_df_rel = vac_df.drop(columns=['Facility Number', 'Facility Address', 'Age Group', 'School Type', 'School District or Name'])

    ### merge predictor date frame with goal data frame
    single_pred_df = pd.merge(predictor_df, vac_df_rel, on=['County'])
    
    return single_pred_df

def prepare_train_test_data(predictor_data_set, drop_columns, logit=False):
    ''' 
    Prepare data for training and testing
    '''
    scaler = StandardScaler()
    drop_columns.append('MMR Vaccination Rate')
    x_data = predictor_data_set.drop(columns=drop_columns)
    x_scaler = scaler.fit_transform(x_data)
    if logit == True:
        y_data = predictor_data_set['MMR Vaccination Rate']/100
        x_scaler = sm.add_constant(scaler.fit_transform(x_data))
    else:
        y_data = predictor_data_set['MMR Vaccination Rate']
        x_scaler = scaler.fit_transform(x_data)

    
    x_scalar_train, x_scalar_test, y_train, y_test = train_test_split(x_scaler, y_data, test_size=0.2, random_state=231)

    data_dict = {}
    data_dict['y_data_all'] = y_data
    data_dict['x_data_all'] = x_data
    data_dict['x_data_scaled_all'] = x_scaler

    data_dict['y_train'] = y_train
    data_dict['y_test'] = y_test
    data_dict['x_scalar_train'] = x_scalar_train
    data_dict['x_scalar_test'] = x_scalar_test


    return data_dict



def linear_regression_fit(predictor_data_set, predictor_name=None, sort_and_save=False):
    '''
    Perform linear regression fit and obtain results.
         
    '''
    scaler = MinMaxScaler()

    y = predictor_data_set['MMR Vaccination Rate']
    if 'Estimate_Total' in predictor_data_set.columns:
        x = predictor_data_set.drop(columns=['MMR Vaccination Rate', 'County', 'Estimate_Total'])
        x_scaler = scaler.fit_transform(x)
    else:
        x = predictor_data_set.drop(columns=['MMR Vaccination Rate', 'County'])
        x_scaler = scaler.fit_transform(x)

    model = LinearRegression()
    model.fit(x_scaler, y)

    coeffs = model.coef_
    intercept = model.intercept_

    predictor_names = x.columns

    results_df = pd.DataFrame({'Predictors': predictor_names, 'Coeff': coeffs})

    results_df['Correlation'] = ['Negative' if c < 0 else 'Positive' for c in results_df['Coeff']]

    x_with_constant = sm.add_constant(x)
    model = sm.OLS(y, x_with_constant).fit()

    ### 95% confidence intervals for all coeff
    conf_intervals = model.conf_int(alpha=0.05).drop('const').reset_index()
    conf_intervals.rename(columns={'index': 'Predictors', 0: 'CI_Low', 1: 'CI_High'}, inplace=True)

    ### Add confidence intercals to results dataframe
    results_df = results_df.merge(conf_intervals)
    results_df['Coeff Signficance'] = ['Not Signficant' if l <= 0 <= u else 'Signficant' for l,u in zip(results_df['CI_Low'], results_df['CI_High'])]

    ### summary of fit with p-values for coeff
    summary = model.summary()

    if sort_and_save == True:
        results_df = results_df.sort_values(by=['Coeff'], ascending=False)
        results_df.to_csv(f"../linear_regression_fit_results/{predictor_name}_linear_regression_fit_county_results.csv")
    

    return intercept, results_df, summary


def determine_corr_feats(feature_df, init_drop_columns):
    ''' 
    Deterine highly correlatead features and remove them
        feature_df (DataFrame): data frame with all features
        init_drop_columns (arr(str)): array of columns to drop before correlation matrix calculation
    '''
    num_feats_dict = {}
    num_feats_dict['init'] = len(feature_df.drop(columns=init_drop_columns).columns)

    corr_matrix = feature_df.drop(columns=init_drop_columns).corr().abs()

    ### mask that isolates top right half of correlation matrix in order to not evaulate collinear feature pairs twice
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    thresh = 0.85
    correlated_drop_cols = [col for col in upper_tri.columns if any(upper_tri[col] > thresh)]
    num_feats_dict['high_corr'] = len(correlated_drop_cols)

    num_feats_dict['final'] = len(feature_df.drop(columns=init_drop_columns).drop(columns=correlated_drop_cols).columns)

    reduced_corr_df = feature_df.drop(columns=correlated_drop_cols)

    print(f"{num_feats_dict['init']} intital features, reduced to {num_feats_dict['final']} features")

    return num_feats_dict, reduced_corr_df, correlated_drop_cols


def bootstrap_confidence_interval(n_bootstrap_samples, predictor_data_set, model_type, drop_columns):
    '''
    Calculate 95% confidence intervals for coefficents from regression model fits
        n_bootstrap_samples (int): number of times to resample for bootstrapping
        predictor_data_set (DataFrame): data frame with predictor and target data
        model_type (str): name of regression model used for fitting
        drop_columns (arr(str)): array of strings of name of columns to drop from features/x set
        rf_estimators (int): number of estimators to use for RandomForestRegressor
    '''
   
    if 'Estimate_Total' in predictor_data_set.columns:
        drop_columns.append('Estimate_Total')
        if model_type == 'Logit':
            data_dict = prepare_train_test_data(predictor_data_set, drop_columns, logit=True)
        else:
            data_dict = prepare_train_test_data(predictor_data_set, drop_columns)
    else:
        x_data = predictor_data_set.drop(columns=drop_columns)
        if model_type == 'Logit':
            data_dict = prepare_train_test_data(predictor_data_set, drop_columns, logit=True)
        else:
            data_dict = prepare_train_test_data(predictor_data_set, drop_columns)
        
    
    coef_list = run_bootstrap(n_bootstrap_samples, data_dict['x_scalar_train'], data_dict['y_train'], model_type)

    if model_type == 'Logit':
        coef_df = pd.DataFrame(coef_list,  columns=(data_dict['x_data_all'].columns).insert(0, 'Constant'))
        ### calculate lower CI (2.5%), upper CI (97.5%) and mean of each predictor coefficient
        ci_lower = np.percentile(coef_df, 2.5, axis=0)
        ci_upper = np.percentile(coef_df, 97.5, axis=0)
        coef_mean = np.mean(coef_df, axis=0).to_numpy()
        
        ci_results_df = pd.DataFrame({'Predictor': (data_dict['x_data_all'].columns).insert(0, 'Constant'),
                                                    'Mean': coef_mean,
                                                    'CI_Lower': ci_lower,
                                                    'CI_Upper': ci_upper})
    else:
        coef_df = pd.DataFrame(coef_list,  columns=data_dict['x_data_all'].columns)

        ### calculate lower CI (2.5%), upper CI (97.5%) and mean of each predictor coefficient
        ci_lower = np.percentile(coef_df, 2.5, axis=0)
        ci_upper = np.percentile(coef_df, 97.5, axis=0)
        coef_mean = np.mean(coef_df, axis=0).to_numpy()

        ci_results_df = pd.DataFrame({'Predictor': data_dict['x_data_all'].columns,
                                            'Mean': coef_mean,
                                            'CI_Lower': ci_lower,
                                            'CI_Upper': ci_upper})
                

    ci_results_df['Coeff Signficance'] = ['Not Signficant' if l <= 0 <= u else 'Signficant' for l,u in zip(ci_results_df['CI_Lower'], ci_results_df['CI_Upper'])]

    '''
    #### Rerun regression with signficant components
    if model_type == "Logit":
        df = pd.DataFrame(data_dict['x_scalar_train'], columns=(data_dict['x_data_all'].columns).insert(0, 'Constant'))
    else:
        df = pd.DataFrame(data_dict['x_scalar_train'], columns=data_dict['x_data_all'].columns)

    sig_predictors = ci_results_df[ci_results_df['Coeff Signficance'] == 'Signficant']['Predictor']
    sig_df = df[sig_predictors]
    x_scalar_train_sig, x_scalar_test_sig, y_train_sig, y_test_sig = train_test_split(sig_df, data_dict['y_data_all'], test_size=0.2, random_state=3241)
    '''

    ### Fit models for predictions
    if model_type == "Ridge":
        model_og = Ridge(alpha=1.0).fit(np.nan_to_num(data_dict['x_scalar_train']), data_dict['y_train'])
        #model_sig = Ridge(alpha=1.0).fit(np.nan_to_num(x_scalar_train_sig), y_train_sig)
    if model_type == "Linear":
        model_og = LinearRegression().fit(np.nan_to_num(data_dict['x_scalar_train']), data_dict['y_train'])
        #model_sig = LinearRegression().fit(np.nan_to_num(x_scalar_train_sig), y_train_sig)
    if model_type == "Logit":
        model_og = sm.Logit(data_dict['y_train'], np.nan_to_num(data_dict['x_scalar_train'])).fit()
        #model_sig = sm.Logit(y_train_sig, np.nan_to_num(x_scalar_train_sig)).fit()

    perf_dict = {}
    vac_rates_dict = {}
  
    y_predict_all = model_og.predict(np.nan_to_num(data_dict['x_scalar_test']))
    perf_dict['R2'] = r2_score(data_dict['y_test'], y_predict_all)
    perf_dict['MAE'] = mean_absolute_error(data_dict['y_test'], y_predict_all)
    perf_dict['RMSE'] = np.sqrt(mean_squared_error(data_dict['y_test'], y_predict_all))

    vac_rates_dict["predict"] = y_predict_all
    vac_rates_dict["test"] = data_dict['y_test']

    print(f"R2 = {perf_dict['R2']}, MAE = {perf_dict['MAE']}, RMSE = {perf_dict['RMSE']}")

    #y_predict_sig = model_sig.predict(np.nan_to_num(x_scalar_test_sig))
    #r2_sig = r2_score(y_test_sig, y_predict_sig)
    #vac_rates_dict["predict_sig"] = y_predict_sig
    #vac_rates_dict["test_sig"] = y_test_sig
    #print(f"R2_og = {r2_all}, R2_sig = {r2_sig}")


    return ci_results_df, perf_dict, vac_rates_dict



def run_bootstrap(n_bootstrap_samples, x_data, y_data, model_type):
    '''  
    Perform bootstrap method to determine CIs for predicator variables
        n_bootstrap_samples (int): number of times to resample for bootstrapping
        x_data (arr): predictor data
        y_data (arr): target data
        model_type (str): name of regression model used for fitting
    '''
    coef_list = []

    for i in range(n_bootstrap_samples):
        x_boot, y_boot = resample(x_data, y_data)

        if model_type == "Ridge":
            model = Ridge(alpha=1.0).fit(np.nan_to_num(x_boot), y_boot)
        if model_type == "Linear":
            model = LinearRegression().fit(np.nan_to_num(x_boot), y_boot)
        if model_type == "Logit":
            model = sm.Logit(y_boot, np.nan_to_num(x_boot)).fit(disp=False)
        
        if model_type == "Logit":
            coef_list.append(model.params.values)
        else:
            coef_list.append(model.coef_)

    return coef_list




def plot_vac_rates_predicted_actual(vac_rates_dict, sig_feats=False):
    '''
    Make scatter plot of predicted values and actual values for measles vaccination rates
        vac_rates_dict (dict): dictionary of target variables, either from testing set or predicted by model
        sig_feates (Boolean): if the predictions are from ONLY signifcant features or not
    '''
    if sig_feats == True:
        plt.scatter(vac_rates_dict['test_sig'], vac_rates_dict['predict_sig'])
    else:
        plt.scatter(vac_rates_dict['test'], vac_rates_dict['predict'])

    plt.xlabel("Actual MMR Vacination Rate")
    plt.ylabel("Predicted MMR Vacination Rate")
    plt.axline((0, 0), slope=1, color='black', linestyle="--")
    plt.grid()


def make_predicted_actual_histo(vac_rates_dict, num_test_bins, num_predict_bins, alpha_test, alpha_predict):
    '''
    Make two histgorams, one for predicted values and one for actual values, for measles vaccination rates
        vac_rates_dict (dict): dictionary of target variables, either from testing set or predicted by model
        num_test_bins (int): number of bins for test/acutal vaccination rates histogram
        num_predict_bins (int): number of bins for predicted vaccination rates histogram
    '''

    plt.hist(vac_rates_dict['test'], color='blue', label='Actual', bins=num_test_bins, alpha=alpha_test)
    plt.hist(vac_rates_dict['predict'], color='orange', label='Prediction', bins=num_predict_bins, alpha=alpha_predict)
    plt.xlabel('MMR Vaccination Rates')
    plt.legend()
    plt.grid()



def ml_alogrithm_analysis(predictor_data_set, model_type, drop_columns, hyper_params, algorithm_type, eval_obj):
    '''
    Train, predict and evaluate ML algorithm 
            predictor_data_set (DataFrame): data frame with predictor and target data
            model_type (str): name of ML algorithm to use
            drop_columns (arr(str)): array of strings of name of columns to drop from features/x set
            hyper_params (dict): hyperparameter values for ML model; n_estimators, learning_rate, max_depth, min_child_weight
            algorithm_type (str): Classification or Regression
    '''

    if eval_obj == "MAE":
        objective = 'reg:absoluteerror'
    if eval_obj == "RMSE":
        objective = 'reg:squarederror'
    
    data_dict = prepare_train_test_data(predictor_data_set, drop_columns)

    if algorithm_type == "Regression":
        if model_type == "RandomForest":
            model_og = xgb.XGBRFRegressor(objective=objective, n_estimators=hyper_params['n_estimators'], learning_rate=hyper_params['learning_rate'],
                                          max_depth=hyper_params['max_depth'], min_child_weight=hyper_params['min_child_weight'], gamma=hyper_params['gamma'],
                                          reg_lambda=hyper_params['reg_lambda'], reg_alpha=hyper_params['reg_alpha'], subsample=hyper_params['subsample'],
                                            colsample_bytree=hyper_params['colsample_bytree'],
                                          random_state=231).fit(data_dict['x_scalar_train'], data_dict['y_train'])
        if model_type == "Boost":
            model_og = xgb.XGBRegressor(objective=objective, n_estimators=hyper_params['n_estimators'], learning_rate=hyper_params['learning_rate'],
                                        max_depth=hyper_params['max_depth'], min_child_weight=hyper_params['min_child_weight'], gamma=hyper_params['gamma'],
                                        reg_lambda=hyper_params['reg_lambda'], reg_alpha=hyper_params['reg_alpha'], subsample=hyper_params['subsample'],
                                        colsample_bytree=hyper_params['colsample_bytree'],
                                        random_state=231).fit(data_dict['x_scalar_train'], data_dict['y_train'])
    if algorithm_type == "Classification":
        if model_type == "RandomForest":
            model_og = xgb.XGBRFClassifier(n_estimators=hyper_params['n_estimators'], learning_rate=hyper_params['learning_rate'],
                                          max_depth=hyper_params['max_depth'], min_child_weight=hyper_params['min_child_weight'], gamma=hyper_params['gamma'],
                                           reg_lambda=hyper_params['reg_lambda'], reg_alpha=hyper_params['reg_alpha'], subsample=hyper_params['subsample'],
                                            colsample_bytree=hyper_params['colsample_bytree'],
                                           random_state=231, bootstrap=True).fit(data_dict['x_scalar_train'], data_dict['y_train'])
        if model_type == "Boost":
            model_og = xgb.XGBClassifier(n_estimators=hyper_params['n_estimators'], learning_rate=hyper_params['learning_rate'],
                                          max_depth=hyper_params['max_depth'], min_child_weight=hyper_params['min_child_weight'], gamma=hyper_params['gamma'],
                                         reg_lambda=hyper_params['reg_lambda'], reg_alpha=hyper_params['reg_alpha'], subsample=hyper_params['subsample'],
                                            colsample_bytree=hyper_params['colsample_bytree'],
                                         random_state=231).fit(data_dict['x_scalar_train'], data_dict['y_train'])
 
    feat_importance = model_og.feature_importances_
    feat_importance_df = pd.DataFrame({'Predictor': data_dict['x_data_all'].columns, 'Importance': feat_importance}).sort_values('Importance', ascending=False)
    
    
    performance_dict = {}
    y_predict_all = model_og.predict(data_dict['x_scalar_test'])
    performance_dict['r2'] = r2_score(data_dict['y_test'], y_predict_all)
    performance_dict['MSE'] = mean_squared_error(data_dict['y_test'], y_predict_all)
    performance_dict['RMSE'] = np.sqrt(mean_squared_error(data_dict['y_test'], y_predict_all))
    performance_dict['MAE'] = mean_absolute_error(data_dict['y_test'], y_predict_all)

    if algorithm_type == "Classification":
        performance_dict['Accuracy'] = accuracy_score(data_dict['y_test'], y_predict_all)*100
        print(f"Accuracy = {performance_dict['Accuracy']}")
    else:
        print(f"MAE = {performance_dict['MAE']}, RMSE = {performance_dict['RMSE']}, R2 = {performance_dict['r2']}")
    
        
    vac_rates_dict = {}
    vac_rates_dict["predict"] = y_predict_all
    vac_rates_dict["test"] = data_dict['y_test']
    
    return feat_importance_df, performance_dict, vac_rates_dict


def k_fold_training(predictor_data_set, model_type, drop_columns, hyper_params, algorithm_type, n_k_splits, eval_obj):
    '''
    Train, predict and evaluate ML algorithm 
        predictor_data_set (DataFrame): data frame with predictor and target data
        model_type (str): name of ML algorithm to use
        drop_columns (arr(str)): array of strings of name of columns to drop from features/x set
        hyper_params (dict): hyperparameter values for ML model; n_estimators, learning_rate, max_depth, min_child_weight
        algorithm_type (str): Classification or Regression
    '''
    
    scaler = StandardScaler()
    drop_columns.append('MMR Vaccination Rate')
    x_data = predictor_data_set.drop(columns=drop_columns)
    x_scaler = scaler.fit_transform(x_data)
    y_data = predictor_data_set["MMR Vaccination Rate"]

    if eval_obj == "MAE":
        objective = 'reg:absoluteerror'
    if eval_obj == "RMSE":
        objective = 'reg:squarederror'

    if algorithm_type == "Regression":
        if model_type == "RandomForest":
            model = xgb.XGBRFRegressor(objective=objective, n_estimators=hyper_params['n_estimators'], learning_rate=hyper_params['learning_rate'],
                                              max_depth=hyper_params['max_depth'], min_child_weight=hyper_params['min_child_weight'], gamma=hyper_params['gamma'],
                                              reg_lambda=hyper_params['reg_lambda'], reg_alpha=hyper_params['reg_alpha'], subsample=hyper_params['subsample'],
                                            colsample_bytree=hyper_params['colsample_bytree'], 
                                              random_state=231)
        if model_type == "Boost":
            model = xgb.XGBRegressor(objective=objective, n_estimators=hyper_params['n_estimators'], learning_rate=hyper_params['learning_rate'],
                                            max_depth=hyper_params['max_depth'], min_child_weight=hyper_params['min_child_weight'], gamma=hyper_params['gamma'],
                                            reg_lambda=hyper_params['reg_lambda'], reg_alpha=hyper_params['reg_alpha'], subsample=hyper_params['subsample'],
                                            colsample_bytree=hyper_params['colsample_bytree'],
                                            random_state=231)
    if algorithm_type == "Classification":
        if model_type == "RandomForest":
            model = xgb.XGBRFClassifier(n_estimators=hyper_params['n_estimators'], learning_rate=hyper_params['learning_rate'],
                                            max_depth=hyper_params['max_depth'], min_child_weight=hyper_params['min_child_weight'], gamma=hyper_params['gamma'],
                                               reg_lambda=hyper_params['reg_lambda'], reg_alpha=hyper_params['reg_alpha'], subsample=hyper_params['subsample'],
                                            colsample_bytree=hyper_params['colsample_bytree'], 
                                               random_state=231, bootstrap=True)
        if model_type == "Boost":
            model = xgb.XGBClassifier(n_estimators=hyper_params['n_estimators'], learning_rate=hyper_params['learning_rate'],
                                              max_depth=hyper_params['max_depth'], min_child_weight=hyper_params['min_child_weight'], gamma=hyper_params['gamma'],
                                             reg_lambda=hyper_params['reg_lambda'], reg_alpha=hyper_params['reg_alpha'], subsample=hyper_params['subsample'],
                                            colsample_bytree=hyper_params['colsample_bytree'], 
                                             random_state=231)

    kfold = KFold(n_splits=n_k_splits, shuffle=True, random_state=231)

    r2_scores = cross_val_score(model, x_scaler, y_data, cv=kfold, scoring='accuracy')
    mae_scores = cross_val_score(model, x_scaler, y_data, cv=kfold, scoring='neg_mean_absolute_error')
    rmse_scores = cross_val_score(model, x_scaler, y_data, cv=kfold, scoring='neg_root_mean_squared_error')

    scores_dict = {}
    scores_dict['r2'] = r2_scores
    scores_dict['r2_mean'] = np.mean(r2_scores)
    scores_dict['mae'] = mae_scores
    scores_dict['mae_mean'] = np.mean(mae_scores)
    scores_dict['rmse'] = rmse_scores
    scores_dict['rmse_mean'] = np.mean(rmse_scores)

    return scores_dict



def k_fold_tuning(predictor_data_set, drop_columns, algorithm_type, n_k_splits, eval_metric):
    ''' 
    Perform hyperparmeter tuning on ML models using k-fold method
        predictor_data_set (DataFrame): data frame with predictor and target data
        algorithm_type (str): name of ML algorithm to use
        drop_columns (arr(str)): array of strings of name of columns to drop from features/x set
        n_k_splits (int): number k fold splits
        eval_metric (str): the type of performance metric to optimize for (MAE or MSE)
    '''
    data_dict = prepare_train_test_data(predictor_data_set, drop_columns)

    if algorithm_type=="Boost":
        model = xgb.XGBRegressor(random_state=231)
    if algorithm_type=='RandomForest':
        model = xgb.XGBRFRegressor(random_state=231)

    kf = KFold(n_splits=n_k_splits, shuffle=True, random_state=231)

    hyper_params = { 
                'learning_rate': stats.uniform(0.01, 0.2),
                'n_estimators': stats.randint(100, 1000),
                'max_depth' : stats.randint(3,8),
                'min_child_weight': stats.randint(1, 10),
                'subsample': stats.uniform(loc=0.6, scale=0.4),
                'colsample_bytree': stats.uniform(loc=0.6, scale=0.4),
                'gamma': stats.uniform(0.0, 0.5),
                'reg_lambda': stats.uniform(0.1, 10.0),
                'reg_alpha': stats.uniform(0.0, 1.0),
                }

    opt_dict ={}
    
    grid_search_r2 = RandomizedSearchCV(estimator=model, param_distributions=hyper_params, cv=kf, scoring="r2")
    grid_search_r2.fit(data_dict['x_scalar_train'], data_dict['y_train'])

    opt_dict['r2_hyperparams'] = grid_search_r2.best_params_
    opt_dict['r2'] = grid_search_r2.best_score_
    print(f"Best R2 = {grid_search_r2.best_score_}")

    if eval_metric == "MAE":
        grid_search_mae = RandomizedSearchCV(estimator=model, param_distributions=hyper_params, cv=kf, scoring="neg_mean_absolute_error")
        grid_search_mae.fit(data_dict['x_scalar_train'], data_dict['y_train'])
        opt_dict['mae_hyperparams'] = grid_search_mae.best_params_
        opt_dict['mae'] = grid_search_mae.best_score_
        print(f"Best MAE = {grid_search_mae.best_score_}")
    elif eval_metric == "MSE":
        grid_search_rmse = RandomizedSearchCV(estimator=model, param_distributions=hyper_params, cv=kf, scoring="neg_root_mean_squared_error")
        grid_search_rmse.fit(data_dict['x_scalar_train'], data_dict['y_train'])
        opt_dict['rmse_hyperparams'] = grid_search_rmse.best_params_
        opt_dict['rmse'] = grid_search_rmse.best_score_
        print(f"Best RMSE = {grid_search_rmse.best_score_}")

    return opt_dict


def create_MMR_classes(og_vac_rates_df, num_of_classes, class_edges):
    ''' 
    Replace MMR vacination rates by categories/classes for input into classficiation algorithm.
        og_vac_rates_df (DataFrame): data frame with actual MMR vacination rates
        num_of_classes (int): number of classifications to define
        class_edges (arr[float]): array of floats that defines the edges for each classfication
    '''

    class_mmr_vac_rates = og_vac_rates_df.copy()
    classRate = class_mmr_vac_rates['MMR Vaccination Rate']

    if num_of_classes == 2:
        class_mmr_vac_rates.loc[classRate < class_edges[0], 'MMR Vaccination Rate'] = 0.0
        class_mmr_vac_rates.loc[classRate >= class_edges[0], 'MMR Vaccination Rate'] = 1.0
    elif num_of_classes == 3:
        class_mmr_vac_rates.loc[classRate < class_edges[0], 'MMR Vaccination Rate'] = 0.0
        class_mmr_vac_rates.loc[(classRate >= class_edges[0]) & (classRate < class_edges[1]), 'MMR Vaccination Rate'] = 1.0
        class_mmr_vac_rates.loc[classRate >= class_edges[1], 'MMR Vaccination Rate'] = 2.0
    elif num_of_classes == 4:
        class_mmr_vac_rates.loc[classRate < class_edges[0], 'MMR Vaccination Rate'] = 0.0
        class_mmr_vac_rates.loc[(classRate >= class_edges[0]) & (classRate < class_edges[1]), 'MMR Vaccination Rate'] = 1.0
        class_mmr_vac_rates.loc[(classRate >= class_edges[1]) & (classRate < class_edges[2]), 'MMR Vaccination Rate'] = 2.0
        class_mmr_vac_rates.loc[classRate >= class_edges[2], 'MMR Vaccination Rate'] = 3.0
    elif num_of_classes == 5:
        class_mmr_vac_rates.loc[classRate < class_edges[0], 'MMR Vaccination Rate'] = 0.0
        class_mmr_vac_rates.loc[(classRate >= class_edges[0]) & (classRate < class_edges[1]), 'MMR Vaccination Rate'] = 1.0
        class_mmr_vac_rates.loc[(classRate >= class_edges[1]) & (classRate < class_edges[2]), 'MMR Vaccination Rate'] = 2.0
        class_mmr_vac_rates.loc[(classRate >= class_edges[2]) & (classRate < class_edges[3]), 'MMR Vaccination Rate'] = 3.0
        class_mmr_vac_rates.loc[classRate >= class_edges[3], 'MMR Vaccination Rate'] = 4.0
    elif num_of_classes == 6:
        class_mmr_vac_rates.loc[classRate < class_edges[0], 'MMR Vaccination Rate'] = 0.0
        class_mmr_vac_rates.loc[(classRate >= class_edges[0]) & (classRate < class_edges[1]), 'MMR Vaccination Rate'] = 1.0
        class_mmr_vac_rates.loc[(classRate >= class_edges[1]) & (classRate < class_edges[2]), 'MMR Vaccination Rate'] = 2.0
        class_mmr_vac_rates.loc[(classRate >= class_edges[2]) & (classRate < class_edges[3]), 'MMR Vaccination Rate'] = 3.0
        class_mmr_vac_rates.loc[(classRate >= class_edges[3]) & (classRate < class_edges[4]), 'MMR Vaccination Rate'] = 4.0
        class_mmr_vac_rates.loc[classRate >= class_edges[4], 'MMR Vaccination Rate'] = 5.0

    return class_mmr_vac_rates



def create_roc_curve(Y_test, pred_prob):
    ''' 
    Create ROC curve and print AUC for model. Doesn't support multi classification
    Args:
        RF_model (RandomForest): the model that has been trained
        X_test: the feature data for testing
        Y_test: the target/case data for testing
    '''

    #pred_prob = RF_model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresh = roc_curve(Y_test, pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label='ROC Curve (area = %0.4f)' % roc_auc) 
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    return roc_auc






def bootstrap_confidence_interval_pca(n_bootstrap_samples, predictor_data_set, model_type, num_components):
    '''
    Calculate 95% confidence intervals for coefficents from regression model fits using PCA and determine R2 with signficant components
        n_bootstrap_samples (int): number of times to resample for bootstrapping
        predictor_data_set (DataFrame): data frame with predictor and target data
        model_type (str): name of regression model used for fitting
    '''
    scaler = StandardScaler()
    pca_var = PCA(n_components=num_components)
    component_names = ['PC'+str(x) for x in range(num_components)]
    random_gen = random.randint(100, 9999)

    y_data = predictor_data_set['MMR Vaccination Rate']

    if 'Estimate_Total' in predictor_data_set.columns:
        x_data = predictor_data_set.drop(columns=['MMR Vaccination Rate', 'County', 'Estimate_Total']).fillna(0)
        x_scaler = scaler.fit_transform(x_data)
        x_pca = pca_var.fit_transform(x_scaler) 
        x_pca_train, x_pca_test, y_train, y_test = train_test_split(x_pca, y_data, test_size=0.2, random_state=random_gen)
    else:
        x_data = predictor_data_set.drop(columns=['MMR Vaccination Rate', 'County']).fillna(0)
        x_scaler = scaler.fit_transform(x_data)
        x_pca = pca_var.fit_transform(x_scaler) 
        x_pca_train, x_pca_test, y_train, y_test = train_test_split(x_pca, y_data, test_size=0.2, random_state=random_gen)

    coef_list = run_bootstrap(n_bootstrap_samples, x_pca_train, y_train, model_type)

    coef_df = pd.DataFrame(coef_list,  columns=component_names)

    ### calculate lower CI (2.5%), upper CI (97.5%) and mean of each predictor coefficient
    ci_lower = np.percentile(coef_df, 2.5, axis=0)
    ci_upper = np.percentile(coef_df, 97.5, axis=0)
    coef_mean = np.mean(coef_df, axis=0).to_numpy()

    ci_results_df = pd.DataFrame({'Predictor': component_names,
                                            'Mean': coef_mean,
                                            'CI_Lower': ci_lower,
                                            'CI_Upper': ci_upper})
    
    ci_results_df["Explained Variance"] = pca_var.explained_variance_ratio_
                
    ci_results_df['Coeff Signficance'] = ['Not Signficant' if l <= 0 <= u else 'Signficant' for l,u in zip(ci_results_df['CI_Lower'], ci_results_df['CI_Upper'])]



    #### Rerun regression with signficant components
    pc_df = pd.DataFrame(x_pca, columns=component_names)
    sig_predictors = ci_results_df[ci_results_df['Coeff Signficance'] == 'Signficant']['Predictor']
    pc_df_sig = pc_df[sig_predictors]

    x_sig_pca_train, x_sig_pca_test, y_sig_pca_train, y_sig_pca_test = train_test_split(pc_df_sig, y_data, test_size=0.2, random_state=random_gen)

    if model_type == "Ridge":
            model_sig_pca = Ridge(alpha=1.0).fit(x_sig_pca_train, y_sig_pca_train)
            model_og = Ridge(alpha=1.0).fit(x_pca_train, y_train)
            #model_sig_pca = Ridge(alpha=1.0).fit(pc_df_sig, y_data)
    if model_type == "Linear":
            model_sig_pca = LinearRegression().fit(x_sig_pca_train, y_sig_pca_train)
            model_og = LinearRegression().fit(x_pca_train, y_train)
            #model_sig_pca = LinearRegression().fit(pc_df_sig, y_data)


    #r2_dict = {}
    y_predict_og = model_og.predict(x_pca_test)
    r2_og = r2_score(y_test, y_predict_og)
    #r2_dict['all'] = r2_og

    y_predict_sig = model_sig_pca.predict(x_sig_pca_test)
    sig_pca_r2 = r2_score(y_sig_pca_test, y_predict_sig)


    vac_rates_dict = {}
    vac_rates_dict["predict_og"] = y_predict_og
    vac_rates_dict["test_og"] = y_test
    vac_rates_dict["predict_sig"] = y_predict_sig
    vac_rates_dict["test_sig"] = y_sig_pca_test


    return ci_results_df, r2_og, sig_pca_r2, vac_rates_dict
