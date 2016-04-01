import numpy as np
import pandas as pd
import scipy.stats as stats
from datetime import datetime
import matplotlib.pyplot as plt
import pylab
from sklearn.linear_model import SGDRegressor
import seaborn as sns

sns.set(style="whitegrid")

def normalize_features(features):
    means = np.mean(features, axis=0)
    std_devs = np.std(features, axis=0)
    normalized_features = (features - means) / std_devs
    return means, std_devs, normalized_features

def recover_params(means, std_devs, norm_intercept, norm_params):
    intercept = norm_intercept - np.sum(means * norm_params / std_devs)
    params = norm_params / std_devs
    return intercept, params

def linear_regression(features, values):
    y = pd.DataFrame(values) # response variable
    x = pd.DataFrame(features) # predictors  or features
    x = sm.add_constant(x) # adds a constant term to the predictors dataframe
    est = sm.OLS(y,x) # perform the regression of the predictors on the response, using the sm.OLS class and its initialization OLs(y, x)
    est = est.fit() # estimate the parameters?
    with open('results_OLS_loglinear.txt', 'w') as f:
           f.write(str(est.summary()))           
    intercept = est.params['const'] 
    params = est.params.ix[1:]
    return intercept, params


def linear_regression_SGD(features, values):
    clf = SGDRegressor(loss='squared_loss', alpha = 0.00001, n_iter = 1000, 
                       shuffle = True, random_state = 2000).fit(features,values)
    intercept = clf.intercept_
    params = clf.coef_           
    return intercept, params

def predictions(dataframe):
    
    dataframe['log_ENTRIESn_hourly'] = np.log1p(dataframe.ENTRIESn_hourly) # log transformation 
    
    features = dataframe[[]] # option 2: features = dataframe[['meantempi', 'rain']]
    dummy_unit = pd.get_dummies(dataframe['UNIT'], prefix='unit')
    dummy_hour = pd.get_dummies(dataframe['hour'], prefix='hour')
    dummy_day_week = pd.get_dummies(dataframe['day_week'], prefix='day_week')
    features = features.join(dummy_hour).join(dummy_day_week).join(dummy_unit) #join(dummy_rain).
    
#    removing one dummy from each group to avoid dummy variable trap
    features.drop(['unit_R003'], axis = 1, inplace = True) 
    features.drop(['hour_0'], axis = 1, inplace = True)
    features.drop(['day_week_0'], axis = 1, inplace = True)   
    values = dataframe['ENTRIESn_hourly']
    values_log = dataframe['log_ENTRIESn_hourly']
    
#    Perform linear regression
    intercept, params = linear_regression_SGD(features, values_log)    
    log_predictions = intercept + np.dot(features, params)
    log_predictions[log_predictions<0] = 1
    predictions = np.expm1(log_predictions) # inverse logarithmic transformation to produce ENTRIESn_hourly   
    residuals = values - predictions

    return predictions

def compute_r_squared(data, predictions):
    SSReg = ((predictions-np.mean(data))**2).sum()
    SST = ((data-np.mean(data))**2).sum()
    r_squared = SSReg / SST
    return r_squared
    
df = pd.read_csv(r"data/turnstile_weather_v2.csv")
#predictions(df)
print "r^2: ", compute_r_squared(df['ENTRIESn_hourly'], predictions(df))
#plot_residuals(df, predictions(df))
