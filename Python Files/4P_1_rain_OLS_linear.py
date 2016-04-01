import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
import statsmodels.api as sm
from datetime import datetime
import matplotlib.pyplot as plt
import pylab
import seaborn as sns

sns.set(style="whitegrid")


def linear_regression(features, values):
    y = pd.DataFrame(values) # response variable
    x = pd.DataFrame(features) # predictors  or features
    x = sm.add_constant(x) # adds a constant term to the predictors dataframe
    est = sm.OLS(y,x) # perform the regression of the predictors on the response, using the sm.OLS class and its initialization OLs(y, x)
    est = est.fit() # estimate the parameters?
    with open('results_rain_OLS_linear.txt', 'w') as f:
           f.write(str(est.summary()))           
    intercept = est.params['const'] 
    params = est.params.ix[1:]
    return intercept, params

def plot_residuals(turnstile_weather, predictions):
#    turnstile_weather['ENTRIESn_hourly'] = (turnstile_weather.ENTRIESn_hourly + 1) # log transformation 
    residuals = turnstile_weather['ENTRIESn_hourly'] - predictions
#    plt.figure()
#    residuals.hist(alpha=1, bins=100, label='ENTRIESn_hourly residuals')
#    plt.title("Residuals Histogram") # add a title
#    plt.ylabel("Frequency") # add a label to the y-axis
#    plt.xlabel("ENTRIESn_hourly residuals") 
##    plt.legend() # add a legend
#    plt.show()
    
    stats.probplot(residuals, dist="norm", plot=pylab)
    print 'linear QQ plot'
    pylab.show()
    
    return plt

def predictions(dataframe):
    
#    dataframe['ENTRIESn_hourly'] = np.log1p(dataframe.ENTRIESn_hourly) # log transformation 
    
#    features = dataframe[['meantempi']]
    features = dataframe[['rain']]
#    dummy_rain = pd.get_dummies(dataframe['rain'], prefix='rain')
#    dummy_unit = pd.get_dummies(dataframe['UNIT'], prefix='unit')
#    dummy_hour = pd.get_dummies(dataframe['hour'], prefix='hour')
#    dummy_day_week = pd.get_dummies(dataframe['day_week'], prefix='day_week')
#    features = features.join(dummy_rain)
    
#    removing one dummy from each group to reduce multicollinearity
#    features.drop(['unit_R003'], axis = 1, inplace = True)
#    features.drop(['hour_0'], axis = 1, inplace = True)
#    features.drop(['day_week_0'], axis = 1, inplace = True)   
#    features.drop(['rain_0'], axis = 1, inplace = True)   
    print features
#    values_log = dataframe['log_ENTRIESn_hourly']
    values = dataframe['ENTRIESn_hourly']
    
#    Perform linear regression
#    intercept, params = linear_regression(features, values_log)    
    intercept, params = linear_regression(features, values)    
    predictions = intercept + np.dot(features, params)
#    log_predictions [log_predictions<0] = 1
#    predictions = np.expm1(log_predictions) # inverse logarithmic transformation to produce ENTRIESn_hourly   
#    residuals = values - predictions
    residuals = values - predictions
    
    print predictions[:5]
    print values[:5]

    '''
    plt.figure()
    residuals.hist(alpha=1, bins=100, label='ENTRIESn_hourly residuals')
    plt.title("Residuals Histogram") # add a title
    plt.ylabel("Frequency") # add a label to the y-axis
    plt.xlabel("ENTRIESn_hourly residuals") 
#    plt.legend() # add a legend
    plt.show()
    '''
    
#    print 'log linear QQ plot'
#    sns.residplot(values_nl, predictions, lowess=True, color="navy")
               
#    plot qq plot
    stats.probplot(residuals, dist="norm", plot=pylab)
    pylab.show()
    return predictions
    
#df = pd.read_csv(r"data/turnstile_data_master_with_weather.csv")
df = pd.read_csv(r"data/turnstile_weather_v2.csv")
predictions(df)
#plot_residuals(turnstile_weather, predictions(turnstile_weather))
