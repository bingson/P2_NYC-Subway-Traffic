import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import datetime

"""
In this question, you need to:
1) implement the linear_regression() procedure
2) Select features (in the predictions procedure) and make predictions.

"""

def linear_regression(features, values):
    """
    Perform linear regression given a data set with an arbitrary number of features.
    
    This can be the same code as in the lesson #3 exercise.
    """
    
    ###########################
    ### YOUR CODE GOES HERE ###
    ###########################
    
    y = pd.DataFrame(values) # response variable
    x = pd.DataFrame(features) # predictors  or features
    x = sm.add_constant(x) # adds a constant term to the predictors dataframe
#    x.drop(['unit_R001'], axis = 1, inplace = True)
#    x.drop(['hour_0'], axis = 1, inplace = True)
#    x.drop(['day_of_wk_0'], axis = 1, inplace = True)

    est = sm.OLS(y,x) # perform the regression of the predictors on the response, using the sm.OLS class and its initialization OLs(y, x)
    est = est.fit() # estimate the parameters?
    
    with open('results_dummyfix.txt', 'w') as f:
           f.write(str(est.summary()))
           
#    est.summary().to_csv("summary.csv")
    intercept = est.params['const']
#    intercept = 0           MZ
    params = est.params.ix[1:]
#    print intercept
    return intercept, params

#"turnstile_data_master_with_weather.csv"

#==============================================================================
# test cases
#==============================================================================
#y = [11,14,19,26] # y-values
#x = [[1,3],[2,4],[3,3],[4,5]] # x-values
#
#linear_regression(x, y)

def predictions(dataframe):
    '''
    The NYC turnstile data is stored in a pandas dataframe called weather_turnstile.
    Using the information stored in the dataframe, let's predict the ridership of
    the NYC subway using linear regression with gradient descent.
    
    You can download the complete turnstile weather dataframe here:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv    
    
    Your prediction should have a R^2 value of 0.40 or better.
    You need to experiment using various input features contained in the dataframe. 
    We recommend that you don't use the EXITSn_hourly feature as an input to the 
    linear model because we cannot use it as a predictor: we cannot use exits 
    counts as a way to predict entry counts. 
    
    Note: Due to the memory and CPU limitation of our Amazon EC2 instance, we will
    give you a random subet (~10%) of the data contained in 
    turnstile_data_master_with_weather.csv. You are encouraged to experiment with 
    this exercise on your own computer, locally. If you do, you may want to complete Exercise
    8 using gradient descent, or limit your number of features to 10 or so, since ordinary
    least squares can be very slow for a large number of features.
    
    If you receive a "server has encountered an error" message, that means you are 
    hitting the 30-second limit that's placed on running your program. Try using a
    smaller number of features.
    '''
    dataframe = pd.read_csv(dataframe)
    ################################ MODIFY THIS SECTION #####################################
    # Select features. You should modify this section to try different features!             #
    # We've selected rain, precipi, Hour, meantempi, and UNIT (as a dummy) to start you off. #
    # See this page for more info about dummy variables:                                     #
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html          #
    ##########################################################################################   
    features = dataframe[['rain']]
#    dummy_rain = pd.get_dummies(dataframe['rain'], prefix='rain')
    dummy_unit = pd.get_dummies(dataframe['UNIT'], prefix='unit')
    dummy_hour = pd.get_dummies(dataframe['Hour'], prefix='hour')

#   dummification of days of the week derived from DATEn column
    date_fn_input = dataframe['DATEn'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    day_of_wk = date_fn_input.apply(lambda x: datetime.strftime(x, '%w'))
    dummy_day_of_wk = pd.get_dummies(day_of_wk, prefix='day_of_wk')

    features = features.join(dummy_unit).join(dummy_hour).join(dummy_day_of_wk)
    
#    removing one dummy from each group to reduce multicollinearity
    features.drop(['unit_R001'], axis = 1, inplace = True)
    features.drop(['hour_0'], axis = 1, inplace = True)
    features.drop(['day_of_wk_0'], axis = 1, inplace = True)
    
        
    values = dataframe['ENTRIESn_hourly']
    
    
    # Perform linear regression
    intercept, params = linear_regression(features, values)
    
    predictions = intercept + np.dot(features, params)
    return predictions
    
#def day_of_week()
    
predictions("data/turnstile_data_master_with_weather.csv")