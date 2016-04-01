import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
import scipy
import statsmodels
import statsmodels.api as sm
from datetime import datetime


"""
In this question, you need to:
1) Implement the linear_regression() procedure using gradient descent.
   You can use the SGDRegressor class from sklearn, since this class uses gradient descent.
2) Select features (in the predictions procedure) and make predictions.

"""

def normalize_features(features):
    ''' 
    Returns the means and standard deviations of the given features, along with a normalized feature
    matrix.
    ''' 
    means = np.mean(features, axis=0)
    std_devs = np.std(features, axis=0)
    normalized_features = (features - means) / std_devs
    return means, std_devs, normalized_features

def recover_params(means, std_devs, norm_intercept, norm_params):
    ''' 
    Recovers the weights for a linear model given parameters that were fitted using
    normalized features. Takes the means and standard deviations of the original
    features, along with the intercept and parameters computed using the normalized
    features, and returns the intercept and parameters that correspond to the original
    features.
    ''' 
    intercept = norm_intercept - np.sum(means * norm_params / std_devs)
    params = norm_params / std_devs
    return intercept, params

def linear_regression_OLS(features, values):
    """
    Perform linear regression given a data set with an arbitrary number of features.
    """  
    y = pd.DataFrame(values) # response variable
    x = pd.DataFrame(features) # predictors  or features
    x = sm.add_constant(x) # adds a constant term to the predictors dataframe
    est = sm.OLS(y,x) # perform the regression of the predictors on the response, using the sm.OLS class and its initialization OLs(y, x)
    est = est.fit() # estimate the parameters?
    intercept = est.params['const']
    params = est.params.ix[1:]
   
    with open('results_0.txt', 'w') as f:
        f.write(str(est.summary()))
           
    return intercept, params
    
def linear_regression_GD(features, values, theta, alpha, num_iterations):
    m = len(values) # how many data points we have
    cost_history = []
    for i in range(num_iterations):
        predicted_values = np.dot(features, theta)
        theta = theta - alpha / m * np.dot((predicted_values - values), features)
        cost = compute_cost(features, values, theta)
        cost_history.append(cost)
    return theta, pandas.Series(cost_history) # leave this line for the grader

def linear_regression_SGD(features, values):
    clf = SGDRegressor().fit(features,values)
    intercept = clf.intercept_
    params = clf.coef_       
    return intercept, params

def predictions(dataframe):
    features = dataframe[['meantempi']]
    
    dummy_unit = pd.get_dummies(dataframe['UNIT'], prefix='unit')
    dummy_hour = pd.get_dummies(dataframe['Hour'], prefix='hour')
    date_fn_input = dataframe['DATEn'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    day_of_wk = date_fn_input.apply(lambda x: datetime.strftime(x, '%w')) # isolate day of the week for dummy variable
    dummy_day_of_wk = pd.get_dummies(day_of_wk, prefix='day_of_wk')
    
    features = features.join(dummy_unit).join(dummy_hour).join(dummy_day_of_wk)
    values = dataframe['ENTRIESn_hourly']

    features_array = features.values 
    values_array = values.values
    print pd.DataFrame(features_array).head()
    
    means, std_devs, normalized_features_array = normalize_features(features_array)

    # Perform ordinary least squares regression
    predictions_OLS = norm_intercept + np.dot(normalized_features_array, norm_params)
    # print pd.DataFrame(predictions).head()

    # Perform gradient descent
    
        

    # Perform stochastic gradient descent
    '''norm_intercept, norm_params = linear_regression_SGD(normalized_features_array, values_array)
    intercept, params = recover_params(means, std_devs, norm_intercept, norm_params)
    predictions = intercept + np.dot(features_array, params)'''
    # The following line would be equivalent:

    return predictions_OLS, predictions_GD, predictions_SGD

def plot_residuals(turnstile_weather, predictions):
    residuals = turnstile_weather['ENTRIESn_hourly'] - predictions
    plt.figure()
    residuals.hist(alpha=1, bins=100, label='ENTRIESn_hourly residuals')
    plt.title("Residuals Histogram") # add a title
    plt.ylabel("Frequency") # add a label to the y-axis
    plt.xlabel("ENTRIESn_hourly residuals") 
#    plt.legend() # add a legend
    plt.show()
    return plt

def compute_r_squared(data, predictions):
    SSReg = ((predictions-np.mean(data))**2).sum()
    SST = ((data-np.mean(data))**2).sum()
    r_squared = SSReg / SST
    return r_squared
    

turnstile_weather = pd.read_csv(r"data\turnstile_data_master_with_weather.csv")       
plot_residuals(turnstile_weather, predictions(turnstile_weather))
print "r^2: ", compute_r_squared(turnstile_weather['ENTRIESn_hourly'], predictions(turnstile_weather))
