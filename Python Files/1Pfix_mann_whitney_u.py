import numpy as np
import scipy.stats as stats
import pandas as pd

def mann_whitney_plus_means(turnstile_weather): 
    rain_ENTRIESn_hourly = turnstile_weather[turnstile_weather['rain']==1].ENTRIESn_hourly #equivalent => turnstile_weather[turnstile_weather.rain == 1]["ENTRIESn_hourly"]
    no_rain_ENTRIESn_hourly = turnstile_weather[turnstile_weather['rain']==0].ENTRIESn_hourly   

    with_rain_mean = np.mean(rain_ENTRIESn_hourly) # the mean of entries with rain
    without_rain_mean = np.mean(no_rain_ENTRIESn_hourly) # the mean of entries without rain
    U, p = stats.mannwhitneyu(rain_ENTRIESn_hourly, no_rain_ENTRIESn_hourly)
    n1, n2 = ((len(rain_ENTRIESn_hourly), len(no_rain_ENTRIESn_hourly)))
    m_u = n1*n2/2.0
    s_u = (n1*n2*(n1+n2+1)/12)**0.5
    z = (U-m_u)/s_u
    p = 2*stats.norm.cdf(z) #2-tailed p-value
    return with_rain_mean, without_rain_mean, U, p

if __name__ == "__main__":
    input_filename = "data/turnstile_weather_v2.csv"
    turnstile_master = pd.read_csv(input_filename)
    student_output = mann_whitney_plus_means(turnstile_master)

    print student_output