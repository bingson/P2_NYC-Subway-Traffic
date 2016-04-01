# -*- coding: utf-8 -*-
import seaborn as sns
#from scipy import stats, integrate
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from ggplot import *
"""
Created on Fri Jul 31 18:50:49 2015

@author: Bing
"""




#def histogram(df,category,group_x, y, bins):
#     df[df[category]==group_x].head()
#    plt.hist(df[category=1])
#    grouped = df.groupby(category)
#    grouped.get_group(group_x)[y_axis].hist(bins=bins)

#seaborn settings
#sns.set_palette("deep", desat=.6)
#sns.set_context(rc={"figure.figsize": (8,4)})
    

#plt.rain_df.fare.hist(bins=30)
#sns.distplot(rain_df.ENTRIESn_hourly, bins = 30, kde = False, rug = False)
#
### Histogram
#     create data series (x-values) for histogram
#
#    
#df[df.ENTRIESn_hourly >= 6000] = 6000       # limit the distorting influence of outliers
#print df.ENTRIESn_hourly.min()
#print df.log_ENTRIESn_hourly.head()
#print df.log_ENTRIESn_hourly.min()
#df.ENTRIESn_hourly = df.ENTRIESn_hourly + 1

# import csv, munge data, apply logarithmic transformation to ENTRIESn_hourly
df = pd.read_csv("data/turnstile_weather_v2.csv", index_col=0)
df_zero = df[df.ENTRIESn_hourly == 0]
print df_zero.ENTRIESn_hourly.value_counts()
zero_hours = df_zero.hour.unique()
#df = df[df.ENTRIESn_hourly != 0] 
df['log_ENTRIESn_hourly'] = np.log1p(df.ENTRIESn_hourly) #log transformation 
pd.options.mode.chained_assignment = None
df.is_copy = False
rain_df = df[df.rain==1]    
no_rain_df = df[df.rain==0]     

# plot rain and no rain NYC subway traffic histograms            
no_rain_df.log_ENTRIESn_hourly.hist(alpha=.5, bins=50, label='No Rain', color = 'darkgreen') 
rain_df.log_ENTRIESn_hourly.hist(alpha=1, bins=50, label='Rain', color = 'lightblue') 
plt.ylabel("Frequency")                         # add label to the y-axis
plt.xlabel("Log_ENTRIESn_hourly")               # add label to the x-axis
plt.legend()                                    # add legend


'''#plt.hist(rain_df.log_ENTRIESn_hourly, alpha=.3)
#sns.rugplot(rain_df.log_ENTRIESn_hourly)'''

# log transformation of hourly entries data
# plot rain and no rain NYC subway traffic histograms
#plt.title("Histogram of Log_ENTRIESn_hourly")   # add title
    
#df.plot(colormap='cubehelix')
#    rain_ENTRIESn_hourly.plot(kind = 'hist', alpha = 0.5)
#    no_rain_ENTRIESn_hourly.plot(kind = 

#plot = ggplot(rain_df, aes(x='ENTRIESn_hourly')) + \
#    geom_histogram(fill='red', binwidth=600) +xlim(0, 5000) + \
#    xlab('ENTRIESn_hourly') + ylab('Frequency') + ggtitle('Hourly Entries during rain')
#print plot
#
#plot = ggplot(entries_when_rain, aes(x='ENTRIESn_hourly')) \
#    + geom_histogram(fill='red', binwidth=600) + xlim(0, 5000) \
#    + xlab('ENTRIESn_hourly') +ylab('Frequency') + ggtitle('Hourly Entries When Rain')
#
#plot = ggplot(entries_when_rain, aes(x='ENTRIESn_hourly')) \
#    + geom_histogram(fill='red', binwidth=600) + xlim(0, 5000) \
#    + xlab('ENTRIESn_hourly') +ylab('Frequency') + ggtitle('Hourly Entries When Rain')
    
#sns.boxplot(rain_df.ENTRIESn_hourly)
#print rain_df.ENTRIESn_hourly.head()
#plt.hist(rain_df.ENTRIESn_hourly.dropna(), 30)
#plt.hist(rain_df.ENTRIESn_hourly, 30, color = sns.desaturate("indianred",1))
#
#category = 'rain'
#group_x= 1
#y = 'ENTRIESn_hourly'
#bins = 100    

#histogram(df,category,group_x, y_axis, bins)