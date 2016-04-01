# -*- coding: utf-8 -*-

#from ggplot import *
import time
from datetime import datetime
import pandas as pd
import seaborn as sns; sns.set()



'''
def plot_weather_data(turnstile_weather):
        turnstile_df = pd.read_csv(turnstile_weather, index_col=0) # import CSV as pandas dataframe
        date_fn_input = turnstile_df['DATEn'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d')) # store DATEn column into datetime format for manipulation   
        turnstile_df['Day of the week'] = date_fn_input.apply(lambda x: datetime.strftime(x, '%w')) # add a column for day of the week
        grouped_hour_weekday = turnstile_df['ENTRIESn_hourly'].groupby([turnstile_df['Day of the week'],turnstile_df['Hour']]).median() # average hourly entries grouped by day of week and hour of day
        weekly_heat_map = grouped_hour_weekday.unstack().transpose() # transpose matrix to present day of the week on the x-axis
        
        old_day = weekly_heat_map.columns # for ggplot x-axis values
        new_day = ['Sun','Mon','Tues','Wed','Thu','Fri','Sat'] # x-axis labels
        weekly_heat_map.rename(columns=dict(zip(weekly_heat_map.columns, new_day)), inplace=True) # change column headers
        
        old_hour = weekly_heat_map.index    
        new_hour = [] # populated by below for loops
        for e in range(12): #build time index for am times
            hour = str(e+1) + " am"
            new_hour.append(hour)
        for e in range(12): #build time index for pm times
            hour = str(e+1) + " pm"
            new_hour.append(hour)
        weekly_heat_map.rename(index=dict(zip(weekly_heat_map.index, new_hour)), inplace=True) # change row headers
        
        sns.heatmap(weekly_heat_map, annot=True,  fmt='.0f', linewidths=.5, cmap = "YlOrRd") # plot heatmap
        return 
        
plot_weather_data(r"data/turnstile_data_master_with_weather.csv")
'''
# import csv, add a day of the week column, and groupy by  
turnstile_df = pd.read_csv(r"data/turnstile_weather_v2.csv", index_col=0) 
turnstile_df.rename(columns={'day_week': 'Day of the Week', 'hour': 'Time Period' }, inplace=True) # assign more descriptive axis labels
grouped_hour_weekday = turnstile_df['ENTRIESn_hourly'].groupby([turnstile_df['Day of the Week'],turnstile_df['Time Period']]).median() # average hourly entries grouped by hour
weekly_heat_map = grouped_hour_weekday.unstack().transpose()    # transpose matrix to present day of the week on the x-axis

# assign alternate x and y axis titles
old_day = weekly_heat_map.columns                              
new_day = ['Mon','Tues','Wed','Thu','Fri','Sat','Sun']                                      # new x-axis labels
weekly_heat_map.rename(columns=dict(zip(weekly_heat_map.columns, new_day)), inplace=True)   # change column headers
old_hour = weekly_heat_map.index    
new_hour = ['8pm - 12am','12am - 4am', '4am - 8am', '8am - 12pm', '12pm - 4pm', '4pm - 8pm']                                                   

weekly_heat_map.rename(index=dict(zip(weekly_heat_map.index, new_hour)), inplace=True)      # change row headers

# plot heatmap

sns.heatmap(weekly_heat_map, annot=True,  fmt='.0f', linewidths=.5, cmap = "Reds")

#==============================================================================
# correlation matrix heat map
#==============================================================================s
'''
plt.figure(figsize=[8,6])
corr = turnstile_df[['ENTRIESn_hourly',
'EXITSn_hourly',
'Day of the week', # Day of the week (0-6)
'fog',
'precipi',
'rain']].corr()
sns.heatmap(corr)
plt.title('Correlation matrix between potential features')
plt.show()

print weekly_heat_map
'''
#for e in range(12):                         
#    hour = str(e+1) + " pm"
#
##        print weekly_heat_map.index
#        print weekly_heat_map.columns
#        sns.set_palette("deep", desat=.6)
#        sns.set_context(rc={"figure.figsize": (8,4)})
#        sns.heatmap(weekly_heat_map)
        
#        heat_map_long = sns.load_dataset("weekly_heat_map")
#        weekly_heat_map.pivot("Hour", "Day of the Week", "Hourly Traffic (ENTRIESn_Hourly)")
#        sns.heatmap(weekly_heat_map, annot=True, fmt="d", linewidths=.5)  
#        sns.heatmap(weekly_heat_map, annot=True,  fmt='.1f', linewidths=.5)
       
#        sns.palplot(sns.color_palette("RdBu_r", 7))


#==============================================================================
# 
#==============================================================================
#def plot_weather_data(turnstile_weather):
#    turnstile_df = pd.read_csv(turnstile_weather) # import CSV as pandas dataframe
#    turnstile_df.drop(turnstile_df.columns[[0]], axis=1, inplace=True)
##    df.drop(df.columns[[0]], axis=1, inplace=True)
##    print turnstile_df
#    
#    turnstile_df['full_time'] = pd.to_datetime(turnstile_df['DATEn'] + ' ' + turnstile_df['TIMEn']) 
#    turnstile_df['day_of_week'] = turnstile_df['full_time'].apply(lambda x: datetime.strftime(x, '%w'))
#
#    day_hour_grp = turnstile_df.groupby('day_of_week', as_index = False).mean()
#    old_day = day_hour_grp.index # for ggplot x-axis values
#    new_day = ['Sun','Mon','Tues','Wed','Thu','Fri','Sat'] # x-axis labels
#    day_hour_grp.rename(columns=dict(zip(day_hour_grp.index, new_day)), inplace=True) # change column headers through dict redirection
#    
#    print day_hour_grp
#    
#    plot = ggplot(day_hour_grp, aes(x=new_day, y='ENTRIESn_hourly')) + geom_bar(stat = 'identity') + \
#        ggtitle('Average ENTRIESn_hourly') + \
#        xlab('Day of the week') + \
#        ylab('ENTRIESn_hourly')
#    print plot
#==============================================================================
#     
#==============================================================================
    
#    print day_hour_grp
    #    df.sort_index(ascending=[False, True], by = ['salary','team']) # sort by salary descending and any salary ties by team ascending
#    print turnstile_df.day_of_week.unique()
    #    turnstile_df = turnstile_df.set_index('full_time')
#    turnstile_df['day_hour'] = turnstile_df['full_time'].apply(lambda x: datetime.strftime(x, '%w%H'))
#    print day_hour_grp('ENTRIESn_hourly').median()
#    print day_hour_grp.head()
#    plot = ggplot(turnstile_df, aes(x='Hour', color = 'day_of_week'))
#    print day_hour_grp['ENTRIESn_hourly']
#    print day_hour_grp.index
#    plot = ggplot(day_hour_grp, aes(x='day_of_week', fill='factor(ENTRIESn_hourly)')) + geom_bar() 
#    plot = ggplot(day_hour_grp, aes(x='Hour', y= 'ENTRIESn_hourly', color = 'day_of_week')) + geom_line()
    
#==================================M============================================
#     
##==============================================================================
#    #construct dataframe
#    test_data = pd.DataFrame({'random_1':pd.Series(np.random.randn(10)), 'category': pd.Series([2,2,1,1,4,4,3,3,5,5])})
#    #group data by category, avoiding turning the category into the index (though that could be OK in some circumstances?)
#    grouped_data = test_data.groupby('category', as_index = False).sum()
#    #take a look - is that what we want to plot?
#    print grouped_data
#    #plot a bar chart with the category on the x axis and the sum of 'random_1' as the height, using stat = 'identity' to do this.
#    plot = ggplot(grouped_data,aes(x = 'category', y= 'random_1')) + geom_point(stat = 'identity')
#    print plot
#==============================================================================
#     
#==============================================================================
    
#    plot_df['day_hour'] = plot_df['day_hour'].apply(lambda x: date.strtime(x, '%w-%H'))
#    plot_df = turnstile_df.set_index('day_hour')
#    plot_df['day_hour'] = pd.to_datetime(plot_df['day_hour'])
#    print plot_df['day_hour'].dtype
#    plot_df = plot_df.set_index(['day_time'])
#    print plot_df.index
#    plot_df = plot_df.reset_index()
#    plot_df['full_time'] = pd.to_datetime(turnstile_df['DATEn'] + ' ' + turnstile_df['TIMEn']) 
#    turnstile_df['day_hour'] = turnstile_df['full_time'].apply(lambda x: datetime.strftime(x, '%w-%H'))
    
#    p = ggplot(plot_df, aes(x = 'day_hour', y = 'ENTRIESn_hourly')) + geom_point()
#    print p
#    turnstile_df['full_time'] = pd.to_datetime(turnstile_df['DATEn'] + ' ' + turnstile_df['TIMEn']) 
#    turnstile_df['day_hour'] = turnstile_df['full_time'].apply(lambda x: datetime.strftime(x, '%w-%H'))
#    turnstile_df = turnstile_df.set_index('full_time')
#    plot_1 = plot_df.groupby(plot_df.index.day_time)
#    the1940s = ts.groupby(ts.index.year).sum().ix['1940-01-01':'1949-12-31']
#    print plot_df.dtypes
#    print plot_df.head()
#    print plot_df.describe()
#    print plot_df.index
#    print plot_df.day_time
    
   
#    assume = False
#    turnstile_df['day_hour'] = turnstile_df['full_time'].apply(lambda x: datetime.strftime(x, '%w-%H'))
#    print plot_df.index
#    print plot_df.head()
#    p = ggplot(plot_df, aes(plot_df.index, 'ENTRIESn_hourly')) + geom_line()
#    print p
    
#==============================================================================
#     print day_hour_grp.median().boxplot(column='ENTRIESn_hourly')
#==============================================================================

#    print day_hour_grp.ENTRIESn_hourly.mean().plot()
    
    
#    day_hour_grp_df = pd.DataFrame()
#    for line in day_hour_grp.median():
#        day_hour_grp_df = day_hour_grp_df.append(line)
  
#    print type(day_hour_grp)
#    print turnstile_df.head()
#    gg = ggplot(turnstile_df, aes('full_time', 'ENTRIESn_hourly')) + geom_line()
#        geom_point(color = 'red') + \
#        ggtitle('HR over time') + \
#        xlab('year') + \.
#        ylab('HR')
#    print gg
#
#    return turnstile_df
#    
#plot_weather_data("turnstile_data_master_with_weather.csv")
    
    #    print turnstile_df.day_hour.head()
#    day_of_wk = turnstile_df['full_time'].apply(lambda x: datetime.strftime(x, '%w'))
#    grouped_week_cycle_df = turnstile_df.groupby('day_hour', as_index = False).mean()   
#    grouped_week_cycle = pd.DataFrame(turnstile_df['ENTRIESn_hourly']).groupby(turnstile_df['day_hour'], as_index = False).mean()
#    grouped_week_cycle_df = pd.DataFrame({'day_time': grouped_week_cycle.index, 'entries':grouped_week_cycle.values})
    # Grouping the dataframe by hour
    
#    turnstile_weather_grp_by_hour = turnstile_weather.groupby('Hour' , as_index = False).sum()
#plot = ggplot( turnstile_weather_grp_by_hour , aes(x='Hour',y='ENTRIESn_hourly')) + xlim(-0.5,23.5) + geom_bar(stat='identity') + ggtitle("Entries per hour")
#    
#    grouped_week_cycle_df = pd.DataFrame(grouped_week_cycle.sum())
#    old_col = grouped_week_cycle.columns # for ggplot x-axis values
#    new_col = grouped_week_cycle.columns.apply(lambda x: str(x))
#    grouped_week_cycle.rename(index=dict(zip(old_col, new_col)), inplace=True) # change row headers
    
#    print dtype(grouped_week_cycle)
#    with open('test1.txt', 'w') as f:
#        f.write(str(grouped_week_cycle_df))       
#    new_col = string(grouped_week_cycle_df.columns)
#    old_col = grouped_week_cycle_df.columns # for ggplot x-axis values
#    grouped_week_cycle_df.rename(columns=dict(zip(old_col, new_col)), inplace=True) # change column headers
#    print grouped_week_cycle_df.index
#    print grouped_week_cycle_df.columns
    
#    old_hour = weekly_heat_map.index    
#    new_hour = [] # populated by below for loops
#    for e in range(12): #build time index for am times
#        hour = str(e+1) + " am"
#        new_hour.append(hour)
#    for e in range(12): #build time index for pm times
#        hour = str(e+1) + " pm"
#        new_hour.append(hour)
#    weekly_heat_map.rename(index=dict(zip(weekly_heat_map.index, new_hour)), inplace=True) # change row headers
    
#    turnstile_df.day_hour = day_hour
    #    print turnstile_df.day_hour
#    unique_day = day_hour[day_hour.str.contains("1-")].unique()
#    print day_hour.unique()
#    grouped_hour = turnstile_df['ENTRIESn_hourly'].groupby(turnstile_df['Hour'])
#    date_fn_input = turnstile_df['DATEn'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d')) # store DATEn column into datetime format for manipulation   

#    turnstile_df['day_of_wk'] = date_fn_input.apply(lambda x: datetime.strftime(x, '%w')) # add a column for day of the week
##    
#    combined_time = pd.to_datetime(str(turnstile_df['DATEn']) + ' ' + str(turnstile_df['Hour']), format = '%Y-%m-%d %H')
#    df.full_time(combined_time, inplace=True)

    
#    turnstile_df['day_plus_hour'] = turnstile_df['day_of_wk'].map(str) + turnstile_df['TIMEn'][:2].map(str)
#==============================================================================
# heat map  
#==============================================================================
#    grouped_hour_weekday = turnstile_df['ENTRIESn_hourly'].groupby([turnstile_df['day_of_wk'],turnstile_df['Hour']]).mean() # average hourly entries grouped by day of week and hour of day
#    weekly_heat_map = grouped_hour_weekday.unstack().transpose() # transpose matrix to present day of the week on the x-axis
#    old_day = weekly_heat_map.columns # for ggplot x-axis values
#    new_day = ['Sun','Mon','Tues','Wed','Thu','Fri','Sat'] # x-axis labels
#    weekly_heat_map.rename(columns=dict(zip(weekly_heat_map.columns, new_day)), inplace=True) # change column headers
#    
#    old_hour = weekly_heat_map.index    
#    new_hour = [] # populated by below for loops
#    for e in range(12): #build time index for am times
#        hour = str(e+1) + " am"
#        new_hour.append(hour)
#    for e in range(12): #build time index for pm times
#        hour = str(e+1) + " pm"
#        new_hour.append(hour)
#    weekly_heat_map.rename(index=dict(zip(weekly_heat_map.index, new_hour)), inplace=True) # change row headers
    
#    gg = ggplot(weekly_heat_map, aes(x= new_day , y=new_hour, fill= weekly_heat_map )) +\
#        geom_tile() + scale_colour_gradient(low="steelblue", high="red")
#==============================================================================
# line chart of weekly traffic
#==============================================================================
#    print turnstile_df[0:1]
#    print turnstile_df['full_time'].unique()
#    print grouped_hour_weekday


#    turnstile_weather_grp_by_hour = turnstile_df.groupby('Hour' , as_index = False).median()
#    weekly_cycle_plot = (turnstile_df, aes())
    
#    plot = ggplot(weekly_heat_map , aes(x='old_hour',y='Mon')) + xlim(-0.5,23.5) + geom_bar(stat='identity') + ggtitle("Entries per hour")
#    print plot
    
#    weekly_cylce_chart = ggplot(data = turnstile_df, aes(x='factor(day_of_wk)', fill ='factor(day_of_wk)')) + geom_bar()

#    print weekly_cylce_chart
#    print meat.head()

#==============================================================================
# organizing data for ggplot input
#==============================================================================





#

#    geom_line(color = 'red') + geom_point(color = 'red') + \


##==============================================================================
#    print weekly_heat_map.index
#    print weekly_heat_map.columns
'''
    You are passed in a dataframe called turnstile_weather. 
    Use turnstile_weather along with ggplot to make a data visualization
    focused on the MTA and weather data we used in assignment #3.  
    You should feel free to implement something that we discussed in class 
    (e.g., scatterplots, line plots, or histograms) or attempt to implement
    something more advanced if you'd like.  

    Here are some suggestions for things to investigate and illustrate:
     * Ridership by time of day or day of week
     * How ridership varies based on Subway station (UNIT)
     * Which stations have more exits or entries at different times of day
       (You can use UNIT as a proxy for subway station.)

    If you'd like to learn more about ggplot and its capabilities, take
    a look at the documentation at:
    https://pypi.python.org/pypi/ggplot/
     
    You can check out:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv
     
    To see all the columns and data points included in the turnstile_weather 
    dataframe. 
     
    However, due to the limitation of our Amazon EC2 server, we are giving you a random
    subset, about 1/3 of the actual data in the turnstile_weather dataframe.
    '''
##==============================================================================
#==============================================================================
# change column headings
#==============================================================================
#    print weekly_heat_map.columns
#    weekly_heat_map = pd.DataFrame(weekly_heat_map)
#   #    old_day = [u'0', u'1', u'2', u'3', u'4', u'5', u'6'] 
#    new_day = ['Sun','Mon','Tues','Wed','Thu','Fri','Sat']
#    weekly_heat_map.rename(columns=dict(zip(old_day, new_day)), inplace=True)
    
#    date_fn_input = turnstile_df['DATEn'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
#    grouped_hour = turnstile_df['ENTRIESn_hourly'].groupby(turnstile_df['Hour'])
#    print new_hour
#    day_of_wk = date_fn_input.apply(lambda x: datetime.strftime(x, '%w'))
#    
#    print day_of_wk.head()
#    dummy_day_of_wk = pd.get_dummies(day_of_wk, prefix='day_of_wk')
##    GROUPBY Hour, cast(strftime('%w', DATEn) as integer)
#    q = """
#    SELECT "C/A", UNIT, SCP, DATEn, TIMEn, DESCn, ENTRIESn
#    FROM turnstile_data_all
#    WHERE DESCn == 'REGULAR'
##    """
#    q="""
#    SELECT ENTRIESn_hourly
#    FROM turnstile_df
#    GROUPBY Hour
#    """
#    turnstile_data = pandasql.sqldf(q, locals())
##    turnstile_data = pandasql.sqldf(q.lower(), locals())
#    turnstile_data = pandasql.sqldf(q, locals())
#    print turnstile_data  

# gg = ggplot.ggplot(hr_year_df, aes('yearID', 'HR')) + \
#        geom_line(color = 'red') + \
#        geom_point(color = 'red') + \
#        ggtitle('HR over time') + \
#        xlab('year') + \
#        ylab('HR')

#if __name__ == "__main__":
#    data = "hr_by_team_year_sf_la.csv"
#    image = "plot.png"
#    gg =  lineplot_compare(data)
#    ggsave(image, gg, width=11, height=8)

#    print weekly_heat_map.index
