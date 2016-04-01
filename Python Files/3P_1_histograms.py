import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ggplot import *

# import csv and separate data into rain and no rain dataframes 
df = pd.read_csv("data/turnstile_weather_v2.csv", index_col=0)
df[df.ENTRIESn_hourly >= 6000] = 6000       # limit outliers
rain_df = df[df.rain==1]                    # create rainy days df
no_rain_df = df[df.rain==0]                 # create non rainy days df

# plot rain and no rain NYC subway traffic histograms
no_rain_df.ENTRIESn_hourly.hist(alpha=.5, bins=50, label='No Rain', color = 'darkgreen') 
rain_df.ENTRIESn_hourly.hist(alpha=1, bins=50, label='Rain', color = 'lightblue') 
plt.ylabel("Frequency")                     # add label to the y-axis
plt.xlabel("ENTRIESn_hourly")               # add label to the x-axis
plt.legend()                                # add legend
