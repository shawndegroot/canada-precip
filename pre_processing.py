import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
import glob

fld_new = 'hum_train_final.csv'

# pre-processing
stations = glob.glob('stations??.csv')
stations2 = glob.glob('stations?.csv') 

LIST = []
# read all 20 stations CSVs (downloaded from ClimateData.ca)
stations_final = stations + stations2

for i in stations_final:
     df = pd.read_csv(i, index_col=None, header=0)
     LIST.append(df)
 
frame = pd.concat(LIST, axis=0, ignore_index=True)

#filtering for humidity
ff = frame[frame['MAX_REL_HUMIDITY'].notna()]

# check for duplicate rows:
pd.concat(g for _, g in ff.groupby("MIsN_TEMPERATURE") if len(g) > 1)
ff['STATION_NAME'].nunique()

#ff.to_csv('hum_train.csv')

dft = pd.read_csv(fld_new)
count=dft.count().sort_values()

dft = dft.drop(columns=['STATION_NAME','STN_ID','LOCAL_DATE', 'CLIMATE_IDENTIFIER','ID','PROVINCE_CODE',
                     'LOCAL_YEAR','LOCAL_MONTH','LOCAL_DAY','MEAN_TEMPERATURE_FLAG','MIN_TEMPERATURE_FLAG',
                      'MAX_TEMPERATURE_FLAG','TOTAL_PRECIPITATION_FLAG','TOTAL_RAIN','TOTAL_RAIN_FLAG',
                      'TOTAL_SNOW','TOTAL_SNOW_FLAG','SNOW_ON_GROUND','SNOW_ON_GROUND_FLAG',
                      'DIRECTION_MAX_GUST_FLAG','SPEED_MAX_GUST_FLAG','COOLING_DEGREE_DAYS_FLAG',
                      'HEATING_DEGREE_DAYS_FLAG','MIN_REL_HUMIDITY_FLAG','MAX_REL_HUMIDITY_FLAG','x','y',
                      'Unnamed: 0','Unnamed: 0.1'],axis=1)
print(dft.columns) 
sns.heatmap(dft.isnull(),yticklabels=False,cbar=False,cmap='Reds_r')
  
#nodata
dft=dft.replace(np.nan,-9999)
dft=dft.reindex(np.random.permutation(dft.index))
 
# outliers
z = np.abs(stats.zscore(dft._get_numeric_data()))
dft= dft[(z < 3).all(axis=1)]

#dft.to_csv('training_noscale.csv')