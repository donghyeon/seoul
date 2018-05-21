
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os


# In[ ]:


# Raw data filenames
data_filenames = ['2014년 1분기.csv', '2014년 2분기.csv', '2014년 3분기.csv', '2014년 4분기.csv',
                  '2015년1분기.csv', '2015년2분기.csv', '2015년3분기.csv', '2015년4분기.csv',
                  '2016년 1분기.csv', '2016년 2분기.csv', '2016년 3분기.csv', '2016년 4분기.csv',
                  '2017년 1분기.xlsx', '2017년 2분기.xlsx', '2017년 3분기.xlsx']

# Read geolocations
df_geo = pd.read_csv('geo.csv')

# Read PM data
dfs = []
for data_filename in data_filenames:
    file_ext = os.path.splitext(data_filename)[1]
    if file_ext == '.csv':
        try:
            df = pd.read_csv(data_filename, encoding='cp949')
        except UnicodeDecodeError:
            df = pd.read_csv(data_filename, encoding='utf-8')
    elif file_ext == '.xlsx':
        df = pd.read_excel(data_filename)
    else:
        raise Exception('Cannot read %s.' % data_filename)
    print('Reading %s' % data_filename)
    
    # Get seoul data only
    seoul_index = df['지역'].str.contains('서울')
    df = df[seoul_index]
    
    # Merge PM and Geolocation info
    df = pd.merge(df, df_geo[['측정소코드', '위도', '경도']], on='측정소코드')
    
    dfs.append(df)

# Concatenate all data
df_pm = pd.concat(dfs, ignore_index=True)
df_pm.set_index('측정일시', inplace=True)
print('Finished!')


# In[ ]:


# Let's preprocess data
# Drop non-numerical columns
exclude_keys = ['지역', '측정소코드', '측정소명', '주소']
df_pm.drop(columns=exclude_keys, inplace=True)

# Standardize PM data
pm_mean = df_pm.mean()
pm_std = df_pm.std()
pm_tensor = (df_pm.values - pm_mean.values) / pm_std.values


# In[ ]:


# Treatment for missing data
nan_mask = np.isnan(pm_tensor)
pm_tensor[nan_mask] = 0 # Set all missing data to 0
columns_have_missing = nan_mask.any(axis=0)
missing_tensor = nan_mask[:, columns_have_missing].astype('float') # Treat missing data as a feature
pm_tensor = np.concatenate((pm_tensor, missing_tensor), axis=1)

