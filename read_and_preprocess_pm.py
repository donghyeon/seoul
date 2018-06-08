
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os


# In[2]:


def read_pm_dataset(airkorea_data_dir):
    # Raw data filenames
    #data_dir = '/home/donghyeon/disk1/dataset/seoul/airkorea'
    data_dir = airkorea_data_dir
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
        file_path = os.path.join(data_dir, data_filename)
        if file_ext == '.csv':
            try:
                df = pd.read_csv(file_path, encoding='cp949')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='utf-8')
        elif file_ext == '.xlsx':
            df = pd.read_excel(file_path)
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
    df_pm.set_index(['측정소코드', '측정일시'], inplace=True)
    df_pm.sort_index(level=['측정소코드', '측정일시'])
    print('Finished!')
    
    return df_pm


# In[3]:


def preprocess_pm(df_pm):
    # Let's preprocess data
    # Drop non-numerical columns
    exclude_keys = ['지역', '측정소명', '주소']
    df_pm.drop(columns=exclude_keys, inplace=True)

    # Make batches by code
    num_locations = df_pm.index.get_level_values('측정소코드').unique().size
    features_dim = df_pm.columns.size

    # Standardize PM data
    pm_mean = df_pm.mean()
    pm_std = df_pm.std()
    pm_tensor = df_pm.values.reshape(num_locations, -1, features_dim)

    pm_tensor = (pm_tensor - pm_mean.values) / pm_std.values
    
    return pm_tensor


# In[4]:


def treat_nan_with_mask(pm_tensor):
    # Treatment for missing data
    nan_mask = np.isnan(pm_tensor)
    pm_tensor[nan_mask] = 0 # Set all missing data to 0
    columns_have_missing = nan_mask.any(axis=(0, 1))
    missing_tensor = nan_mask[:, :, columns_have_missing].astype('float') # Treat missing data as a feature
    pm_tensor = np.concatenate((pm_tensor, missing_tensor), axis=2) # Concatenate the original features and the missing data feature
    
    return pm_tensor

