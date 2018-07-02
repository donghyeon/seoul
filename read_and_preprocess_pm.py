import pandas as pd
import numpy as np
import os


def read_pm_dataset(airkorea_data_dir):
    # Raw data filenames
    # data_dir = '/home/donghyeon/disk1/dataset/seoul/airkorea'
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
    df_pm.sort_index(level=['측정소코드', '측정일시'], inplace=True)
    print('Finished!')
    
    return df_pm


def make_target_values(df_pm, target_dict):
    """
    :param df_pm:
    :param target_dict: {'target_column': [list of target_hours(ints)], ...}
    :return: df_pm
    """
    for target_column in target_dict:
        for hour in target_dict[target_column]:
            new_column = '%s_%dh' % (target_column, hour)
            df_pm[new_column] = np.nan

    loc_codes = df_pm.index.get_level_values('측정소코드').unique()
    for target_column in target_dict:
        for loc_code in loc_codes:
            target_series = df_pm.xs(loc_code, level='측정소코드')[target_column]

            for hour in target_dict[target_column]:
                new_target = target_series.iloc[hour:]
                new_target.set_axis(target_series.index[:-hour], inplace=True)
                new_target = new_target.reindex(target_series.index)

                idx_slicer = pd.IndexSlice
                new_column = '%s_%dh' % (target_column, hour)
                df_pm.loc[idx_slicer[loc_code, :], new_column] = new_target.values

    return df_pm


def preprocess_pm(df_pm, target_dict):
    # Let's preprocess data
    # Drop non-numerical columns
    exclude_keys = ['지역', '측정소명', '주소']
    df_pm = df_pm.drop(columns=exclude_keys)

    # Get label_keys and target_max_hour from target_dict
    label_keys = []
    target_max_hour = 0
    for target_column in target_dict:
        for hour in target_dict[target_column]:
            target_key = '%s_%dh' % (target_column, hour)
            label_keys.append(target_key)
            target_max_hour = max(hour, target_max_hour)

    # Drop non accessible future labels (NaNs)
    date_index = df_pm.index.get_level_values('측정일시').unique()
    df_pm = df_pm.drop(date_index[-target_max_hour:], level='측정일시')

    # Split the data into features and labels
    df_features = df_pm.drop(columns=label_keys)
    df_labels = df_pm[label_keys]

    # Make batches by code
    num_locations = df_pm.index.get_level_values('측정소코드').unique().size
    features_dim = df_features.columns.size

    # Standardize PM features data
    pm_mean = df_features.mean()
    pm_std = df_features.std()
    pm_features = df_features.values.reshape(num_locations, -1, features_dim)

    pm_features = (pm_features - pm_mean.values) / pm_std.values

    # Standardize PM labels data
    label_keys = []
    for target_column in target_dict:
        for hour in target_dict[target_column]:
            label_keys.append(target_column)

    labels_dim = len(label_keys)

    labels_mean = pm_mean[label_keys]
    labels_std = pm_std[label_keys]
    pm_labels = df_labels.values.reshape(num_locations, -1, labels_dim)

    pm_labels = (pm_labels - labels_mean.values) / labels_std.values

    return pm_features, pm_labels, df_pm



def treat_nan_by_mask(pm_tensor):
    # Treatment for missing data
    nan_mask = np.isnan(pm_tensor)
    pm_tensor[nan_mask] = 0  # Set all missing data to 0
    columns_have_missing = nan_mask.any(axis=(0, 1))
    missing_tensor = nan_mask[:, :, columns_have_missing].astype('float')  # Treat missing data as a feature
    # Concatenate the original features and the missing data feature
    pm_tensor = np.concatenate((pm_tensor, missing_tensor), axis=2)
    
    return pm_tensor


def treat_nan_by_interpolation(pm_tensor, df_pm):
    # TODO: interpolate values only for accidentally missing data (not for non-accessible data)
    pm_tensor = np.nan_to_num(pm_tensor)
    return pm_tensor