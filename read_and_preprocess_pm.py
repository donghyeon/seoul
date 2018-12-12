import numpy as np
import pandas as pd
import os


def read_pm_dataset(airkorea_data_dir):
    # Raw data filenames
    # data_dir = '/home/donghyeon/disk1/dataset/seoul/airkorea'
    data_dir = airkorea_data_dir
    data_filenames = ['2014년 1분기.csv', '2014년 2분기.csv', '2014년 3분기.csv', '2014년 4분기.csv',
                      '2015년1분기.csv', '2015년2분기.csv', '2015년3분기.csv', '2015년4분기.csv',
                      '2016년 1분기.csv', '2016년 2분기.csv', '2016년 3분기.csv', '2016년 4분기.csv',
                      '2017년 1분기.xlsx', '2017년 2분기.xlsx', '2017년 3분기.xlsx', '2017년 4분기.xlsx']

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
    
    return df_pm


def _get_station_codes_measuring_pm25(df_pm):
    station_codes = df_pm.index.get_level_values('측정소코드').unique()

    station_codes_pm25 = []
    for code in station_codes:
        df = df_pm.xs(code, level='측정소코드')
        if not pd.isna(df['PM25']).all():
            station_codes_pm25.append(code)
    station_codes_pm25 = pd.Series(station_codes_pm25, name='측정소코드_PM25')

    return station_codes_pm25


def get_dataframe_with_complete_pm25(df_pm):
    station_codes = df_pm.index.get_level_values('측정소코드').unique()
    station_codes_pm25 = _get_station_codes_measuring_pm25(df_pm)
    station_codes_to_drop = station_codes.drop(station_codes_pm25)

    time_stamps = df_pm.index.get_level_values('측정일시').unique()
    time_stamps_mask_2014 = (time_stamps / 1000000).astype(int) == 2014
    time_stamps_to_drop = time_stamps[time_stamps_mask_2014]

    df_pm25 = df_pm.drop(labels=station_codes_to_drop, level='측정소코드').drop(
        labels=time_stamps_to_drop, level='측정일시')

    return df_pm25


class TargetPM(object):
    def __init__(self, keys, hours):
        """
        :param keys: list of PM_keys to predict, ex) [TargetPM.PM10, TargetPM.PM25]
        :param hours: list of hours (ints) to predict, ex) [3, 6, 12, 24]
        """
        self._keys = keys
        if not isinstance(keys, list):
            self._keys = [keys]

        self._hours = hours
        if not isinstance(hours, list):
            self._hours = [hours]

    PM10 = 'PM10'
    PM25 = 'PM25'
    _column_name_definition = '%s_%03d'

    @property
    def keys(self):
        return self._keys

    @property
    def hours(self):
        return self._hours

    @property
    def label_columns(self):
        return self.get_all_label_column_names()

    def get_label_column_name(self, key, hour):
        return self._column_name_definition % (key, hour)

    @staticmethod
    def get_key_hour_from_column_name(column_name):
        key, hour = column_name.split('_')
        return key, int(hour)

    def get_label_column_names_by_key(self, key):
        return [self.get_label_column_name(key, hour)
                for hour in self._hours]

    def get_label_column_names_by_hour(self, hour):
        return [self.get_label_column_name(key, hour)
                for key in self._keys]

    def get_all_label_column_names(self):
        return [self.get_label_column_name(key, hour)
                for key in self._keys
                for hour in self._hours]

    def to_dict(self):
        return {'keys': self._keys, 'hours': self._hours}


def make_target_values(df_pm, target_pm):
    """
    :param df_pm:
    :param target_pm: an instance of TargetPM
    :return: df_pm
    """
    for label_column in target_pm.label_columns:
        df_pm[label_column] = np.nan

    # TODO: improve readability
    station_codes = df_pm.index.get_level_values('측정소코드').unique()
    for key in target_pm.keys:
        for code in station_codes:
            target_series = df_pm.xs(code, level='측정소코드')[key]

            for hour in target_pm.hours:
                new_target = target_series.iloc[hour:]
                new_target.set_axis(target_series.index[:-hour], inplace=True)
                new_target = new_target.reindex(target_series.index)

                idx_slicer = pd.IndexSlice
                label_column = target_pm.get_label_column_name(key, hour)
                df_pm.loc[idx_slicer[code, :], label_column] = new_target.values

    df_pm = convert_dtype_for_numeric_columns(df_pm, np.float32)
    return df_pm


def preprocess_pm(df_pm, target_pm):
    # Let's preprocess data
    # Drop non-numerical columns
    exclude_keys = ['지역', '측정소명', '주소']
    df_pm = df_pm.drop(columns=exclude_keys)

    # Translate all Hangul columns into alphabet columns to avoid an regexp error
    df_pm = df_pm.rename(columns={'위도': 'Latitude', '경도': 'Longitude'})

    # Get label_columns and target_max_hour from target_pm
    label_columns = target_pm.label_columns
    target_max_hour = max(target_pm.hours)

    # Drop non-accessible future labels (NaNs)
    date_index = df_pm.index.get_level_values('측정일시').unique()
    df_pm = df_pm.drop(date_index[-target_max_hour:], level='측정일시')

    # Split the data into features and labels
    df_features = df_pm.drop(columns=label_columns)
    df_labels = df_pm[label_columns]

    return df_pm, df_features, df_labels


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


def treat_nan_by_fill_methods(df_pm, forward_fill=True, backward_fill=True):
    # forward fill first, and backward fill next
    if forward_fill:
        df_pm = df_pm.ffill()
    if backward_fill:
        df_pm = df_pm.bfill()
    return df_pm


def get_statistics_for_standardization(dataset):
    """
    :param dataset: pandas DataFrame, training dataset
    :return: pandas Series tuple (mean, stddev)
    """
    return dataset.mean(), dataset.std()


def convert_dtype_for_numeric_columns(dataframe, dtype):
    numeric_columns_to_dtype = {}
    for column_name in dataframe.select_dtypes(include=np.number).columns:
        numeric_columns_to_dtype[column_name] = dtype
    return dataframe.astype(numeric_columns_to_dtype)


def split_data_to_train_eval(df_features, df_labels):
    idx = pd.IndexSlice
    dates_train = idx[:2016123124]
    dates_eval = idx[2017010101:]

    df_features_train = df_features.xs(dates_train, level='측정일시', drop_level=False)
    df_features_eval = df_features.xs(dates_eval, level='측정일시', drop_level=False)
    df_labels_train = df_labels.xs(dates_train, level='측정일시', drop_level=False)
    df_labels_eval = df_labels.xs(dates_eval, level='측정일시', drop_level=False)

    return df_features_train, df_labels_train, df_features_eval, df_labels_eval
