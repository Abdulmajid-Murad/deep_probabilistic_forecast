import numpy as np
import pandas as pd


def data_loader(data_dir, task, historical_sequence_length=24, forecast_horizon=24, start_train='2019-01-01', end_train='2019-12-31', start_test='2020-01-01', end_test='2020-02-01'):

    df_aq = pd.read_csv(data_dir+'/air_quality_measurements.csv',index_col='time',  parse_dates=True)
    df_weather= pd.read_csv(data_dir+'/weather.csv', index_col='time', parse_dates=True)
    df_traffic = pd.read_csv(data_dir+'/traffic.csv', index_col='Time', parse_dates=True)
    df_traffic = df_traffic.add_prefix('traffic_')
    df_street_cleaning = pd.read_csv(data_dir+'/street_cleaning.csv', index_col='time', parse_dates=True)
    df_street_cleaning = df_street_cleaning.add_prefix('street_cleaning_')

    columns = df_aq.columns.values.copy()

    if task == 'regression':
        reg_cols = [col for col in columns if ('_class' not in col) and ('_threshold' not in col)]
        y_reg = df_aq[reg_cols].copy()
        # alleviate exponential effects, the target variable is log-transformed as per the Uber paper. add 1 to avi0d log(0)
        # use np.exp(y)-1 to refers.
        y_reg= np.log(y_reg+1)
        y_reg_mean=y_reg.mean()
        y_reg_std=y_reg.std()
        y_reg = (y_reg- y_reg_mean)/y_reg_std
        y = y_reg
        stats = {'y_reg_mean': y_reg_mean.values, 'y_reg_std': y_reg_std.values, 'reg_cols': reg_cols}

    elif task == 'classification':
        thre_cols = [col for col in columns if ('_threshold' in col)]
        y_thre = df_aq[thre_cols].copy()
        y = y_thre
        stats = {'thre_cols': thre_cols}

    df_weather_mean=df_weather.mean()
    df_weather_std=df_weather.std()
    df_weather = (df_weather - df_weather_mean)/df_weather_std

    df_traffic_mean=df_traffic.mean()
    df_traffic_std=df_traffic.std()
    df_traffic = (df_traffic - df_traffic_mean)/df_traffic_std

    start_train = pd.to_datetime(start_train)
    end_train = pd.to_datetime(end_train)

    y_train = y[start_train:end_train].copy()
    train_index = y_train.index
    df_traffic_train, df_weather_train, df_street_cleaning_train = df_traffic[start_train:end_train].copy(), df_weather[start_train:end_train].copy(), df_street_cleaning[start_train:end_train].copy()


    timediff = forecast_horizon*np.ones(shape=(y_train.shape[0],))
    #deal with the first day
    for i , diff in enumerate(timediff[:forecast_horizon+1]):
        timediff[i] = min(diff, i )

    latest_idx = train_index - pd.to_timedelta(timediff, unit='H')
    latest_measurement_train= y_train.loc[latest_idx].copy()
    latest_measurement_train.index = train_index
    latest_measurement_train= latest_measurement_train.add_prefix('latest_')
    X_train = pd.concat([latest_measurement_train, df_traffic_train, df_weather_train, df_street_cleaning_train], axis=1)

    y_train = y_train.values
    X_train = X_train.values

    start_test = pd.to_datetime(start_test) - pd.to_timedelta(historical_sequence_length, unit='H')#to allow for historical_sequence_length
    end_test = pd.to_datetime(end_test)
    y_test = y[start_test:end_test].copy()
    test_index = y_test.index
    df_traffic_test, df_weather_test, df_street_cleaning_test = df_traffic[start_test:end_test].copy(), df_weather[start_test:end_test].copy(), df_street_cleaning[start_test:end_test].copy()


    timediff = forecast_horizon*np.ones(shape=(y_test.shape[0],))
    #deal with the first day
    for i , diff in enumerate(timediff[:forecast_horizon+1]):
        timediff[i] = min(diff, i )

    latest_idx = test_index - pd.to_timedelta(timediff, unit='H')
    latest_measurement_test= y_test.loc[latest_idx].copy()
    latest_measurement_test.index = test_index
    latest_measurement_test= latest_measurement_test.add_prefix('latest_')
    X_test = pd.concat([latest_measurement_test, df_traffic_test, df_weather_test, df_street_cleaning_test], axis=1)

    y_test = y_test.values
    X_test = X_test.values

    stats['start_test'], stats['end_test'], stats['historical_sequence_length'] = start_test, end_test, historical_sequence_length
    stats['X_train_max'] = X_train.max(0)
    stats['X_train_min'] = X_train.min(0)

    return X_train, y_train, X_test, y_test, stats


