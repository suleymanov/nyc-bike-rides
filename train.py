import os
from datetime import datetime, timedelta

import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import root_mean_squared_error


def read_and_format_data():

    pn_data = 'sample_data'
    df = pd.concat([
        pd.read_csv(pn_data + os.sep + fn) 
        for fn in filter(lambda x: x.endswith('csv'), os.listdir(pn_data))
    ])

    df = df[['ride_id', 'started_at', 'ended_at', 'start_station_name', 'end_station_name']]
    df['started_at'] = pd.to_datetime(df['started_at'])
    df['ended_at'] = pd.to_datetime(df['ended_at'])
    df['month'] = df['started_at'].apply(lambda x: x.month)
    df['hour'] = df['started_at'].apply(lambda x: x.hour)
    df['weekday'] = df['started_at'].apply(lambda x: x.weekday())

    all_stations = list(df['end_station_name'].dropna().unique())

    return df, all_stations


def create_dataset(df, all_stations):

    df['time'] = df['ended_at'].dt.floor('30min')
    counts = (
        df.dropna(subset=['end_station_name'])
          .groupby(['time', 'end_station_name'])
          .size()
          .rename('count')
          .reset_index()
    )
    times = pd.date_range(df['time'].min(), df['time'].max(), freq='30min')

    grid = (
        pd.MultiIndex.from_product(
            [times, all_stations],
            names=['time', 'end_station_name']
        )
        .to_frame(index=False)
        .merge(counts, on=['time', 'end_station_name'], how='left')
    )
    grid['count'] = grid['count'].fillna(0)
    grid = grid.sort_values(['end_station_name', 'time'])
    g = grid.groupby('end_station_name')['count']

    # Previous periods only — no leakage
    grid['count_30'] = g.transform(lambda x: x.shift(1))
    grid['count_60'] = g.transform(lambda x: x.shift(1).rolling(2).sum())
    grid['count_120'] = g.transform(lambda x: x.shift(1).rolling(4).sum())

    # Calendar features
    grid['month'] = grid['time'].dt.month
    grid['weekday'] = grid['time'].dt.weekday
    grid['hour'] = grid['time'].dt.hour
    grid['hour_part'] = (grid['time'].dt.minute == 30).astype(int)

    # Next 30-minute arrivals
    grid['y'] = g.transform(lambda x: x.shift(-1))
    grid = grid.fillna(0)
    dataset = grid.copy()
    dataset = dataset.sort_values(by='time')

    # pick up last 30 days: 20 for train, 10 for validation
    dt_end = dataset['time'].max()
    dt_start = dt_end - timedelta(days=30)
    dataset = dataset[(dt_start <= dataset['time']) & (dataset['time'] <= dt_end)]

    dt_split = dt_end - timedelta(days=10)
    data_train = dataset[dataset['time'] < dt_split]
    data_val = dataset[dataset['time'] >= dt_split]
    data_train = data_train.drop('time', axis=1)
    data_val = data_val.drop('time', axis=1)

    feature_columns = [
        'end_station_name', 
        'count_30', 'count_60', 'count_120', 
        'month', 'weekday', 'hour', 'hour_part'
    ]

    return data_train, data_val, feature_columns


def train_once(X_train, X_val, y_train, y_val, cat_features, use_model='xgboost'):

    if use_model == 'xgboost':
        model = XGBRegressor(
            n_estimators=500, max_depth=7, learning_rate=0.05, 
            enable_categorical=True, tree_method='hist'
        )
        model.fit(X_train, y_train)

    elif use_model == 'catboost':
        model = CatBoostRegressor()
        model.fit(X_train, y_train, cat_features=cat_features, verbose=False)

    elif use_model == 'lightgbm':
        model = LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=7, num_leaves=31, verbose=-1
        )
        model.fit(X_train, y_train, categorical_feature=cat_features)

    else:
        raise ValueError(f'Model not supported: {use_model}')

    yhat_train = model.predict(X_train)
    yhat_val = model.predict(X_val)
    rmse_train = root_mean_squared_error(y_train, yhat_train)
    rmse_val = root_mean_squared_error(y_val, yhat_val)
    
    if use_model == 'lightgbm':
        model.booster_.save_model(f'model_{use_model}')
    else:
        model.save_model(f'model_{use_model}')

    print(f'\n\tModel: {use_model}')
    print(f'\tRMSE (train): {rmse_train}')
    print(f'\tRMSE (val): {rmse_val}')


def main():
    df, all_stations = read_and_format_data()
    print('\n1) Data read complete')
    
    data_train, data_val, feature_columns = create_dataset(df, all_stations)
    print('\n2) Featurization complete')
    cat_features = ['end_station_name']

    X_train = data_train[feature_columns]
    X_val = data_val[feature_columns]
    X_train['end_station_name'] = X_train['end_station_name'].astype('category')
    X_val['end_station_name'] = X_val['end_station_name'].astype('category')
    y_train = data_train['y']
    y_val = data_val['y']

    train_once(X_train, X_val, y_train, y_val, cat_features, 'lightgbm')
    print('\n3) Model trained and saved')


if __name__ == '__main__':
    main()
