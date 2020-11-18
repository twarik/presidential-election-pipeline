import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def preprocess():
    '''
    This function/component will:
    1. Load the Road_Safety dataset
    2. Clean and transform the dataset
    3. Split the dataset into train and test set
    4. Use np.save to save our dataset to disk so that it can be reused by later components
    '''
    main = pd.read_csv("https://raw.githubusercontent.com/sophiabj/stage-f-01-road-safety/master/data/dftRoadSafety_Accidents_2016.csv")
    #main = pd.read_csv("dftRoadSafety_Accidents_2016.csv")
    pd.set_option("display.max_columns", 32)
    main.fillna(method='ffill',inplace=True)
    main.to_csv('Time_series', index=False)
    Time_series = pd.read_csv('Time_series', infer_datetime_format=True, parse_dates={'datetime':[9,11]},
                                index_col=['datetime'], header = 0,)
    Time_series_2 = Time_series[['Accident_Index','Number_of_Casualties']]
    Time_series_2.isna().sum()
    Time_series_2.index = pd.to_datetime(Time_series.index)
    df_daily = Time_series_2.resample('D').mean()

    df_daily = df_daily.reset_index()

    df_daily = df_daily.rename(columns={"datetime": "ds", "Number_of_Casualties": 'y'})

    train = df_daily[(df_daily["ds"] > '2016-01-01') & (df_daily["ds"] <= '2016-12-01')]

    test = df_daily[(df_daily["ds"] > '2016-12-01')]

    np.save('train.npy', train)
    np.save('test.npy', test)

    print("Preprocessing Done")


if __name__ == '__main__':
    print('Preprocessing Road Safety data...')
    print(' ')
    preprocess()
