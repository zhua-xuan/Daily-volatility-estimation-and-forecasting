import pandas as pd
import holidays


def data_read(file_path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df = df[df['SYM_ROOT'] == 'CSCO']
    df = df[['DATE', 'TIME_M', 'PRICE', 'EX','TR_SCOND']]
    return df

def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:    
    return data.drop_duplicates()

def remove_out_of_range_values(data: pd.DataFrame) -> pd.DataFrame:    
    return data[(data['PRICE'] > 0)]

def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    required_columns = ['PRICE', 'DATE', 'TIME_M']
    return data.dropna(subset=required_columns)

def filter_hours(data: pd.DataFrame, start_time: str = '09:30:00', end_time: str = '16:00:00') -> pd.DataFrame:    
    data['DATETIME'] = pd.to_datetime(data['DATE'] + ' ' + data['TIME_M'])
    data = data.set_index('DATETIME')
    filtered_data = data.between_time(start_time, end_time).reset_index(drop = True)
    return filtered_data

def filter_corrected_trades(data: pd.DataFrame) -> pd.DataFrame:
    return data[data['TR_CORR'] == 0]

def filter_sale_conditions(data: pd.DataFrame) -> pd.DataFrame:
    valid_conditions = ['E', 'F']
    invalid_conditions = ['T', 'Z']
    data = data[data['TR_SCOND'].str.contains('|'.join(valid_conditions), na=False)]
    data = data[~data['TR_SCOND'].str.contains('|'.join(invalid_conditions), na=False)]
    return data

def handle_duplicate_timestamps(data: pd.DataFrame) -> pd.DataFrame:
    data['DATETIME'] = pd.to_datetime(data['DATE'] + ' ' + data['TIME_M'])
    return data.groupby('DATETIME', as_index=False).agg({
        'PRICE': 'median',
        'EX': 'first',
        'TR_SCOND': 'first',
        'DATE': 'first'
    })

def filter_most_frequent_exchange(data: pd.DataFrame, frequency: str = 'M'):
    data['DATE'] = pd.to_datetime(data['DATE'])
    data['period'] = data['DATE'].dt.to_period(frequency)
    exchange_counts = data.groupby(['period', 'EX']).size().reset_index(name='count')
    most_frequent_exchanges = exchange_counts.loc[exchange_counts.groupby('period')['count'].idxmax()]
    data = pd.merge(data, most_frequent_exchanges, on='period', suffixes=('', '_y'))
    data = data[data['EX'] == data['EX_y']].drop(columns=['EX_y', 'count', 'period'])
    #exchange_list = most_frequent_exchanges[['period', 'EX']].values.tolist()
    return data

def data_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    cleaned_data = data.copy()
    cleaned_data = filter_hours(cleaned_data)
    cleaned_data = remove_duplicates(cleaned_data)
    cleaned_data = handle_missing_values(cleaned_data)
    cleaned_data = remove_out_of_range_values(cleaned_data)
    cleaned_data = filter_sale_conditions(cleaned_data)
    cleaned_data = handle_duplicate_timestamps(cleaned_data)
    cleaned_data = filter_most_frequent_exchange(cleaned_data)
    return cleaned_data

def data_concat(file_paths: list) -> pd.DataFrame:
    dfs = [pd.read_csv(file_path) for file_path in file_paths]
    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df

def resample(data: pd.DataFrame) -> pd.DataFrame:
    data.set_index('DATETIME', inplace=True)
    data.index = pd.to_datetime(data.index)
    resampled_data = data['PRICE'].resample('1s').mean().reset_index()
    #resampled_data = resampled_data.sort_values(by='DATETIME').reset_index(drop=True)
    return resampled_data

def filter_hour_after(data: pd.DataFrame, start_time: str = '09:30:00', end_time: str = '16:00:00') -> pd.DataFrame:    
    data.set_index('DATETIME', inplace=True)
    filtered_data = data.between_time(start_time, end_time)
    return filtered_data

def filter_date(df):
    df = pd.read_csv('data/new/1s_data_in.csv') 
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])

    df = df[~df['DATETIME'].dt.weekday.isin([5, 6])]

    us_holidays = holidays.US()
    df = df[~df['DATETIME'].dt.date.isin(us_holidays)]