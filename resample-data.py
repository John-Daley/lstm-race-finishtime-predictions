# resample minute data to total for each day
from pandas import read_csv
# load the new file
dataset = read_csv('date-sorted-race-data.csv',dtype={"datetime": float})
# resample data to daily
dataset.info()
#dataset = dataset.astype('float64')
daily_groups = dataset.resample('D')
daily_data = daily_groups.sum()
# summarize
print(daily_data.shape)
print(daily_data.head())
# save
daily_data.to_csv('resampled-race-data.csv')