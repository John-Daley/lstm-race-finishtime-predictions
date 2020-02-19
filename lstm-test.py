import pandas as pd; 
import numpy as np;
import keras;
import tensorflow as tf;
from keras.preprocessing.sequence import TimeseriesGenerator;
import matplotlib.pyplot as plt


filename = "formated-race-data-for-one.csv"
downloadedFile =pd.read_csv(filename)
print(downloadedFile.info())





fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 5
plt.rcParams["figure.figsize"] = fig_size

plt.title('Distance vs Time')
plt.ylabel('time')
plt.xlabel('distance')
plt.grid(True)
plt.autoscale(axis='x',tight=True)
plt.plot(downloadedFile.info())