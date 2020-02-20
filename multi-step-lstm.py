# following the how to develop multi-step lstm from https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
# The following code is based on the encoder-decoder model presented there with a custom dataset 

from math import sqrt
from numpy import split
from numpy import array
from numpy import split
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

# here I will attempt to divide the training and test sets
