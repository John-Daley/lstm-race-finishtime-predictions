import pandas as pd; 
import numpy as np;
import keras;
import tensorflow as tf;
from keras.preprocessing.sequence import TimeseriesGenerator;
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense

filename = "distance-sorted-race-data.csv"
downloadedFile =pd.read_csv(filename)
print(downloadedFile.info())
# distanceTraveled = downloadedFile["Distance-left"]
# #print(distanceTraveled)
# timeTraveled = downloadedFile["time"]
# #print(timeTraveled)
# totalDistanceTraveled = 0;
# timeIntervals = 0;
# currentSpeed= 0;
# timeUntilFinish =0;
# currentSpeedForPlotting = []
# for i in range(0, len(timeTraveled)-1):
#     totalDistanceTraveled = distanceTraveled[i+1] - distanceTraveled[i]
#    # print(totalDistanceTraveled)
#     timeIntervals = timeTraveled[i] - timeTraveled[i+1]
#   #  print(timeIntervals)
#     currentSpeed = totalDistanceTraveled/timeIntervals
#     print(str(currentSpeed) + " m/s")
#     timeUntilFinish = distanceTraveled[i]/currentSpeed
#     print(str(timeUntilFinish/60) + " Minutes until racer is finished")
#     currentSpeedForPlotting.append(currentSpeed)
# for times in range(len(timeTraveled)):
#   totalDistanceTraveled = distanceTraveled[times+1] - distanceTraveled[times]
#   print(totalDistanceTraveled + "km")


# # fig_size = plt.rcParams["figure.figsize"]
# # fig_size[0] = 15
# # fig_size[1] = 5
# # plt.rcParams["figure.figsize"] = fig_size

plt.title('Distance vs Time')
plt.ylabel('meters per second')
plt.xlabel('Distance per km')
plt.grid(True)

plt.autoscale(axis='x',tight=True)
plt.plot(downloadedFile["505"])
plt.plot(downloadedFile["506"])
plt.plot(downloadedFile["507"])
plt.show()

# distanceLeftData = downloadedFile["Distance-left"].values
# distanceLeftData = distanceLeftData.reshape((-1,1))

# splitPercent = 0.80
# split = int(splitPercent*len(distanceLeftData))

# distanceLeftTrain = distanceLeftData[:split]
# distanceLeftTest = distanceLeftData[split:]

# timeTrain = downloadedFile["time"][:split]
# timeTest = downloadedFile["time"][split:]

# print(str(len(distanceLeftTrain)) + " distance train")
# print(str(len(distanceLeftTest))+ " distance test")


# look_back = 4

# train_generator = TimeseriesGenerator(distanceLeftTrain, distanceLeftTrain, length=look_back, batch_size=20)
# test_generator = TimeseriesGenerator(distanceLeftTest,distanceLeftTest, length=look_back, batch_size=1)


# model = Sequential()
# model.add(
#     LSTM(10,activation="relu",input_shape =(look_back,1)
#     )
# )
# model.add(Dense(1))
# model.compile(optimizer="adam", loss="mse")

# numberOfEpocs = 30
# model.fit_generator(train_generator, epochs=numberOfEpocs,verbose=1)