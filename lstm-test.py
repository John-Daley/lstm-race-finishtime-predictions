import pandas as pd; 
import numpy as np;
import keras;
import tensorflow as tf;
from keras.preprocessing.sequence import TimeseriesGenerator;
import matplotlib.pyplot as plt


filename = "formated-race-data-for-one.csv"
downloadedFile =pd.read_csv(filename)
print(downloadedFile.info())
distanceTraveled = downloadedFile["Distance-left"]
#print(distanceTraveled)
timeTraveled = downloadedFile["time"]
#print(timeTraveled)
totalDistanceTraveled = 0;
timeIntervals = 0;
currentSpeed= 0;
timeUntilFinish =0;

for i in range(0, len(timeTraveled)-1):
    totalDistanceTraveled = distanceTraveled[i+1] - distanceTraveled[i]
    print(totalDistanceTraveled)
    timeIntervals = timeTraveled[i] - timeTraveled[i+1]
    print(timeIntervals)
    currentSpeed = totalDistanceTraveled/timeIntervals
    print(str(currentSpeed) + " m/s")
    timeUntilFinish = distanceTraveled[i]/currentSpeed
    print(str(timeUntilFinish/60) + " Minutes until racer is finished")

# for times in range(len(timeTraveled)):
#   totalDistanceTraveled = distanceTraveled[times+1] - distanceTraveled[times]
#   print(totalDistanceTraveled + "km")


# fig_size = plt.rcParams["figure.figsize"]
# fig_size[0] = 15
# fig_size[1] = 5
# plt.rcParams["figure.figsize"] = fig_size

plt.title('Distance vs Time')
plt.ylabel('time')
plt.xlabel('distance')
plt.grid(True)
#plt.autoscale(axis='x',tight=True)
plt.plot(downloadedFile["time"])
plt.plot(downloadedFile["Distance-left"])
plt.show()