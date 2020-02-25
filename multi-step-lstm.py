# following the how to develop multi-step lstm from https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
# The following code is based on the encoder-decoder model presented there with a custom dataset 

# univariate multi-step encoder-decoder lstm
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed


#split the data set into train/test sets
# # # here I will attempt to divide the training and test sets
def split_dataset(data):
    train, test = data[1:40], data[41:57] #split data 80 train and 20 test
    #split sets into races, I think this will need to be done in the future
    train = array(split(train, len(train)))
    test = array(split(test, len(test)))
    return train,test
   
   # this method will be sued to evaluate the prediction times for racers 
def evaluate_forecasts(actual, predicted):
    scores = list()
    #calculate the RMSE score for each day
    for i in range(actual.shape[1]):
        #calc mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        #calc rmse
        rmse = sqrt(mse)
        #store the scores
        scores.append(rmse)
    #calculate the overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col])**2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score,scores

#summarize the above scores
def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores]) 
    print('%s: [%.3f] %s' % (name, score, s_scores))

#below the code will convert the history into inputs and outputs
#n_out was used to denoted the 7 days a week in the example. 
def to_supervised(train,n_input, n_out=1): 
    #flatten the data
    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    print("shape 0 " + str(train.shape[0]))
    print("shape 1 " + str(train.shape[1]))
    print("shape 2 " + str(train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step through the entire history ( prevous outputs? ) one time step at a time
    for _ in range(len(data)):
        in_end = in_start + n_input
        out_end = in_end + n_out
        #this will check if we have enough data for this instance
        if out_end <= len(data):
            x_input = data[in_start:in_end, 0]
            x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y.append(data[in_end:out_end, 0])
        in_start += 1
    return array(X), array(y)

# the following method will be used to train the model
def build_model(train, n_input):
    #data preperation 
    train_x, train_y = to_supervised(train, n_input)
    #define the parameters such as verbose, epocs, batch_size 
    verbose, epochs, batch_size = 0, 20, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    #reshape the output into the required format [samples,timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    
    ## ## below will be TimeDistributed wrapper to allow for th fully connected layer and the output later to be used to process each step
   # ## The model will be defined here under - the first number is the the hidden layer of neurons(units) 
# ## it will use the Sequential library from keras, the activation will be relu, and the input shape will consist of the 
# # the number of steps and the number of features. 
# #decoder model
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))

#sequence of vectors to be preseneted to the decoder
 ## this is currently set for week days, I will need to think about how the race is set up to understand how to set this.
    model.add(RepeatVector(n_outputs))
## encoder model below 
    model.add(LSTM(200, activation='relu', return_sequences=True))
## below will be TimeDistributed wrapper to allow for th fully connected layer and the output later to be used to process each step

    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')
    #fit the network
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model
    
# make a forecast for a runner to finish
def forecast(model, history, n_input):
    #flatten the data
    data = array(history)
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    # retrieve last ovservations for the input data
    input_x = data[-n_input:, 0]
    #reshape to our 3 d array format [1, n_input,1]
    input_x = input_x.reshape((1, len(input_x), 1))
    #forcast next week
    yhat = model.predict(input_x, verbose=0)
    return yhat
    
#evaluate a single model 
def evaluate_model(train, test, n_input):
    #fit model
    model = build_model(train, n_input)
    #history is a list of the race data
    history = [x for x in train]
    # walk-forward aka week to week validation over each week
    predictions = list()
    for i in range(len(test)):
        #predict the week in his example i just want the race
        yhat_sequence = forecast(model, history, n_input)
        #store predictions
        predictions.append(yhat_sequence)
        #get the real overvation and add to the history for predicting the finish time
        history.append(test[i, :])
    #evaluate preictions for races
    predictions = array(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores
#load dataset
dataset = read_csv("runner-sorted-data.csv")
#print(dataset)
train, test = split_dataset(dataset.values)
#print("training " + str(train) + " training set")
#print("testing " + str(test) + " testing set")
#evaluate the model and get scores
n_input = 14
score, scores = evaluate_model(train, test, n_input)
#summarize scores
summarize_scores('lstm', score, scores)

#plot scores
days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
pyplot.plot(days, scores, marker='o', label='lstm')
pyplot.show()


print("shape 0 " + str(train.shape[0]))
print("shape 1 " + str(train.shape[1]))
print("shape 2 " + str(train.shape[2]))



