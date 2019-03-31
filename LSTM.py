import matplotlib.pyplot as plt
import keras
import math


start = datetime(2018, 12, 30)
end = datetime(2014, 12, 30)
df = get_historical_data("HMSY", end, start, output_format='pandas')
#print (df)
#df.plot()
#plt.show()
#df.to_csv("dataFrame.csv")
print (df)
plt.figure(figsize = (21,10))
plt.plot(range(df.shape[0]),(df['low']+df['high'])/2.0)
#plt.show()
highs = df['high'].tolist()
lows = df['low'].tolist()
averages = list()
for i in range(len(highs)):
	averages.append((highs[i] + lows[i])/2)
print (len(averages))
scaler = MinMaxScaler(feature_range=(0, 1))
averages = np.array(averages).reshape(-1,1)
averages = scaler.fit_transform(averages)
training = averages[:950]
testing = averages[950:]
trainingx = list()
trainingy = list()
for i in range(len(training)-26):
	trainingx.append(training[i:(i+25),0])
	trainingy.append(training[i+25,0])
testingx = list()
testingy = list()
for i in range(len(testing)-26):
	testingx.append(testing[i:(i+25),0])
	testingy.append(testing[i+25,0])
trainingx = np.array(trainingx)
trainingy = np.array(trainingy)
testingx = np.array(testingx)
testingy = np.array(testingy)
#print (testx)
#print(testy)
#print(trainingx[0])
#trainingx = np.array(trainingx).reshape(-1,1)
#testingx = np.array(testingx).reshape(-1,1)
#print(trainingx[0])
trainingx = np.reshape(trainingx, (trainingx.shape[0], 1, trainingx.shape[1]))
testingx = np.reshape(testingx, (testingx.shape[0], 1, testingx.shape[1]))
model = keras.models.Sequential()
#print(trainingx[0])
model.add(keras.layers.LSTM(15, input_shape=(1, 25)))
model.add(keras.layers.Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(trainingx, trainingy, epochs=50, batch_size=50, verbose=1)
trainingGuess = model.predict(trainingx)
#What's going on here?
trainingGuess = scaler.inverse_transform(trainingGuess)
trainingy = scaler.inverse_transform([trainingy])
trainingError = math.sqrt(mean_squared_error(trainingy[0], trainingGuess[:,0]))
#print (trainingError)
testingGuess = model.predict(testingx)
testingGuess = scaler.inverse_transform(testingGuess)
testingError = math.sqrt(mean_squared_error(testingy[0],testingGuess[:,0]))




