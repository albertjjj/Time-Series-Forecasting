import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.sparse import data
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
import time

#data
dataFull = np.array([266.0, 145.9, 183.1, 119.3, 180.3, 168.5, 231.8, 224.5, 192.8, 122.9, 336.5, 185.9, 194.3, 149.5, 210.1, 273.3, 191.4, 287.0, 226.0, 303.6, 289.9, 421.6, 264.5, 342.3, 339.7, 440.4, 315.9, 439.3, 401.3, 437.4, 575.5, 407.6, 682.0, 475.3, 581.3, 646.9])
dates = np.array(['1-01','1-02','1-03','1-04','1-05','1-06','1-07','1-08','1-09','1-10','1-11','1-12','2-01','2-02','2-03','2-04','2-05','2-06','2-07','2-08','2-09','2-10','2-11','2-12','3-01','3-02','3-03','3-04','3-05','3-06','3-07','3-08','3-09','3-10','3-11','3-12'])
tempTemp = []
for i in range(0, 36):
    tempTemp.append(i)


weights = np.array([0.1, 0.2, 0.3, 0.4])

#simple moving average
d = pd.Series(dataFull)
d = d.rolling(4).mean()

#weighted moving average
wma = d.rolling(4).apply(lambda x: np.sum(weights*x))

#create training data
training, test = dataFull[:len(dataFull)-12], dataFull[len(dataFull)-12:]
trainingDates = dates[:len(dataFull)-24]
peMa = d[len(d)-12:]

#autoregressive
startArTime = time.time()
model = AutoReg(training, lags = 6)
model_fit = model.fit()
predictions = model_fit.predict(start=len(training), end=len(training)+len(test)-1, dynamic=False)
endArTime = time.time()
elaArTime = endArTime - startArTime

#predictive MA
startWmaTime = time.time()
maModel = ARIMA(training, order=(0, 0, 1), trend='t')
maModel_fit = maModel.fit()
maPredictions = maModel_fit.predict(start=len(training), end=len(training)+len(test)-1)
endWmaTime = time.time()
elaWmaTime = endWmaTime - startWmaTime

#percentage error
arPe = []
maPe=[]
for i in range(len(predictions)):
    #use peMa in place of test for comparing to actual moving average and add 24 to i because peMa starts from 24
    arCalc = (((predictions[i] - test[i]) / test[i]) * 100)
    maCalc = (((maPredictions[i] - test[i]) / test[i]) * 100)
    if arCalc < 0:
        arCalc = arCalc * -1 
    if maCalc < 0:
        maCalc = maCalc * -1
    arPe.append(round(arCalc, 1))
    maPe.append(round(maCalc, 1))

print(arPe)
print(maPe)

maAdded = 0
arAdded = 0
for i in range(len(arPe)):
    arAdded+=arPe[i]
    maAdded+=maPe[i]

arAvg = arAdded/len(arPe)
arAvg = round(arAvg, 1)
print("Autoregressive percentage error: " + str(arAvg))

maAvg = maAdded/len(maPe)
maAvg = round(maAvg, 1)
print("Moving average percentage error: " + str(maAvg))

#plot graph

plt.xlabel("AR: Time=" + str(round(elaArTime, 4)) + "s Percentage error=" + str(arAvg) + "%\nWMA: time=" + str(round(elaWmaTime, 4)) + "s Percentage error=" + str(maAvg) + "%")
plt.plot(tempTemp, dataFull, color = 'red')
plt.plot(dates, d, color='green')
plt.plot(dates, wma)

plt.plot(test, color='red')
plt.plot(trainingDates, peMa, color='green')
plt.plot(predictions, color='blue')
plt.plot(maPredictions, color='purple')

'''
print("Actual: "+str(peMa))
print("AR predictions: "+str(predictions))
print("MA predictions: "+str(maPredictions))
'''

plt.show()