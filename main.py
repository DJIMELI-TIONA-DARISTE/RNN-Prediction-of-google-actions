#prédiction du prix des action google au mois de janvier 2017

#Librairie
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#préparation des données
    #jeu de données d'entrainement
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train[["Open"]].values
    #modification de l'échelle du jeux de données:la normalisation
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

    #création de la structure avec 60 timesteps  et 1 sortie
x_train = []
y_train = []
for i in range(60,1258):
    x_train.append(training_set_scaled[(i-60):i,0])
    y_train.append(training_set_scaled[i,0])
x_train = np.array(x_train)
y_train = np.array(y_train)

    #Reshaping
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# Construction RNN

    #Librairies
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential() #initialisation du modéle

    #1e couche de LSTM + Dropout
regressor.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(0.2))

    #2e couche de LSTM + Dropout
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

    #3e couche de LSTM + Dropout
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

    #4e couche de LSTM + Dropout
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

    #couche de sortie
regressor.add(Dense(units=1))

    #compilation 
regressor.compile(optimizer="adam", loss="mean_squared_error")

    #entrainement
regressor.fit(x_train, y_train, epochs=150, batch_size=32)

#prédiction et visualisation
    #données de test 2017
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test[["Open"]].values

    #prédiction pour 2017 
dataset_total = pd.concat((dataset_train["Open"],dataset_test["Open"]),axis=0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
 
x_test = []
for i in range(60,80):
    x_test.append(inputs[(i-60):i,0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


    #visualisation des résultats
plt.plot(real_stock_price, label="prix réel de l'action Google")
plt.plot(predicted_stock_price, label="prix prédit de l'action Google")
plt.title("prédiction de l'action Google")
plt.xlabel("Jour")
plt.ylabel("Prix de l'action")
plt.legend()
plt.show()
  
#Evaluation du modèle
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))






