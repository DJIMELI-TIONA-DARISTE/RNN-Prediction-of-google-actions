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


#prédiction et visualisation
