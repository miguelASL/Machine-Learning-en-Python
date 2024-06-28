# Regresion Lineal Multiple 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Dividir el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split # Esta clase se encarga de dividir el dataset en dos partes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0) # 20% de los datos para test

#Codificar datos categoricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # Esta clase se encarga de codificar los datos categoricos

labelencoder_X = LabelEncoder() # Se crea un objeto de la clase LabelEncoder
X[:, 3] = labelencoder_X.fit_transform(X[:, 3]) # Se codifica la columna 3
onehotencoder = OneHotEncoder(categorical_features = [3]) # Se crea una columna por cada categoria
X = onehotencoder.fit_transform(X).toarray() # Se crea una columna por cada categoria

# Evitar la trampa de las variables ficticias
X = X[:, 1:] # Se elimina una columna para evitar la trampa de las variables ficticias

# Dividir el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split # Esta clase se encarga de dividir el dataset en dos partes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # 20% de los datos para test

# Escalado de variables
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

# Ajustar el modelo de Regresion Lineal Multiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression() # Crear un objeto de la clase LinearRegression
regression.fit(X_train, y_train) # Ajustar el modelo de regresion lineal multiple con el conjunto de entrenamiento

# Predecir el conjunto de test
y_pred = regression.predict(X_test) # Vector de predicciones de los datos de test 

# Construir el modelo de eliminacion hacia atras
import statsmodels.formula.api as sm # Se importa la libreria statsmodels.formula.api
X = np.append(arr = np.ones((50, 1)).astype(int), values= X, axis=1) # Se añade una columna de unos al principio de la matriz X para poder realizar la eliminacion hacia atras
SL = 0.05 # Nivel de significacion

X_opt= X[:, [0, 1, 2, 3, 4, 5]] # Se crea una matriz X_opt que contendra las variables mas significativas
regression_OLS = sm.OLS(endog=y, exog=X_opt).fit() # Se crea un objeto de la clase OLS
regression_OLS.summary() # Se muestra un resumen de la regresion

X_opt= X[:, [0, 1, 3, 4, 5]]
regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regression_OLS.summary()

X_opt= X[:, [3, 4, 5]]
regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regression_OLS.summary()

X_opt= X[:, [0, 3, 5]]
regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regression_OLS.summary()

X_opt= X[:, [0, 3]]
regression_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regression_OLS.summary()