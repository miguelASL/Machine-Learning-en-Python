# Regresion Lineal Simple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Dividir el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split # Esta clase se encarga de dividir el dataset en dos partes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0) # 20% de los datos para test

# Escalado de variables
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

# Crear modelo de Regresión Lienal Simple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression # Esta clase se encarga de hacer la regresion lineal

regression = LinearRegression()
regression.fit(X_train, y_train) # X_train es la matriz de variables independientes y_train es el vector de variables dependientes

# Predecir el conjunto de test
y_pred = regression.predict(X_test) # regression.predict(X_train) es el vector de variables dependientes predichas

# Visualizar los resultados de entrenamiento
# X_train es la matriz de variables independientes y_train es el vector de variables dependientes
# y_train es el vector de variables dependientes reales
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regression.predict(X_train), color = 'blue')
plt.title('Salario vs Experiencia (Conjunto de Entrenamiento)')
plt.xlabel('Años de Experiencia')
plt.ylabel('Salario')
plt.show()

# Visualizar los resultados de entrenamiento
plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_train, regression.predict(X_train), color = 'orange')
plt.title('Salario vs Experiencia (Conjunto de Testing)')
plt.xlabel('Años de Experiencia')
plt.ylabel('Salario')
plt.show()
