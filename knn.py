# knn.py

from sklearn.neighbors import KNeighborsClassifier  # Importamos el clasificador K-Vecinos más Cercanos
from sklearn.model_selection import GridSearchCV  # Para optimizar hiperparámetros con búsqueda en cuadrícula
from sklearn.metrics import accuracy_score  # Para evaluar el modelo con precisión
from main import get_data  # Importamos la función get_data() desde main.py para obtener los datos

# Importamos los datos desde el archivo main.py
X_train, X_val, y_train, y_val = get_data()

# Definimos el espacio de búsqueda de los hiperparámetros de KNN
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 10],  # Número de vecinos a considerar
    'weights': ['uniform', 'distance'],  # Tipo de peso (uniforme o por distancia)
    'metric': ['euclidean', 'manhattan']  # Tipo de métrica (Euclidiana o Manhattan)
}

# Creamos el modelo de KNN
knn_model = KNeighborsClassifier()

# Creamos una búsqueda en cuadrícula para encontrar los mejores hiperparámetros
grid_search_knn = GridSearchCV(estimator=knn_model, param_grid=param_grid_knn, cv=5, scoring='accuracy')

# Entrenamos el modelo con el conjunto de entrenamiento
grid_search_knn.fit(X_train, y_train)

# Realizamos predicciones con el mejor modelo encontrado
y_pred_knn = grid_search_knn.best_estimator_.predict(X_val)

# Calculamos la precisión del modelo en el conjunto de validación
accuracy_knn = accuracy_score(y_val, y_pred_knn)

# Imprimimos los mejores parámetros y la precisión
print("Mejores parámetros de K-NN:", grid_search_knn.best_params_)
print("Precisión de K-NN:", accuracy_knn)
