# random_forest.py

from sklearn.ensemble import RandomForestClassifier  # Importamos el clasificador de Bosque Aleatorio
from sklearn.model_selection import GridSearchCV  # Para optimizar hiperparámetros con búsqueda en cuadrícula
from sklearn.metrics import accuracy_score  # Para evaluar el modelo con precisión
from main import get_data  # Importamos la función get_data() desde main.py para obtener los datos

# Importamos los datos desde el archivo main.py
X_train, X_val, y_train, y_val = get_data()

# Definimos el espacio de búsqueda de los hiperparámetros del Bosque Aleatorio
param_grid_rf = {
    'n_estimators': [50, 100, 200],  # Número de árboles en el bosque
    'max_depth': [3, 5, 10, None],  # Profundidad máxima de los árboles
    'min_samples_split': [2, 5, 10],  # Número mínimo de muestras para dividir un nodo
    'max_features': ['auto', 'sqrt', 'log2']  # Número de características a considerar en cada árbol
}

# Creamos el modelo de Bosque Aleatorio
rf_model = RandomForestClassifier(random_state=42)

# Creamos una búsqueda en cuadrícula para encontrar los mejores hiperparámetros
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, scoring='accuracy')

# Entrenamos el modelo con el conjunto de entrenamiento
grid_search_rf.fit(X_train, y_train)

# Realizamos predicciones con el mejor modelo encontrado
y_pred_rf = grid_search_rf.best_estimator_.predict(X_val)

# Calculamos la precisión del modelo en el conjunto de validación
accuracy_rf = accuracy_score(y_val, y_pred_rf)

# Imprimimos los mejores parámetros y la precisión
print("Mejores parámetros de Random Forest:", grid_search_rf.best_params_)
print("Precisión de Random Forest:", accuracy_rf)
