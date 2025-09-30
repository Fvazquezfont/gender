# svm.py

from sklearn.svm import SVC  # Importamos el clasificador de Máquinas de Vectores de Soporte (SVM)
from sklearn.model_selection import GridSearchCV  # Para optimizar hiperparámetros con búsqueda en cuadrícula
from sklearn.metrics import accuracy_score  # Para evaluar el modelo con precisión
from main import get_data  # Importamos la función get_data() desde main.py para obtener los datos

# Importamos los datos desde el archivo main.py
X_train, X_val, y_train, y_val = get_data()

# Definimos el espacio de búsqueda de los hiperparámetros del SVM
param_grid_svm = {
    'C': [0.1, 1, 10],  # Parámetro de regularización
    'kernel': ['linear', 'rbf'],  # Tipos de núcleo (lineal o radial)
    'gamma': ['scale', 'auto']  # Tipo de gamma (escalado o automático)
}

# Creamos el modelo SVM
svm_model = SVC(random_state=42)

# Creamos una búsqueda en cuadrícula para encontrar los mejores hiperparámetros
grid_search_svm = GridSearchCV(estimator=svm_model, param_grid=param_grid_svm, cv=5, scoring='accuracy')

# Entrenamos el modelo con el conjunto de entrenamiento
grid_search_svm.fit(X_train, y_train)

# Realizamos predicciones con el mejor modelo encontrado
y_pred_svm = grid_search_svm.best_estimator_.predict(X_val)

# Calculamos la precisión del modelo en el conjunto de validación
accuracy_svm = accuracy_score(y_val, y_pred_svm)

# Imprimimos los mejores parámetros y la precisión
print("Mejores parámetros de SVM:", grid_search_svm.best_params_)
print("Precisión de SVM:", accuracy_svm)
