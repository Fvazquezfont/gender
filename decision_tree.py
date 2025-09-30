# decision_tree.py

from sklearn.tree import DecisionTreeClassifier  # Importamos el clasificador de Árbol de Decisión
from sklearn.model_selection import GridSearchCV  # Para optimizar hiperparámetros con búsqueda en cuadrícula
from sklearn.metrics import accuracy_score  # Para evaluar el modelo con precisión
from main import get_data  # Importamos la función get_data() desde main.py para obtener los datos
import joblib  # Para guardar el modelo entrenado
# Importamos los datos desde el archivo main.py
X_train, X_val, y_train, y_val = get_data()

# Definimos el espacio de búsqueda de los hiperparámetros del Árbol de Decisión
param_grid_dt = {'max_depth': [1, 3, 5, 7, 10, None]}  # Probaremos diferentes profundidades del árbol

# Creamos el modelo de Árbol de Decisión
dt_model = DecisionTreeClassifier(random_state=42)

# Creamos una búsqueda en cuadrícula para encontrar los mejores hiperparámetros
grid_search_dt = GridSearchCV(estimator=dt_model, param_grid=param_grid_dt, cv=5, scoring='accuracy')

# Entrenamos el modelo con el conjunto de entrenamiento
grid_search_dt.fit(X_train, y_train)

# Realizamos predicciones con el mejor modelo encontrado
y_pred_dt = grid_search_dt.best_estimator_.predict(X_val)

# Calculamos la precisión del modelo en el conjunto de validación
accuracy_dt = accuracy_score(y_val, y_pred_dt)

# Imprimimos los mejores parámetros y la precisión
print("Mejores parámetros de Árbol de Decisión:", grid_search_dt.best_params_)
print("Precisión de Árbol de Decisión:", accuracy_dt)

# Guardamos el modelo entrenado
joblib.dump(grid_search_dt.best_estimator_, 'decision_tree_model.pkl')  # Guard