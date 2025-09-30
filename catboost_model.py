# catboost.py

from catboost import CatBoostClassifier  # Importamos el modelo de CatBoost
from sklearn.metrics import accuracy_score  # Para evaluar la precisión del modelo
from main import get_data  # Importamos la función get_data() desde main.py para obtener los datos
import joblib  # Para guardar el modelo entrenado

# Importamos los datos desde el archivo main.py
X_train, X_val, y_train, y_val = get_data()

# Creamos el modelo de CatBoost
catboost_model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=10, random_state=42, verbose=0)

# Entrenamos el modelo con el conjunto de entrenamiento
catboost_model.fit(X_train, y_train)

# Realizamos predicciones con el modelo entrenado
y_pred_catboost = catboost_model.predict(X_val)

# Calculamos la precisión del modelo en el conjunto de validación
accuracy_catboost = accuracy_score(y_val, y_pred_catboost)

# Imprimimos la precisión
print(f"Precisión de CatBoost: {accuracy_catboost}")

# Guardamos el modelo entrenado
joblib.dump(catboost_model, 'catboost_model.pkl')  # Guardamos el modelo
