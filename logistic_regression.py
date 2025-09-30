# logistic_regression.py

from sklearn.linear_model import LogisticRegression  # Importamos el modelo de Regresión Logística
from sklearn.metrics import accuracy_score  # Para evaluar la precisión del modelo
from main import get_data  # Importamos la función get_data() desde main.py para obtener los datos
import joblib  # Para guardar el modelo entrenado
# Importamos los datos desde el archivo main.py
X_train, X_val, y_train, y_val = get_data()

# Creamos el modelo de Regresión Logística
lr_model = LogisticRegression(max_iter=1000, random_state=42)

# Entrenamos el modelo con el conjunto de entrenamiento
lr_model.fit(X_train, y_train)

# Realizamos predicciones con el modelo entrenado
y_pred_lr = lr_model.predict(X_val)

# Calculamos la precisión del modelo en el conjunto de validación
accuracy_lr = accuracy_score(y_val, y_pred_lr)

# Imprimimos la precisión
print(f"Precisión de Regresión Logística: {accuracy_lr}")

# Guardamos el modelo entrenado
joblib.dump(lr_model, 'logistic_regression_model.pkl')