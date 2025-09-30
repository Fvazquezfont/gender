# lightgbm.py

import lightgbm as lgb  # Importamos la librería LightGBM
from sklearn.metrics import accuracy_score  # Para evaluar la precisión del modelo
from main import get_data  # Importamos la función get_data() desde main.py para obtener los datos

# Importamos los datos desde el archivo main.py
X_train, X_val, y_train, y_val = get_data()

# Creamos el modelo de LightGBM
lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=42)

# Entrenamos el modelo con el conjunto de entrenamiento
lgb_model.fit(X_train, y_train)

# Realizamos predicciones con el modelo entrenado
y_pred_lgb = lgb_model.predict(X_val)

# Calculamos la precisión del modelo en el conjunto de validación
accuracy_lgb = accuracy_score(y_val, y_pred_lgb)

# Imprimimos la precisión
print(f"Precisión de LightGBM: {accuracy_lgb}")
