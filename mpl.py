# mlp.py

from sklearn.neural_network import MLPClassifier  # Importamos el modelo de Perceptrón Multicapa (MLP)
from sklearn.metrics import accuracy_score  # Para evaluar la precisión del modelo
from main import get_data  # Importamos la función get_data() desde main.py para obtener los datos

# Importamos los datos desde el archivo main.py
X_train, X_val, y_train, y_val = get_data()

# Creamos el modelo de MLP con dos capas ocultas
mlp_model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)

# Entrenamos el modelo con el conjunto de entrenamiento
mlp_model.fit(X_train, y_train)

# Realizamos predicciones con el modelo entrenado
y_pred_mlp = mlp_model.predict(X_val)

# Calculamos la precisión del modelo en el conjunto de validación
accuracy_mlp = accuracy_score(y_val, y_pred_mlp)

# Imprimimos la precisión
print(f"Precisión de MLP: {accuracy_mlp}")
