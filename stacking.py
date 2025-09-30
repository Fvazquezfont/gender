# stacking.py

from sklearn.ensemble import StackingClassifier  # Importamos el modelo de Stacking
from sklearn.linear_model import LogisticRegression  # Usamos LogisticRegression como clasificador final
from sklearn.svm import SVC  # Usamos SVM como uno de los clasificadores base
from sklearn.tree import DecisionTreeClassifier  # Usamos Árbol de Decisión como uno de los clasificadores base
from sklearn.metrics import accuracy_score  # Para evaluar la precisión del modelo
from main import get_data  # Importamos la función get_data() desde main.py para obtener los datos

# Importamos los datos desde el archivo main.py
X_train, X_val, y_train, y_val = get_data()

# Definimos los clasificadores base para el Stacking
base_learners = [
    ('decision_tree', DecisionTreeClassifier(random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
]

# Definimos el clasificador final (Logistic Regression)
final_estimator = LogisticRegression()

# Creamos el modelo de Stacking
stacking_model = StackingClassifier(estimators=base_learners, final_estimator=final_estimator)

# Entrenamos el modelo con el conjunto de entrenamiento
stacking_model.fit(X_train, y_train)

# Realizamos predicciones con el modelo entrenado
y_pred_stack = stacking_model.predict(X_val)

# Calculamos la precisión del modelo en el conjunto de validación
accuracy_stack = accuracy_score(y_val, y_pred_stack)

# Imprimimos la precisión
print(f"Precisión de Stacking Classifier: {accuracy_stack}")
