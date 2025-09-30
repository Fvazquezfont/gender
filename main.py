import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Cargar los CSV
male_df = pd.read_csv("male_spanish_names_reduced.csv")
female_df = pd.read_csv("female_spanish_names_reduced.csv")

# Asignar etiquetas: 1 = masculino, 0 = femenino
male_df["Gender"] = 1
female_df["Gender"] = 0

# Unir ambos datasets
names_df = pd.concat([male_df, female_df], ignore_index=True)

# Ordenar aleatoriamente todos los registros
names_df = names_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Función para extraer características
def extract_features(df):
    features = pd.DataFrame()
    names = df["Name"].str.lower()
    features["length"] = names.str.len()
    features["last_letter"] = names.str[-1]
    features["first_letter"] = names.str[0]
    features["ends_with_vowel"] = names.str[-1].isin(list("aeiou")).astype(int)

    # Codificación de letras (variables categóricas → numéricas)
    le_first = LabelEncoder()
    le_last = LabelEncoder()
    features["first_letter"] = le_first.fit_transform(features["first_letter"])
    features["last_letter"] = le_last.fit_transform(features["last_letter"])

    return features

# Extraer características
features_df = extract_features(names_df)

# Definir etiquetas (género)
label = names_df["Gender"]

# Dividir los datos en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(features_df, label, test_size=0.2, random_state=42)

# 1. **Árbol de Decisión (Decision Tree)**
# Ajuste de hiperparámetros: max_depth (profundidad)
param_grid_dt = {
    'max_depth': [1, 3, 5, 7, 10, None]
}
dt_model = DecisionTreeClassifier(random_state=42)
grid_search_dt = GridSearchCV(estimator=dt_model, param_grid=param_grid_dt, cv=5, scoring='accuracy')
grid_search_dt.fit(X_train, y_train)

print("Mejores parámetros de Árbol de Decisión:", grid_search_dt.best_params_)
print("Mejor precisión de Árbol de Decisión:", grid_search_dt.best_score_)

# 2. **Random Forest (Bosque Aleatorio)**
# Ajuste de hiperparámetros: n_estimators, max_depth, min_samples_split, max_features
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'max_features': ['auto', 'sqrt', 'log2']
}
rf_model = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)

print("Mejores parámetros de Random Forest:", grid_search_rf.best_params_)
print("Mejor precisión de Random Forest:", grid_search_rf.best_score_)

# 3. **Support Vector Machine (SVM)**
# Ajuste de hiperparámetros: C, kernel, gamma
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
svm_model = SVC(random_state=42)
grid_search_svm = GridSearchCV(estimator=svm_model, param_grid=param_grid_svm, cv=5, scoring='accuracy')
grid_search_svm.fit(X_train, y_train)

print("Mejores parámetros de SVM:", grid_search_svm.best_params_)
print("Mejor precisión de SVM:", grid_search_svm.best_score_)

# 4. **K-Nearest Neighbors (KNN)**
# Ajuste de hiperparámetros: n_neighbors, weights, metric
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
knn_model = KNeighborsClassifier()
grid_search_knn = GridSearchCV(estimator=knn_model, param_grid=param_grid_knn, cv=5, scoring='accuracy')
grid_search_knn.fit(X_train, y_train)

print("Mejores parámetros de KNN:", grid_search_knn.best_params_)
print("Mejor precisión de KNN:", grid_search_knn.best_score_)

# 5. **Naive Bayes**
# No requiere mucho ajuste, se usa el modelo directamente
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

y_pred_nb = nb_model.predict(X_val)
accuracy_nb = accuracy_score(y_val, y_pred_nb)
print(f"Naive Bayes Accuracy: {accuracy_nb}")

# 6. **XGBoost (Extreme Gradient Boosting)**
# Ajuste de hiperparámetros: n_estimators, learning_rate, max_depth
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 10]
}
xgb_model = xgb.XGBClassifier(random_state=42)
grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, cv=5, scoring='accuracy')
grid_search_xgb.fit(X_train, y_train)

print("Mejores parámetros de XGBoost:", grid_search_xgb.best_params_)
print("Mejor precisión de XGBoost:", grid_search_xgb.best_score_)



#Decision Tree: Bueno para empezar, fácil de interpretar.
#Random Forest: Más robusto, no propenso a sobreajuste.
#SVM: Bueno para datos de alta dimensión, pero puede ser lento.
#KNN: Simple, pero puede ser ineficiente con grandes conjuntos de datos.
#Naive Bayes: Rápido, pero asume independencia entre características.
#XGBoost: Muy potente, pero puede ser complejo de ajustar.