# main.py

import pandas as pd  # Importamos pandas para manejar los datos en formato DataFrame
from sklearn.model_selection import train_test_split  # Para dividir los datos en entrenamiento y validación
from sklearn.preprocessing import LabelEncoder  # Para codificar las letras del nombre como valores numéricos
from sklearn.metrics import accuracy_score  # Para calcular la precisión de los modelos

# Cargar los archivos CSV con los nombres y etiquetas de género
male_df = pd.read_csv("male_spanish_names_reduced.csv")  # Cargamos el archivo de nombres masculinos
female_df = pd.read_csv("female_spanish_names_reduced.csv")  # Cargamos el archivo de nombres femeninos

# Asignamos etiquetas de género: 1 para masculino, 0 para femenino
male_df["Gender"] = 1
female_df["Gender"] = 0

# Unimos ambos dataframes en uno solo
names_df = pd.concat([male_df, female_df], ignore_index=True)  # Concatenamos los dataframes de nombres

# Mezclamos los datos de manera aleatoria para evitar sesgos
names_df = names_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Función para extraer características de los nombres
def extract_features(df):
    features = pd.DataFrame()  # Creamos un DataFrame vacío para las nuevas características
    names = df["Name"].str.lower()  # Convertimos los nombres a minúsculas para uniformidad
    
    # Extraemos características como la longitud del nombre, última y primera letra
    features["length"] = names.str.len()  # Longitud del nombre
    features["last_letter"] = names.str[-1]  # Última letra del nombre
    features["first_letter"] = names.str[0]  # Primera letra del nombre
    features["ends_with_vowel"] = names.str[-1].isin(list("aeiou")).astype(int)  # 1 si termina en vocal, 0 si no

    # Codificación de las letras (transformar letras a números)
    le_first = LabelEncoder()  # Creamos un objeto para codificar la primera letra
    le_last = LabelEncoder()  # Creamos un objeto para codificar la última letra
    features["first_letter"] = le_first.fit_transform(features["first_letter"])  # Codificamos la primera letra
    features["last_letter"] = le_last.fit_transform(features["last_letter"])  # Codificamos la última letra

    return features  # Devolvemos las características extraídas

# Extraemos las características de los nombres
features_df = extract_features(names_df)

# Etiquetas de género (0 o 1)
label = names_df["Gender"]

# Dividimos los datos en un conjunto de entrenamiento y otro de validación (80% - 20%)
X_train, X_val, y_train, y_val = train_test_split(features_df, label, test_size=0.2, random_state=42)

# Función para devolver los datos preprocesados, útil para importarlos en otros archivos
def get_data():
    return X_train, X_val, y_train, y_val
