# 1. Load the input CSVs (holding male and female names separately) using pandas.read_csv().
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Cargar los CSV
male_df = pd.read_csv("male_spanish_names_reduced.csv")
female_df = pd.read_csv("female_spanish_names_reduced.csv")


# 2. Merge the datasets, ensuring they're correctly labelled (0 = female name, 1 = male name) to form the target array.


# Asignar etiquetas: 1 = masculino, 0 = femenino
male_df["Gender"] = 1
female_df["Gender"] = 0

# Unir ambos datasets
names_df = pd.concat([male_df, female_df], ignore_index=True)
print("names_df")

# Merge the datasets
# Ordenar aleatoriamente todos los registros
names_df = names_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 3. Extract features from the names in the combined dataset. For instance, you could consider the length of the name, the last letter of the name, and so forth.

# Function to extract features from names
def extract_features(df):

    features = pd.DataFrame()  # create an empty pandas DataFrame called features.

    # Convert all names to lowercase for consistency
    names = df["Name"].str.lower()

    # Keep the original names (in lowercase) NO PUEDE ENTRAR EN EL TRAINING
    #features["name"] = names

    # Add gender column (0 = female, 1 = male) NO PUEDE ENTRAR EN EL TRAINING
    #features["gender"] = df["Gender"]

    # Name length
    features["length"] = names.str.len()

    # Last letter
    features["last_letter"] = names.str[-1]

    # First letter
    features["first_letter"] = names.str[0]

    # Does the name end with a vowel? (boolean 0/1) #"ends_with_vowel" será 1 si el nombre termina en vocal, 0 si no.
    features["ends_with_vowel"] = names.str[-1].isin(list("aeiou")).astype(int)

    # Encode letters (categorical variables → numeric)
    le_first = LabelEncoder()
    le_last = LabelEncoder()

    features["first_letter"] = le_first.fit_transform(features["first_letter"])
    features["last_letter"] = le_last.fit_transform(features["last_letter"])

    return features

# Extract features
features_df = extract_features(names_df)
print("features_df")
print(features_df.head())
print(features_df.tail())
# 4. Split the dataset into training and validation sets using sklearn.model_selection.train_test_split().
from sklearn.model_selection import train_test_split
label = names_df["Gender"]
print(label)
X_train, X_val, y_train, y_val = train_test_split(features_df, label, test_size=0.3, random_state=42)
# 5. Train a machine learning model (e.g., a decision tree or logistic regression) using the training set.
from sklearn.tree import DecisionTreeClassifier
# Train a Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
DecisionTreeClassifier.fit(X_train, y_train)
# Crear y entrenar el modelo
def train_and_evaluate(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"{model_name} Accuracy: {accuracy}")
# Modelos a evaluar