import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from main import extract_features, get_data
import joblib

# Function to load the test data and evaluate the model
def evaluate_model_on_test(test_file_path, model):
    # Load the test dataset
    test_df = pd.read_csv(test_file_path)

    # Extract features from the test dataset
    test_features = extract_features(test_df)
    
    # The actual gender labels from the test dataset
    y_test = test_df["Gender"]

    # Predict the gender using the trained model
    y_pred = model.predict(test_features)

    # Calculate the accuracy of the model on the test dataset
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test dataset: {accuracy}")
model = joblib.load('logistic_regression_model.pkl')  # Load the trained model (change filename as needed)
# Evaluate the model on the test dataset
evaluate_model_on_test("test_dataset.csv", model)
