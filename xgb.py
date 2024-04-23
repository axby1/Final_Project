from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from xgboost import DMatrix,train





# Load or preprocess your data as in your existing code
data = pd.read_csv("feature.csv")
data.drop(columns='Unnamed: 0', inplace=True)
data.replace(True, 1, inplace=True)
data.replace(False, 0, inplace=True)
y = data["File"]
data = data.drop(columns="File")

encoder = LabelEncoder()
encoder.fit(y)
Y = encoder.transform(y)

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(data)

# Combine features and labels into a single DataFrame
dataset = pd.concat([pd.DataFrame(X), pd.Series(Y, name='Label')], axis=1)

# Shuffle the dataset
shuffled_dataset = shuffle(dataset, random_state=42)

# Separate the shuffled features and labels
shuffled_X = shuffled_dataset.drop(columns='Label')
shuffled_Y = shuffled_dataset['Label']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(shuffled_X, shuffled_Y, test_size=0.2, random_state=42)

# Create DMatrix objects for XGBoost
dtrain = DMatrix(X_train, label=Y_train)
dtest = DMatrix(X_test, label=Y_test)

# Set XGBoost parameters for multi-class classification
params = {
    'objective': 'multi:softmax',  # for multi-class classification
    'num_class': len(np.unique(Y)),  # Number of unique classes
    'eval_metric': 'mlogloss',  # log-likelihood loss
}

# Train XGBoost model
num_rounds = 500
xg_model = train(params, dtrain, num_rounds)

# Make predictions on the test set
xg_predictions = xg_model.predict(dtest)

# Convert predictions to integers
xg_predictions = xg_predictions.astype(int)

# Evaluate XGBoost model
xg_accuracy = accuracy_score(Y_test, xg_predictions)
xg_classification_report = classification_report(Y_test, xg_predictions)

# Display XGBoost metrics
print("\nXGBoost Metrics:")
print("Accuracy:", xg_accuracy)
print("Classification Report:")
target_names = ['Benign', 'Defacement', 'Malware', 'Phishing', 'Spam']
print(xg_classification_report)



