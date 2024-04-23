from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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

# Use CatBoostClassifier directly
cat_model = CatBoostClassifier(iterations=500, loss_function='MultiClass', classes_count=len(encoder.classes_), eval_metric='MultiClass', random_seed=42)
cat_model.fit(X_train, Y_train, eval_set=(X_test, Y_test), verbose=False)

# Make predictions on the test set
cat_predictions = cat_model.predict(X_test)

# Evaluate CatBoost model
cat_accuracy = accuracy_score(Y_test, cat_predictions)
cat_classification_report = classification_report(Y_test, cat_predictions)

# Display CatBoost metrics
print("\nCatBoost Metrics:")
print("Accuracy:", cat_accuracy)
print("Classification Report:")
target_names = ['Benign', 'Defacement', 'Malware', 'Phishing', 'Spam']
print(cat_classification_report)
