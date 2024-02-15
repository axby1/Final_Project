import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM, Dense

import data_creation_v3 as d

# Load or preprocess your data as in your existing code
data = pd.read_csv("feature.csv")
data.drop(columns='Unnamed: 0', inplace=True)
data.replace(True, 1, inplace=True)
data.replace(False, 0, inplace=True)
y = data["File"]
data = data.drop(columns="File")

# Encoding labels using LabelEncoder
encoder = LabelEncoder()
encoder.fit(y)
Y = encoder.transform(y)

# Scaling features using MinMaxScaler
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

# Reshape data for LSTM input
sequence_length = 1  # Replace with appropriate value

# Reshape data for LSTM input using the fixed sequence length
X_train_lstm = np.array([X_train.iloc[i:i+sequence_length].values for i in range(len(X_train) - sequence_length + 1)])
X_test_lstm = np.array([X_test.iloc[i:i+sequence_length].values for i in range(len(X_test) - sequence_length + 1)])

# Define LSTM model for multi-class classification
num_classes = len(np.unique(Y))  # Number of unique classes
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, input_shape=(sequence_length, X_train.shape[1])))
lstm_model.add(Dense(num_classes, activation='softmax'))  # Softmax for multi-class
lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # Use sparse categorical crossentropy for integer labels
lstm_model.fit(X_train_lstm, Y_train, epochs=10, batch_size=32)

# Evaluate LSTM
lstm_probabilities = lstm_model.predict(X_test_lstm)

# Get the predicted class for each sample
lstm_predictions = np.argmax(lstm_probabilities, axis=1)

# Evaluate LSTM
lstm_accuracy = accuracy_score(Y_test, lstm_predictions)
lstm_classification_report = classification_report(Y_test, lstm_predictions)

# Display LSTM metrics
print("\nLSTM Metrics:")
print("Accuracy:", lstm_accuracy)
print("Classification Report:")
target_names = ['Benign', 'Defacement', 'Malware', 'Phishing', 'Spam']
classification_rep = classification_report(Y_test, lstm_predictions, target_names=target_names)

# Print the classification report
print(classification_rep)




# Save the trained LSTM model
lstm_model.save("lstm_model.h5")

# Load the saved LSTM model
loaded_lstm_model = load_model("lstm_model.h5")

# Now, you can use the loaded model to make predictions on new data
while 1:
    url = input("enter a url: ")
    features = d.UrlFeaturizer(url).run()

    selected_features = ['bodyLength', 'bscr', 'dse', 'dsr', 'entropy', 'hasHttp', 'hasHttps',
                 'has_ip', 'numDigits', 'numImages', 'numLinks', 'numParams',
                 'numTitles', 'num_%20', 'num_@', 'sbr', 'scriptLength', 'specialChars',
                 'sscr', 'urlIsLive', 'urlLength']


    # Reshape the features for LSTM input

    # Assuming 'features' is a dictionary returned by the UrlFeaturizer class
    input_features = np.array([features[feature_name] for feature_name in selected_features])

    # Reshape the features for LSTM input
    input_features_lstm = input_features.reshape((1, sequence_length, len(selected_features)))

    # Make predictions using the loaded LSTM model
    loaded_lstm_probabilities = loaded_lstm_model.predict(input_features_lstm)
    loaded_lstm_predictions = np.argmax(loaded_lstm_probabilities, axis=1)



    # Map predictions back to class labels using the encoder
    predicted_classes = encoder.inverse_transform(loaded_lstm_predictions)

    # Print the predicted class
    print("Predicted class:", predicted_classes[0])
