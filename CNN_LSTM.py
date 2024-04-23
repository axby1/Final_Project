import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

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
Y = encoder.fit_transform(y)

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

# Reshape data for CNN-LSTM input
sequence_length = 1  # Replace with appropriate value

# Reshape data for CNN-LSTM input using the fixed sequence length
X_train_cnn_lstm = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_cnn_lstm = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define CNN-LSTM model for multi-class classification
num_classes = len(np.unique(Y))
cnn_lstm_model = Sequential()
cnn_lstm_model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
cnn_lstm_model.add(MaxPooling1D(pool_size=2))
cnn_lstm_model.add(Flatten())
cnn_lstm_model.add(Dense(50, activation='relu'))

cnn_lstm_model.add(Dense(60, activation='relu'))  # Additional hidden layer
cnn_lstm_model.add(Dense(50, activation='relu'))  # Additional hidden layer
cnn_lstm_model.add(Dense(40, activation='relu'))  # Additional hidden layer
cnn_lstm_model.add(Dense(30, activation='relu'))  # Additional hidden layer
cnn_lstm_model.add(Dense(20, activation='relu'))  # Additional hidden layer
cnn_lstm_model.add(Dense(15, activation='relu'))  # Additional hidden layer




cnn_lstm_model.add(Dense(num_classes, activation='softmax'))  # Softmax for multi-class
cnn_lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
cnn_lstm_model.fit(X_train_cnn_lstm, Y_train, epochs=25, batch_size=32)

# Evaluate CNN-LSTM
cnn_lstm_probabilities = cnn_lstm_model.predict(X_test_cnn_lstm)
cnn_lstm_predictions = np.argmax(cnn_lstm_probabilities, axis=1)
cnn_lstm_accuracy = accuracy_score(Y_test, cnn_lstm_predictions)
target_names = ['Benign', 'Defacement', 'Malware', 'Phishing', 'Spam']
cnn_lstm_classification_report = classification_report(Y_test, cnn_lstm_predictions,target_names=target_names)

# Display CNN-LSTM metrics
print("\nCNN-LSTM Metrics:")
print("Accuracy:", cnn_lstm_accuracy)
print("Classification Report:")
print(cnn_lstm_classification_report)

# Save the trained CNN-LSTM model
cnn_lstm_model.save("cnn_lstm_model.h5")

# Load the saved CNN-LSTM model
loaded_cnn_lstm_model = load_model("cnn_lstm_model.h5")

# Now, you can use the loaded model to make predictions on new data
while True:
    url = input("Enter a URL: ")
    features = d.UrlFeaturizer(url).run()

    selected_features = ['bodyLength', 'bscr', 'dse', 'dsr', 'entropy', 'hasHttp', 'hasHttps',
                         'has_ip', 'numDigits', 'numImages', 'numLinks', 'numParams',
                         'numTitles', 'num_%20', 'num_@', 'sbr', 'scriptLength', 'specialChars',
                         'sscr', 'urlIsLive', 'urlLength']

    # Reshape the features for CNN-LSTM input
    input_features = np.array([features[feature_name] for feature_name in selected_features])
    input_features_cnn_lstm = input_features.reshape((1, len(selected_features), 1))

    # Make predictions using the loaded CNN-LSTM model
    loaded_cnn_lstm_probabilities = loaded_cnn_lstm_model.predict(input_features_cnn_lstm)
    loaded_cnn_lstm_predictions = np.argmax(loaded_cnn_lstm_probabilities, axis=1)

    # Map predictions back to class labels using the encoder
    predicted_classes = encoder.inverse_transform(loaded_cnn_lstm_predictions)

    # Print the predicted class
    print("Predicted class:", predicted_classes[0])
