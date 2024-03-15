import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

import data_creation_v3 as d
import tensorflow
from keras.models import Sequential
from keras.layers import LSTM, Dense

if __name__ == '__main__':
    # converts all the features to their numerical equivalent and assigned labels(0-4) for the 5 different datasets
    data = pd.read_csv("feature.csv")
    data.drop(columns='Unnamed: 0',inplace=True)
    data.replace(True,1,inplace = True)
    data.replace(False,0,inplace = True)
    y = data["File"]
    data = data.drop(columns = "File")

    encoder = LabelEncoder()
    encoder.fit(y)
    Y = encoder.transform(y) # Y holds the label number

    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(data)
    X = pd.DataFrame(X)  # X holds the 21 features

    #print(X.head())

    # Assuming you have X (features) and Y (labels)
    # X and Y should be NumPy arrays or pandas DataFrames/Series

    # Combine features and labels into a single DataFrame
    dataset = pd.concat([pd.DataFrame(X), pd.Series(Y, name='Label')], axis=1)

    # Numerical representation (e.g., scaling, encoding)
    # Perform any numerical transformations on the features if needed

    # Shuffle the dataset
    #shuffled_dataset = shuffle(dataset, random_state=42)  # Set a random_state for reproducibility(random state remains constant)
    shuffled_dataset = shuffle(dataset) #randomized everytime
    # Separate the shuffled features and labels
    shuffled_X = shuffled_dataset.drop(columns='Label')
    shuffled_Y = shuffled_dataset['Label']

    #print(shuffled_Y)



    # Assuming you have 'shuffled_X' as features and 'shuffled_Y' as labels

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(shuffled_X, shuffled_Y, test_size=0.2, random_state=42)
    #
    # # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, Y_train)
    rf_predictions = rf_model.predict(X_test)

    # Evaluate Random Forest
    print("Random Forest Metrics:")
    print("Accuracy:", accuracy_score(Y_test, rf_predictions))
    print("Classification Report:")
    target_names = ['Benign', 'Defacement', 'Malware', 'Phishing', 'Spam']
    print(classification_report(Y_test, rf_predictions,target_names=target_names))

    np.save('lblenc.npy', encoder.classes_)
    scalerfile = 'scaler.sav'
    pickle.dump(scaler, open(scalerfile, 'wb'))

    # Define order of features
    order = ['bodyLength', 'bscr', 'dse', 'dsr', 'entropy', 'hasHttp', 'hasHttps',
             'has_ip', 'numDigits', 'numImages', 'numLinks', 'numParams',
             'numTitles', 'num_%20', 'num_@', 'sbr', 'scriptLength', 'specialChars',
             'sscr', 'urlIsLive', 'urlLength']

    # Load encoder classes
    encoder = LabelEncoder()
    encoder_path = "lblenc.npy"
    if os.path.exists(encoder_path):
        encoder.classes_ = np.load(encoder_path, allow_pickle=True)
    else:
        print(f"Error: File not found - {encoder_path}")


    # Load scaler
    scaler_path = 'scaler.sav'
    if os.path.exists(scaler_path):
        scaler = pickle.load(open(scaler_path, 'rb'))
    else:
        print(f"Error: File not found - {scaler_path}")


    url=input("enter a valid url")

    # Extract features
    features = d.UrlFeaturizer(url).run()
    test = pd.DataFrame([features[i] for i in order]).replace(True, 1).replace(False, 0).to_numpy().reshape(1, -1)
    # Make prediction
    scaled_features = scaler.transform(test)
    predicted = rf_model.predict(scaled_features)
    predicted_class = encoder.inverse_transform(predicted)

    print(f"Predicted class: {predicted_class}")



    # feature_importances = rf_model.feature_importances_
    # feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
    # feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    # print("Feature Importance:")
    # print(feature_importance_df)
    #
    # import shap
    #
    # # Calculate SHAP values
    # explainer = shap.TreeExplainer(rf_model)
    # shap_values = explainer.shap_values(X_test)
    #
    # # Summary plot
    # shap.summary_plot(shap_values, X_test, feature_names=X_test.columns)




    # LSTM
    # Assuming reshaping is needed for LSTM input

    # X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    # X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
    #
    # lstm_model = Sequential()
    # # Define your LSTM architecture here, for example:
    # lstm_model.add(LSTM(units=50, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
    # lstm_model.add(Dense(1, activation='sigmoid'))
    # lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # lstm_model.fit(X_train_lstm, Y_train, epochs=10, batch_size=32)
    #
    # # Evaluate LSTM
    # lstm_probabilities = lstm_model.predict(X_test_lstm)
    #
    # # Applying a threshold (e.g., 0.5) to convert probabilities to binary predictions
    # lstm_predictions = (lstm_probabilities > 0.5).astype(int)
    #
    # # Evaluate LSTM
    # lstm_accuracy = accuracy_score(Y_test, lstm_predictions)
    # lstm_classification_report = classification_report(Y_test, lstm_predictions)
    #
    # # Display LSTM metrics
    # print("\nLSTM Metrics:")
    # print("Accuracy:", lstm_accuracy)
    # print("Classification Report:")
    # print(lstm_classification_report)
