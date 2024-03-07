import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import pickle
import os
import data_creation_v3 as d

def preprocessing(file_path="feature.csv", random_state=42):
    data = pd.read_csv(file_path)
    data.drop(columns='Unnamed: 0', inplace=True)
    data.replace(True, 1, inplace=True)
    data.replace(False, 0, inplace=True)

    y = data["File"]
    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(data.drop(columns="File"))

    encoder = LabelEncoder().fit(y)
    Y = encoder.transform(y)

    shuffled_dataset = shuffle(pd.concat([pd.DataFrame(X), pd.Series(Y, name='Label')], axis=1), random_state=random_state)
    X_train, X_test, Y_train, Y_test = train_test_split(shuffled_dataset.drop(columns='Label'), shuffled_dataset['Label'], test_size=0.2, random_state=random_state)

    return X_train, X_test, Y_train, Y_test, encoder

def train_rf(X_train, Y_train,X_test,Y_test):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, Y_train)
    rf_predictions = rf_model.predict(X_test)

    # Evaluate Random Forest
    print("Random Forest Metrics:")
    print("Accuracy:", accuracy_score(Y_test, rf_predictions))
    print("Classification Report:")
    target_names = ['Benign', 'Defacement', 'Malware', 'Phishing', 'Spam']
    print(classification_report(Y_test, rf_predictions,target_names=target_names))
    return rf_model

def save_model_objects(encoder, scaler, model, encoder_path="lblenc.npy", scaler_path='scaler.sav', model_path='rf_model.sav'):
    np.save(encoder_path, encoder.classes_)
    pickle.dump(scaler, open(scaler_path, 'wb'))
    pickle.dump(model, open(model_path, 'wb'))

def load_model_objects(encoder_path="lblenc.npy", scaler_path='scaler.sav', model_path='rf_model.sav'):
    encoder = LabelEncoder()
    if os.path.exists(encoder_path):
        encoder.classes_ = np.load(encoder_path, allow_pickle=True)
    else:
        print(f"Error: File not found - {encoder_path}")

    scaler = pickle.load(open(scaler_path, 'rb')) if os.path.exists(scaler_path) else None
    model = pickle.load(open(model_path, 'rb')) if os.path.exists(model_path) else None

    return encoder, scaler, model

def classify_url(url, encoder, scaler, model, order):
    features = d.UrlFeaturizer(url).run()
    test = pd.DataFrame([features[i] for i in order]).replace(True, 1).replace(False, 0).to_numpy().reshape(1, -1)

    scaled_features = scaler.transform(test)
    predicted = model.predict(scaled_features)
    predicted_class = encoder.inverse_transform(predicted)

    return predicted_class[0]

def main():
    X_train, X_test, Y_train, Y_test, encoder = preprocessing()
    rf_model = train_rf(X_train, Y_train,X_test,Y_test)
    save_model_objects(encoder, MinMaxScaler(feature_range=(0, 1)).fit(X_train), rf_model)

    encoder, scaler, rf_model = load_model_objects()

    url = input("Enter a valid URL: ")
    order = ['bodyLength', 'bscr', 'dse', 'dsr', 'entropy', 'hasHttp', 'hasHttps',
             'has_ip', 'numDigits', 'numImages', 'numLinks', 'numParams',
             'numTitles', 'num_%20', 'num_@', 'sbr', 'scriptLength', 'specialChars',
             'sscr', 'urlIsLive', 'urlLength']

    predicted_class = classify_url(url, encoder, scaler, rf_model, order)
    print(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    main()
