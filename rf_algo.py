import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle
import data_creation_v3 as d
import os
import numpy as np

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

def main(url):
    encoder, scaler, rf_model = load_model_objects()

    order = ['bodyLength', 'bscr', 'dse', 'dsr', 'entropy', 'hasHttp', 'hasHttps',
             'has_ip', 'numDigits', 'numImages', 'numLinks', 'numParams',
             'numTitles', 'num_%20', 'num_@', 'sbr', 'scriptLength', 'specialChars',
             'sscr', 'urlIsLive', 'urlLength']

    predicted_class = classify_url(url, encoder, scaler, rf_model, order)
    print(f"Predicted class: {predicted_class}")
    return predicted_class

if __name__ == "__main__":
    main()
