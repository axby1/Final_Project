from sys import displayhook
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

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve,auc

from io import BytesIO
import base64
from IPython.display import HTML

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
    #new //expands the url and sends the expanded url through to the UrlFeaturizer()
    session = requests.Session()  # Use a session for connection pooling
    try:
        response = session.head(url, allow_redirects=True, timeout=5)
        url = response.url
    except requests.RequestException as e:
        print(f"Error expanding URL {url}: {e}")
    # / new
    finally:
        #print("new: ",url)
        features = d.UrlFeaturizer(url).run()
        test = pd.DataFrame([features[i] for i in order]).replace(True, 1).replace(False, 0).to_numpy().reshape(1, -1)

        scaled_features = scaler.transform(test)
        predicted = model.predict(scaled_features)
        predicted_class = encoder.inverse_transform(predicted)

        return predicted_class[0]


def visualize_confusion_matrix(y_true, y_pred, target_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    for i in range(len(target_names)):
        for j in range(len(target_names)):
            plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.show()

def visualize_roc_curve(model, X_test, y_test):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    y_score = model.predict_proba(X_test)

    for i in range(model.n_classes_):
        fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    colors = ['b', 'g', 'r', 'c', 'm']
    for i, color in zip(range(model.n_classes_), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class')
    plt.legend(loc="lower right")
    plt.show()



def plot_feature_importance(model, feature_names,X_train):
    # Get feature importances from the model
    importances = model.feature_importances_
    
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names based on sorted feature importances
    sorted_feature_names = [feature_names[i] for i in indices]

    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(X_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train.shape[1]), sorted_feature_names, rotation=90)
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()




def classification_report_image(y_true, y_pred, target_names):
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    # Plot classification report as a table
    plt.figure(figsize=(10, 6))
    plt.axis('off')
    plt.table(cellText=report_df.values,
              colLabels=report_df.columns,
              rowLabels=report_df.index,
              loc='center')
    
    # Save the plot as an image
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Encode the image as a base64 string
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Create HTML code for displaying the image in a popup
    html = f'<img src="data:image/png;base64,{image_base64}" />'
    
    return html    




def main(url):
    X_train, X_test, Y_train, Y_test, encoder = preprocessing()
    rf_model = train_rf(X_train, Y_train,X_test,Y_test)
    save_model_objects(encoder, MinMaxScaler(feature_range=(0, 1)).fit(X_train), rf_model)

    encoder, scaler, rf_model = load_model_objects()

    #url = input("Enter a valid URL: ")
    order = ['bodyLength', 'bscr', 'dse', 'dsr', 'entropy', 'hasHttp', 'hasHttps',
             'has_ip', 'numDigits', 'numImages', 'numLinks', 'numParams',
             'numTitles', 'num_%20', 'num_@', 'sbr', 'scriptLength', 'specialChars',
             'sscr', 'urlIsLive', 'urlLength']

    predicted_class = classify_url(url, encoder, scaler, rf_model, order)
    print(f"Predicted class: {predicted_class}")

    visualize_confusion_matrix(Y_test, rf_model.predict(X_test), target_names=['Benign', 'Defacement', 'Malware', 'Phishing', 'Spam'])
    visualize_roc_curve(rf_model, X_test, Y_test)
    plot_feature_importance(rf_model, order,X_train)
    y_pred=rf_model.predict(X_test)
    classification_report_image_html = classification_report_image(Y_test, y_pred, target_names=['Benign', 'Defacement', 'Malware', 'Phishing', 'Spam'])
    
    print(f"Predicted class: {predicted_class}")

    # Display the classification report image
    print("Classification Report:")
    print(classification_report_image_html)


    return predicted_class
    
    

if __name__ == "__main__":
    main()








