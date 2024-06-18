from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import mode

app = Flask(__name__)

# Load and preprocess the dataset
DATA_PATH = "datasets/Training.csv"
data = pd.read_csv(DATA_PATH).dropna(axis=1)

# Encoding the target value into numerical
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Initialize and train models
final_rf_model = RandomForestClassifier(random_state=18)
final_nb_model = GaussianNB()
final_svm_model = SVC()

final_rf_model.fit(X, y)
final_nb_model.fit(X, y)
final_svm_model.fit(X, y)

# Symptom index dictionary
symptoms = X.columns.values
symptom_index = {symptom: index for index, symptom in enumerate(symptoms)}

data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_
}


def predictDisease(symptoms):
    symptoms = symptoms.split(",")

    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        try:
            index = data_dict["symptom_index"][symptom.strip()]
            input_data[index] = 1
        except KeyError:
            print(f"Warning: Symptom '{symptom}' not found in dataset")
            continue

    input_data = np.array(input_data).reshape(1, -1)

    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]

    predictions = [rf_prediction, nb_prediction, svm_prediction]
    final_prediction = max(set(predictions), key=predictions.count)

    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction
    }
    return predictions


'''def predictDisease(symptoms):
    symptoms = symptoms.split(",")

    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        try:
            index = data_dict["symptom_index"][symptom.strip()]
            input_data[index] = 1
        except KeyError:
            print(f"Warning: Symptom '{symptom}' not found in dataset")
            continue

    input_data = np.array(input_data).reshape(1, -1)

    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
    final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]

    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction
    }
    return predictions
'''
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        symptoms = request.form['symptoms']
        prediction = predictDisease(symptoms)
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
