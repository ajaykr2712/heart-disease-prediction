from flask import Flask, render_template, request
import pickle
import numpy as np

model_filename = 'heart-disease-prediction-kmeans-model.pkl'
heart_disease_model = pickle.load(open(model_filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        patient_age = int(request.form['age'])
        patient_sex = request.form.get('sex')
        chest_pain_type = request.form.get('cp')
        resting_blood_pressure = int(request.form['trestbps'])
        cholesterol_level = int(request.form['chol'])
        fasting_blood_sugar = request.form.get('fbs')
        resting_ecg = int(request.form['restecg'])
        max_heart_rate_achieved = int(request.form['thalach'])
        exercise_induced_angina = request.form.get('exang')
        st_depression = float(request.form['oldpeak'])
        st_slope = request.form.get('slope')
        number_of_major_vessels = int(request.form['ca'])
        thalassemia = request.form.get('thal')
        
        input_features = np.array([[patient_age, patient_sex, chest_pain_type, resting_blood_pressure, cholesterol_level, fasting_blood_sugar, resting_ecg, max_heart_rate_achieved, exercise_induced_angina, st_depression, st_slope, number_of_major_vessels, thalassemia]])
        prediction_result = heart_disease_model.predict(input_features)
        
        return render_template('result.html', prediction=prediction_result)

if __name__ == '__main__':
    app.run(debug=True)
