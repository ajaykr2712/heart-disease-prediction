from flask import Flask, render_template, request
import pickle
import numpy as np

model_filename = 'heart-disease-prediction-kmeans-model.pkl'
loaded_model = pickle.load(open(model_filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home_page():
	return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def make_prediction():
    if request.method == 'POST':

        patient_age = int(request.form['patient_age'])
        patient_sex = request.form.get('patient_sex')
        chest_pain_type = request.form.get('chest_pain_type')
        resting_blood_pressure = int(request.form['resting_blood_pressure'])
        serum_cholesterol = int(request.form['serum_cholesterol'])
        fasting_blood_sugar = request.form.get('fasting_blood_sugar')
        resting_ecg = int(request.form['resting_ecg'])
        max_heart_rate = int(request.form['max_heart_rate'])
        exercise_induced_angina = request.form.get('exercise_induced_angina')
        st_depression = float(request.form['st_depression'])
        st_slope = request.form.get('st_slope')
        num_major_vessels = int(request.form['num_major_vessels'])
        thalassemia = request.form.get('thalassemia')
        
        input_data = np.array([[patient_age, patient_sex, chest_pain_type, resting_blood_pressure, 
                                serum_cholesterol, fasting_blood_sugar, resting_ecg, max_heart_rate, 
                                exercise_induced_angina, st_depression, st_slope, num_major_vessels, thalassemia]])
        prediction_result = loaded_model.predict(input_data)
        
        return render_template('result.html', prediction=prediction_result)
        
if __name__ == '__main__':
	app.run(debug=True)
