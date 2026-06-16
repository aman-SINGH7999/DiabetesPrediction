from flask import Flask, render_template, request
from joblib import load

model = load('logistic_modal.joblib')
scaler = load('scaler.joblib')

app = Flask(__name__)


@app.route('/')
def home():
  return render_template('index.html')


@app.route('/predict', methods=['post'])
def predict():
  pregnancies = float(request.form['Pregnancies'])
  glucose = float(request.form['Glucose'])
  bloodpressure = float(request.form['BloodPressure'])
  skinthickness = float(request.form['SkinThickness'])
  insulin = float(request.form['Insulin'])
  bmi = float(request.form['BMI'])
  dpf = float(request.form['DiabetesPedigreeFunction'])
  age = float(request.form['Age'])

  data = [[
      pregnancies,
      glucose,
      bloodpressure,
      skinthickness,
      insulin,
      bmi,
      dpf,
      age
    ]]
  
  scaled_data = scaler.transform(data)
  prediction = model.predict(scaled_data)
  probabilities = model.predict_proba(scaled_data)[0]

  if prediction[0] == 1:
    result = "Diabetic"
    confidence = round(probabilities[1] * 100, 2)
  else:
    result = "Non Diabetic"
    confidence = round(probabilities[0] * 100, 2)

  return render_template(
    'index.html', 
    result=result,
    confidence=confidence,
    Pregnancies=pregnancies,
    Glucose=glucose,
    BloodPressure=bloodpressure,
    SkinThickness=skinthickness,
    Insulin=insulin,
    BMI=bmi,
    DiabetesPedigreeFunction=dpf,
    Age=age
  )

@app.route('/clear')
def clear():
    return render_template('index.html')

if __name__ == '__main__':
  app.run(debug=True)