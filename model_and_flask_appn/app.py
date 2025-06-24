from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

with open('c_card_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Prediction', methods=['GET'])
def prediction():
    return render_template('index1.html')

# Redirect to home
@app.route('/Home', methods=['GET'])
def my_home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_feature = [float(x) for x in request.form.values()]
        features_values = np.array([input_feature])

        feature_name = [
            'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'AMT_INCOME_TOTAL',
            'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
            'NAME_HOUSING_TYPE', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
            'CNT_FAM_MEMBERS', 'paid_off','#_of_pastdues','no_loan'
        ]

        x = pd.DataFrame(features_values, columns=feature_name)

        pred = model.predict(x)[0]
        prediction = "Eligible" if pred == 1 else "Not Eligible"

        print("Input features:", x.to_dict(orient='records')[0])
        print("Prediction result:", prediction)

        return render_template('result.html', prediction=prediction)
    
    except Exception as e:
        print("Error during prediction:", str(e))
        return render_template('error.html', error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
