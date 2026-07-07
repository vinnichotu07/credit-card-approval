from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model and encoders safely
try:
    model = pickle.load(open('model.pkl', 'rb'))
    encoders = pickle.load(open('encoders.pkl', 'rb'))
except FileNotFoundError:
    print("Warning: Ensure 'model.pkl' and 'encoders.pkl' are placed in the same execution folder.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # 1. Parse layout values into target strings for label encoders
            gender = "FEMALE" if request.form.get('GENDER') == "1" else "MALE"
            own_car = "YES" if request.form.get('OWN_CAR') == "1" else "NO"
            own_realty = "YES" if request.form.get('OWN_REALTY') == "1" else "NO"

            income_type = request.form.get('INCOME_TYPE')
            education = request.form.get('EDUCATION')
            family_status = request.form.get('FAMILY_STATUS')
            housing_type = request.form.get('HOUSING_TYPE')
            
            income = float(request.form.get('INCOME', 0))
            days_birth = float(request.form.get('DAYS_BIRTH', 0))
            days_employed = float(request.form.get('DAYS_EMPLOYED', 0))
            family_members = int(request.form.get('FAMILY_MEMBERS', 1))
            emi_paid = int(request.form.get('EMI_PAID', 0))
            emi_pastdue = int(request.form.get('EMI_PASTDUE', 0))
            no_loans = int(request.form.get('NO_LOANS', 0))

            # Helper decoder function
            def get_encoded_value(key_options, raw_value):
                for option in key_options:
                    if option in encoders:
                        return encoders[option].transform([raw_value])[0]
                return 0

            # 2. Extract mapped categories
            gender_encoded = get_encoded_value(['CODE_GENDER', 'GENDER'], gender)
            own_car_encoded = get_encoded_value(['FLAG_OWN_CAR', 'OWN_CAR'], own_car)
            own_realty_encoded = get_encoded_value(['FLAG_OWN_REALTY', 'OWN_REALTY'], own_realty)
            
            income_type_encoded = get_encoded_value(['NAME_INCOME_TYPE', 'INCOME_TYPE'], income_type)
            education_encoded = get_encoded_value(['NAME_EDUCATION_TYPE', 'EDUCATION'], education)
            family_status_encoded = get_encoded_value(['NAME_FAMILY_STATUS', 'FAMILY_STATUS'], family_status)
            housing_type_encoded = get_encoded_value(['NAME_HOUSING_TYPE', 'HOUSING_TYPE'], housing_type)

            # 3. Assemble inputs (14 fields)
            features = [
                gender_encoded, own_car_encoded, own_realty_encoded, income,
                income_type_encoded, education_encoded, family_status_encoded, housing_type_encoded,
                days_birth, days_employed, family_members, emi_paid, emi_pastdue, no_loans
            ]
            
            # 4. Fill required 15th column structural constraint padding
            if len(features) == 14:
                features.append(0)
            
            # 5. Run classification inference execution
            prediction = model.predict([np.array(features)])
            
            if prediction[0] == 1:
                return jsonify(status="approved", prediction_text="✨ Application Approved! Profile fits preferred validation metrics.")
            else:
                return jsonify(status="rejected", prediction_text="❌ Application Declined. Profile falls outside required baseline thresholds.")
            
        except Exception as e:
            return jsonify(status="error", prediction_text=f"Processing Interruption: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)