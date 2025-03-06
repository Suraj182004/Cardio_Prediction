from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
try:
    with open('random_forest_model.pkl', 'rb') as file:
        model_data = pickle.load(file)
    
    rf_model = model_data['model']
    features = model_data['features']
    
    print("Model loaded successfully!")
    print(f"Features: {features}")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        bmi = float(request.form['bmi'])
        physical_health = float(request.form['physical_health'])
        mental_health = float(request.form['mental_health'])
        sleep_time = float(request.form['sleep_time'])
        sex = request.form['sex']
        gen_health = request.form['gen_health']
        physical_activity = request.form['physical_activity']
        alcohol = request.form['alcohol']
        stroke = request.form['stroke']
        diff_walking = request.form['diff_walking']
        diabetic = request.form['diabetic']
        asthma = request.form['asthma']
        kidney_disease = request.form['kidney_disease']
        skin_cancer = request.form['skin_cancer']
        
        # Print received data for debugging
        print("\nReceived form data:")
        print(f"BMI: {bmi}")
        print(f"Physical Health: {physical_health}")
        print(f"Mental Health: {mental_health}")
        print(f"Sleep Time: {sleep_time}")
        print(f"Sex: {sex}")
        print(f"General Health: {gen_health}")
        print(f"Physical Activity: {physical_activity}")
        print(f"Alcohol: {alcohol}")
        print(f"Stroke: {stroke}")
        print(f"Difficulty Walking: {diff_walking}")
        print(f"Diabetic: {diabetic}")
        print(f"Asthma: {asthma}")
        print(f"Kidney Disease: {kidney_disease}")
        print(f"Skin Cancer: {skin_cancer}")
        
        # Create feature array using the same encoding as training
        features_array = np.zeros(len(features))
        
        # Numeric features
        features_array[features.index('BMI')] = bmi
        features_array[features.index('PhysicalHealthDays')] = physical_health
        features_array[features.index('MentalHealthDays')] = mental_health
        features_array[features.index('SleepHours')] = sleep_time
        
        # Binary encode Sex
        features_array[features.index('Sex_Male')] = 1 if sex == 'Male' else 0
        
        # Binary encode General Health
        health_mapping = {
            'Excellent': 'Health_Excellent',
            'Very good': 'Health_VeryGood',
            'Good': 'Health_Good',
            'Fair': 'Health_Fair',
            'Poor': 'Health_Poor'
        }
        for health_type, feature_name in health_mapping.items():
            if feature_name in features:
                features_array[features.index(feature_name)] = 1 if gen_health == health_type else 0
        
        # Binary encode Yes/No features
        features_array[features.index('PhysicalActivities')] = 1 if physical_activity == 'Yes' else 0
        features_array[features.index('AlcoholDrinkers')] = 1 if alcohol == 'Yes' else 0
        features_array[features.index('HadStroke')] = 1 if stroke == 'Yes' else 0
        features_array[features.index('DifficultyWalking')] = 1 if diff_walking == 'Yes' else 0
        features_array[features.index('HasDiabetes')] = 1 if diabetic == 'Yes' else 0
        features_array[features.index('HadAsthma')] = 1 if asthma == 'Yes' else 0
        features_array[features.index('HadKidneyDisease')] = 1 if kidney_disease == 'Yes' else 0
        features_array[features.index('HadSkinCancer')] = 1 if skin_cancer == 'Yes' else 0
        
        # Reshape for prediction
        features_array = features_array.reshape(1, -1)
        
        # Print feature array for debugging
        print("\nFeature array:")
        for i, feature in enumerate(features):
            print(f"{feature}: {features_array[0, i]}")
        
        # Make prediction
        prediction = rf_model.predict(features_array)[0]
        probability = rf_model.predict_proba(features_array)[0]
        
        # Get probability of heart attack
        heart_attack_prob = probability[1]  # Probability of class 1 (heart attack)
        
        print(f"\nRaw prediction: {prediction} (1=Yes, 0=No)")
        print(f"Raw probability: {heart_attack_prob:.4f}")
        print(f"Full probability array: {probability}")
        
        # Calculate risk score based on known risk factors
        risk_score = 0
        
        # Major risk factors
        if bmi >= 30:  # Obesity
            risk_score += 2
            print("Risk factor: Obesity")
        
        if physical_health >= 15:  # Poor physical health
            risk_score += 2
            print("Risk factor: Poor physical health")
        
        if stroke == 'Yes':  # Previous stroke
            risk_score += 3
            print("Risk factor: Previous stroke")
        
        if diabetic == 'Yes':  # Diabetes
            risk_score += 2
            print("Risk factor: Diabetes")
        
        if gen_health == 'Poor':  # Poor general health
            risk_score += 2
            print("Risk factor: Poor general health")
        
        if diff_walking == 'Yes':  # Difficulty walking
            risk_score += 1
            print("Risk factor: Difficulty walking")
        
        if kidney_disease == 'Yes':  # Kidney disease
            risk_score += 2
            print("Risk factor: Kidney disease")
        
        if physical_activity == 'No':  # No physical activity
            risk_score += 1
            print("Risk factor: No physical activity")
        
        if sleep_time < 6:  # Poor sleep
            risk_score += 1
            print("Risk factor: Poor sleep")
        
        print(f"Calculated risk score: {risk_score}")
        
        # Adjust probability based on risk score
        if risk_score >= 10:
            heart_attack_prob = max(heart_attack_prob, 0.8)
            print("High risk score detected - setting probability to at least 0.8")
        elif risk_score >= 7:
            heart_attack_prob = max(heart_attack_prob, 0.6)
            print("Moderate-high risk score detected - setting probability to at least 0.6")
        elif risk_score >= 5:
            heart_attack_prob = max(heart_attack_prob, 0.4)
            print("Moderate risk score detected - setting probability to at least 0.4")
        
        print(f"Final adjusted probability: {heart_attack_prob:.4f}")
        
        # Determine risk level and message
        if heart_attack_prob < 0.3:
            risk = "Lower"
            message = "Your risk factors suggest a lower probability of heart disease. However, maintaining a healthy lifestyle is still important."
        elif heart_attack_prob < 0.6:
            risk = "Moderate" 
            message = "Your risk factors suggest a moderate probability of heart disease. Consider consulting a healthcare provider."
        else:
            risk = "Higher"
            message = "Your risk factors suggest a higher probability of heart disease. Please consult a healthcare provider."
            
        return render_template('result.html', 
                             prediction=risk,
                             message=message,
                             probability=round(heart_attack_prob * 100, 1))
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)