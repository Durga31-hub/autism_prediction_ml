from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS 
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load the model and encoders
MODEL_PATH = 'models/autism_model.joblib'
ENCODERS_PATH = 'models/encoders.joblib'
FEATURES_PATH = 'models/feature_names.joblib'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found. Run train_model.py first.")

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)
feature_names = joblib.load(FEATURES_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Prepare input data
        # Data should come as a dictionary with feature names as keys
        input_df = pd.DataFrame([data])
        
        # Ensure column order matches training
        input_df = input_df[feature_names]
        
        # Encode categorical features if they are passed as strings
        for col, encoder in encoders.items():
            if col in input_df.columns and col != 'class':
                # Convert to string if it's not already
                val = str(input_df[col].iloc[0]).lower()
                # Use encoder to transform
                try:
                    input_df[col] = encoder.transform([val])
                except ValueError:
                    # If value not seen in training, default to first class or handle error
                    input_df[col] = 0 

        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] # Probability of YES
        
        # Invert target encoding to get human-readable result
        target_encoder = encoders['class']
        result_label = target_encoder.inverse_transform([prediction])[0]

        import shap
        
        # Initialize SHAP explainer if not exists
        if not hasattr(app, 'explainer'):
            app.explainer = shap.TreeExplainer(model)
            
        shap_values = app.explainer.shap_values(input_df)

        # For RandomForest, shap_values is typically a list of arrays (one per class) 
        # or a 3D array (n_samples, n_features, n_classes) in newer versions.
        class_idx = model.predict(input_df)[0]
        
        if isinstance(shap_values, list):
            sv_single = shap_values[class_idx][0]
        else:
            if len(shap_values.shape) == 3:
                sv_single = shap_values[0, :, class_idx]
            elif len(shap_values.shape) == 2:
                # If only one class is returned
                sv_single = shap_values[0]
            else:
                sv_single = shap_values

        feature_names_readable = {
            'a1': 'Social Interaction', 'a2': 'Shared Attention', 'a3': 'Social Communication',
            'a4': 'Imagination', 'a5': 'Repetitive Behavior', 'a6': 'Social Interest',
            'a7': 'Attention to Detail', 'a8': 'Communication', 'a9': 'Play Patterns',
            'a10': 'Understanding Others', 'sex': 'Gender', 'jaundice': 'Jaundice History', 'family_asd': 'Family History of ASD'
        }
        
        # We want to show top features contributing to THIS specific prediction reaching its conclusion
        impacts = []
        for i in range(len(feature_names)):
            feat_name = feature_names_readable.get(feature_names[i], feature_names[i])
            val = sv_single[i]
            # Extra safety, if it's still an array, take the first element
            if hasattr(val, '__len__'):
                val = val[0]
            
            impacts.append({
                "feature": feat_name,
                "impact": float(val) # how much it pushed towards the predicted class
            })
            
        # Sort by absolute impact to find "top contributing factors"
        sorted_impacts = sorted(impacts, key=lambda x: abs(x["impact"]), reverse=True)[:5]

        return jsonify({
            'success': True,
            'prediction': result_label,
            'probability': float(probability),
            'factors': sorted_impacts,
            'message': 'Autism Screening Completed'
        })


    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
