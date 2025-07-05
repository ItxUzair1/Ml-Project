from flask import Flask, request, jsonify,render_template
from src.pipeline.predict_pipeline import PredictPipeline
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import os


app= Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        logging.info(f"Received input data: {data}")

        # Convert input data to DataFrame
        input_data = pd.DataFrame([data])
        logging.info(f"Input DataFrame: {input_data}")

        input_data["writing score"] = input_data["writing score"].astype(int)
        input_data["reading score"] = input_data["reading score"].astype(int)


        # Initialize prediction pipeline
        predict_pipeline = PredictPipeline()
        predict_pipeline.load_model_and_preprocessor()

        # Make predictions
        predictions = predict_pipeline.predict(input_data)
        logging.info(f"Predictions: {predictions}")

        return jsonify({'prediction': predictions[0]})
    except Exception as e:
        raise CustomException(e) # type: ignore

if __name__ == '__main__':
    try:
        port = int(os.environ.get("PORT", 5000))
        app.run(host='0.0.0.0', port=port)
    except Exception as e:
        logging.error(f"Error starting the Flask app: {e}")
        raise CustomException(e) # type: ignore