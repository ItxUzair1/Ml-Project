import sys
import os
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import joblib
from src.utils import load_model

@dataclass
class PredictPipelineConfig:
    model_path = os.path.join("models", "trained_model.pkl")
    preprocessor_path= os.path.join("models", "preprocessor.pkl")

class PredictPipeline:
    def __init__(self):
        self.predict_pipeline_config = PredictPipelineConfig()
        self.model = None
        self.preprocessor = None
    
    def load_model_and_preprocessor(self):
        try:
            logging.info("Loading model and preprocessor")
            self.model = load_model(self.predict_pipeline_config.model_path)
            self.preprocessor = load_model(self.predict_pipeline_config.preprocessor_path)
            logging.info("Model and preprocessor loaded successfully")
        except Exception as e:
            raise CustomException(e, sys) # type: ignore
        
    
    def predict(self,input_data):
        try:
            transformed_data=self.preprocessor.transform(input_data) # type: ignore
            logging.info("Data transformed using preprocessor")

            predictions=self.model.predict(transformed_data) # type: ignore
            logging.info("Predictions made successfully")

            return predictions
        except Exception as e:
            raise CustomException(e, sys) # type: ignore
 
        
    