import numpy as np
import pandas as pd
import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
import joblib
from src.utils import evaluate_model
from src.config.params import get_model_params


@dataclass
class ModelTrainConfig:
    trained_model_file_path = os.path.join("models", "trained_model.pkl")

class ModelTrain:
    def __init__(self):
        self.model_train_config = ModelTrainConfig()

    def train_model(self, X_train, y_train, X_test, y_test):
        try: 
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso Regression": Lasso(),
                "Ridge Regression": Ridge(),
                "KNeighbors Regressor": KNeighborsRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "Support Vector Regressor": SVR(),
                "CatBoost Regressor": CatBoostRegressor(verbose=0),
                "XGBoost Regressor": XGBRegressor()
            }

            params: dict = get_model_params()

            report, best_models = evaluate_model(X_train, y_train, X_test, y_test, models, params)

            best_model_name = max(report, key=lambda x: report[x]["R2 Score"])
            best_model = best_models[best_model_name] 

    
            os.makedirs(os.path.dirname(self.model_train_config.trained_model_file_path), exist_ok=True)
            joblib.dump(best_model, self.model_train_config.trained_model_file_path)
            logging.info(f"Best model '{best_model_name}' saved at {self.model_train_config.trained_model_file_path}")

            return best_model_name, report

        except Exception as e:
            raise CustomException(e, sys) # type: ignore

