from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error,root_mean_squared_error
from src.exception import CustomException
from src.logger import logging
import sys
from sklearn.model_selection import RandomizedSearchCV
import joblib

def evaluate_model(X_train, y_train, X_test, y_test, models: dict,params: dict):
    try:
        report = {}
        best_models = {}
        for model_name, model in models.items():
            params_grid=params.get(model_name,{})

            if params_grid:
                logging.info(f"Performing Randomized Search for {model_name}")
                grid_model = RandomizedSearchCV(model, param_distributions=params_grid, n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=-1)
                grid_model.fit(X_train, y_train)
                best_model = grid_model.best_estimator_
                logging.info(f"Best parameters for {model_name}: {grid_model.best_params_}")
            else:
                 model.fit(X_train, y_train)
                 best_model = model

                 
            y_pred = best_model.predict(X_test) # type: ignore

           

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)
            

            report[model_name] = {
                "R2 Score": r2,
                "Mean Absolute Error": mae,
                "Mean Squared Error": mse,
                "Root Mean Squared Error": rmse
            }
            best_models[model_name] = best_model  # Save model

        return report, best_models


    except Exception as e:
        raise CustomException(e, sys) # type: ignore
    

def load_model(file_path: str):
    try:
        logging.info(f"Loading object from {file_path}")
        return joblib.load(file_path)
    except Exception as e:
        raise CustomException(e, sys) # type: ignore
