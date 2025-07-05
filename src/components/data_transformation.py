import sys
import os
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib


@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join("models", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, categorical_columns,numeric_columns):
        logging.info("Creating pipeline for categorical columns")
        try:
            cat_pipeline = Pipeline(steps=[
                ("onehot", OneHotEncoder(handle_unknown='ignore'))
            ])
            logging.info("Pipeline for categorical columns created successfully")

            num_pipeline = Pipeline(steps=[
                ("scaler", StandardScaler())
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", cat_pipeline, categorical_columns),
                    ("num", num_pipeline, numeric_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys) # type: ignore

    def apply_transformation(self, train_data_path, test_data_path):
        logging.info("Applying transformation to the data")
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info("Loaded training and testing data")
            logging.info(f"Train Data Shape: {train_df.shape}, Test Data Shape: {test_df.shape}")

            target_name = "math score"
            X_train = train_df.drop(columns=[target_name])
            y_train = train_df[target_name]

            X_test = test_df.drop(columns=[target_name])
            y_test = test_df[target_name]

            categorical_features = X_train.select_dtypes(include="object").columns.tolist()
            numeric_features = X_train.select_dtypes(exclude="object").columns.tolist()
            preprocessor = self.get_data_transformer_object(categorical_features,numeric_features)

            X_train_encoded = preprocessor.fit_transform(X_train)
            X_test_encoded = preprocessor.transform(X_test)

            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_path), exist_ok=True)
            joblib.dump(preprocessor, self.data_transformation_config.preprocessor_path)
            logging.info("Saved preprocessor object")

            return X_train_encoded, y_train, X_test_encoded, y_test,self.data_transformation_config.preprocessor_path

        except Exception as e:
            raise CustomException(e, sys) # type: ignore 
