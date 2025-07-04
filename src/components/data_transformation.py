import sys
import os
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder
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

    def get_data_transformer_object(self, categorical_columns):
        logging.info("Creating pipeline for categorical columns")
        try:
            cat_pipeline = Pipeline(steps=[
                ("onehot", OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys) # type: ignore

    def feature_engineering(self, df: pd.DataFrame):
        try:
            df["total score"] = df["math score"] + df["reading score"] + df["writing score"]
            df["average score"] = df["total score"] / 3
            logging.info("Created 'total score' and 'average score'")
            df.drop(columns=["math score", "reading score", "writing score", "total score"], axis=1, inplace=True)
            logging.info("Dropped original score columns")
            return df
        except Exception as e:
            raise CustomException(e, sys) # type: ignore

    def apply_transformation(self, train_data_path, test_data_path):
        logging.info("Applying transformation to the data")
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info("Loaded training and testing data")

            train_df = self.feature_engineering(train_df)
            test_df = self.feature_engineering(test_df)

            target_name = "average score"
            X_train = train_df.drop(columns=[target_name])
            y_train = train_df[target_name]

            X_test = test_df.drop(columns=[target_name])
            y_test = test_df[target_name]

            categorical_features = X_train.select_dtypes(include="object").columns.tolist()
            preprocessor = self.get_data_transformer_object(categorical_features)

            X_train_encoded = preprocessor.fit_transform(X_train)
            X_test_encoded = preprocessor.transform(X_test)

            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_path), exist_ok=True)
            joblib.dump(preprocessor, self.data_transformation_config.preprocessor_path)
            logging.info("Saved preprocessor object")

            return X_train_encoded, y_train, X_test_encoded, y_test,self.data_transformation_config.preprocessor_path

        except Exception as e:
            raise CustomException(e, sys) # type: ignore 
