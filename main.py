from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_train import ModelTrain


if __name__ == "__main__":
    obj=DataIngestion()
    train_path,test_path=obj.initate_data_ingestion()
    dt=DataTransformation()
    X_train,y_train,X_test,y_test,_=dt.apply_transformation(train_path,test_path)
    model_train = ModelTrain()
    best_model_name, report = model_train.train_model(X_train, y_train, X_test, y_test)
    print(f"Best Model: {best_model_name}")
    print("Model Evaluation Report:", report.get(best_model_name, {}))