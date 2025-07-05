
def get_model_params():
    params = {
    "Linear Regression": {},

    "Lasso Regression": {
        "alpha": [0.01, 0.1, 1, 10],
        "max_iter": [1000, 5000],
        "selection": ["cyclic", "random"]
    },

    "Ridge Regression": {
        "alpha": [0.01, 0.1, 1, 10],
        "solver": ["auto", "svd", "cholesky", "lsqr"]
    },

    "KNeighbors Regressor": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "metric": ["minkowski", "euclidean", "manhattan"]
    },

    "Decision Tree Regressor": {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["squared_error", "friedman_mse"]
    },

    "Random Forest Regressor": {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "bootstrap": [True, False]
    },

    "AdaBoost Regressor": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 1]
    },

    "Gradient Boosting Regressor": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0]
    },

    "Support Vector Regressor": {
        "kernel": ["rbf", "linear"],
        "C": [0.1, 1.0, 10],
        "gamma": ["scale", "auto"],
        "epsilon": [0.1, 0.2, 0.3]
    },

    "CatBoost Regressor": {
        "iterations": [500, 1000],
        "learning_rate": [0.01, 0.05, 0.1],
        "depth": [4, 6, 8],
        "l2_leaf_reg": [1, 3, 5],
        "verbose": [False]
    },

    "XGBoost Regressor": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }
}
    return params