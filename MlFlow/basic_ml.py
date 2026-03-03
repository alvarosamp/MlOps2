import os 
import mlflow
import argparse
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_data():
    URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    try:
        df = pd.read_csv(URL, sep=";")
        return df
    except Exception as e:
        raise e
    
def eval_function(actul, pred):
    rmse = np.sqrt(mean_squared_error(actul, pred))
    mae = mean_absolute_error(actul, pred)
    r2 = r2_score(actul, pred)
    return rmse, mae, r2
    
def main(alpha, l1_ratio):
    df = load_data()
    TARGET = 'quality'
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mlflow.set_experiment('ML-Model-1')
    with mlflow.start_run():
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        model=ElasticNet(alpha = alpha, l1_ratio = l1_ratio, random_state = 42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse, mae, r2 = eval_function(y_test, y_pred)
        print(f"RMSE: {rmse}, MAE: {mae}, R2: {r2}")
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)
        mlflow.sklearn.log_model(model, name="model")



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--alpha", "-a", type=float, default=0.5)  # Can still be float
    args.add_argument("--l1-ratio", "-l1", type=float, default=0.5)  # Changed to float and valid default
    parsed_args = args.parse_args()
    main(parsed_args.alpha, parsed_args.l1_ratio)