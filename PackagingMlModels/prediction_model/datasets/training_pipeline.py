import pandas as pd 
import numpy as  np
from pathlib import Path
import os
import sys
import joblib

try:
    PACKAGE_ROOT = Path(__file__).resolve().parent.parent
except Exception:
    PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent


from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset, save_pipeline, separete_data, split_data    
import prediction_model.processing.preprocessing as pp 
import prediction_model.pipeline as pipe 

def perform_training():
    dataset = load_dataset(config.FILE_NAME)
    X, y = separete_data(dataset)
    y = y.apply(lambda x: 1 if x.strip() == 'Approved' else 0) # Remove o espaço e compara a palavra
    X_train, X_test, y_train, y_test = split_data(X, y)
    test_data = X_test.copy()
    test_data[config.TARGET] = y_test
    test_data.to_csv(os.path.join(config.DATAPATH, config.TEST_FILE))
    pipe.classification_pipeline.fit(X_train, y_train)
    save_pipeline(pipe.classification_pipeline, config.PIPELINE_FILE)

if __name__ == '__main__':
    perform_training()