import pandas as pd 
import numpy as  np
import pathlib
from pathlib import Path
import os
import sys
import joblib

try:
    PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent
except Exception:
    PACKAGE_ROOT = pathlib.Path(os.path.abspath(os.path.dirname(__file__))).parent

sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset, save_pipeline, separate_data
import prediction_model.processing.preprocessing as pp 
import prediction_model.pipeline as pipe 

def perform_training():
    dataset = load_dataset(config.FILE_NAME)
    X_train, X_test, y_train, y_test = separate_data(dataset)
    y_train = y_train.apply(lambda x: 1 if x.strip() == 'Approved' else 0)
    y_test = y_test.apply(lambda x: 1 if x.strip() == 'Approved' else 0)
    test_data = X_test.copy()
    test_data[config.TARGET] = y_test
    test_data.to_csv(os.path.join(config.DATAPATH, config.TEST_FILE), index=False)
    pipe.classification_pipeline.fit(X_train, y_train)
    save_pipeline(pipe.classification_pipeline, config.PIPELINE_FILE)

if __name__ == '__main__':
    perform_training()