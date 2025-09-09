import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
import sys
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))
from prediction_model.config import config
from prediction_model.processing.data_handling import load_pipeline, load_dataset

classification_pipeline = load_pipeline(config.PIPELINE_FILE)

def generate_predictions(input_data=None):
    if input_data is None:
        test_data = load_dataset(config.TEST_FILE)
    else:
        test_data = input_data
    # Drop target column if present
    if config.TARGET in test_data.columns:
        test_data = test_data.drop(columns=[config.TARGET])
    pred = classification_pipeline.predict(test_data)
    output = np.where(pred == 1, 'Y', 'N')  # Match test expectation: 'Y'/'N'
    # Return as dict to match test usage: single_prediction.get('prediction')[0]
    return {'prediction': output.tolist()}

if __name__=='__main__':
    generate_predictions()