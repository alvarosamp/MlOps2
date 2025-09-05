import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
import sys
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))
from prediction_model.config import config
from prediction_model.processing.data_handling import load_pipeline,load_dataset,separete_data

classification_pipeline = load_pipeline(config.PIPELINE_FILE)

def generate_predictions():
    test_data = load_dataset(config.TEST_FILE)
    X,y = separete_data(test_data)
    pred = classification_pipeline.predict(X)
    output = np.where(pred ==1, 'Approved', 'Not approved')
    print(output)
    return output

if __name__=='__main__':
    generate_predictions()