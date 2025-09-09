import pytest
import os
import sys
from pathlib import Path
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))
print(PACKAGE_ROOT)

from PackagingMlModels.prediction_model.config import config
from PackagingMlModels.prediction_model.processing.data_handling import load_dataset
from PackagingMlModels.prediction_model.predict import generate_predictions

#Output from predict script not nul
#Output from predict script is str data type
# The output is Y an example data

#Fixtures -> functions before test function --> ensure single_prediction

@pytest.fixture
def single_prediction():
    test_dataset = load_dataset(config.TEST_FILE)
    single_row = test_dataset[:1]
    result = generate_predictions(single_row)
    return result

def test_single_pred_not_none(single_prediction): # output is not none
    assert single_prediction is not None

def test_single_pred_str_type(single_prediction): # data type is string
    assert isinstance(single_prediction.get('prediction')[0],str)

def test_single_pred_validate(single_prediction): # check the output is Y or N
    assert single_prediction.get('prediction')[0] in ('Y', 'N')