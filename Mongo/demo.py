from src.logger import logging
from src.exception import MyException
import sys
from src.pipline.training_pipeline import TrainingPipeline
if __name__ == "__main__":
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.start_data_ingestion()
    except Exception as e:
        raise MyException(e,sys) from e