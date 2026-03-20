import sys
import pandas as pd
import numpy as np
from typing import Optimal
from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import DATABASE_NAME
from src.exception import MyException

class Proj1Data:
    '''
    A class to export MongoDB records as a pandas DataFrame
    '''
    def __init__(self) -> None:
        try:
            self.mongo_client = MongoDBClient(database_name= DATABASE_NAME)
        except Exception as e:
            raise MyException(e, sys)
        
    def export_collection_as_dataframe(self, collection_name: str, database_name: Optimal[str] = None) -> pd.DataFrame:
        '''
        Exports an entire MongoDB collecetion as a pandas DataFrame. 
        
        Parameters:
        collection_name (str): The name of the MongoDB collection to export.
        database_name (str, optional): The name of the MongoDB database to use. If not provided, the default database from the MongoDBClient will be used.
        
        retrurn : pd.DataFrame: A pandas DataFrame containing the data from the specified MongoDB collection.
        '''
        
        try:
            #Acess specified collection from the default or specified database
            if database_name is None: # If dataset is not defined, use the default database from the MongoDBClient
                 collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client.client[database_name][collection_name]

            #Convert collection data to DataFrame and preprocess
            print ('Fetching data from MongoDB collection...')
            df = pd.DataFrame(list(collection.find())) #Find retorna todos os documentos da collection
            print(f' Data feteched successfully. Number of records: {len(df)}')
            if 'id' in df.columns.to_list():
                df = df.drop(columns=['id'], axis =1) # Drop the 'id' column if it exists
            df.replace({np.nan: None}, inplace=True) # Replace NaN values with None for better compatibility with MongoDB
            
            return df  
        
        except Exception as e:
            raise MyException(e, sys)  