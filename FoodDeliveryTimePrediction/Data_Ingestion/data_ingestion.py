import os
import sys
from FoodDeliveryTimePrediction import Constants
from FoodDeliveryTimePrediction.Logging.logger import logging
from FoodDeliveryTimePrediction.Exception.exception import FoodDeliveryTimePredictionException
from FoodDeliveryTimePrediction.Entity.config_entity import DataIngestionConfig
from FoodDeliveryTimePrediction.Entity.artifacts_entity import DataIngestionArtifacts
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise FoodDeliveryTimePredictionException(e,sys)
        
    def export_data_as_dataframe(self):
        try:
            df = pd.read_csv('D:\Food Delivery Time Prediction\Data\RawData\Food_Delivery_Times.csv')
            return df
        except Exception as e:
            raise FoodDeliveryTimePredictionException(e,sys)
        
    def export_data_into_feature_stored(self,dataframe:pd.DataFrame):
        try:
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            
            dir_path=os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            
            
        except Exception as e:
            raise FoodDeliveryTimePredictionException(e,sys)
    
    def split_data_as_train_test(self,dataframe:pd.DataFrame):
        try:
            train_set,test_set=train_test_split(dataframe,test_size=self.data_ingestion_config.train_test_split_ratio)

            logging.info('Perform Train Test Split on the Dataframe')

            logging.info('Exited split_data_as_train_test of Data_Ingestion Class')

            dir_path=os.path.dirname(self.data_ingestion_config.train_file_path)
            
            os.makedirs(dir_path,exist_ok=True)

            logging.info('Exporting Dataframe as Train and Test File')

            train_set.to_csv(self.data_ingestion_config.train_file_path)
            test_set.to_csv(self.data_ingestion_config.test_file_path)
            
            logging.info('Exported Train and Test file path')
            
        except Exception as e:
            raise FoodDeliveryTimePredictionException(e,sys)
    
    def initiate_data_ingestion(self):
        try:
            dataframe=self.export_data_as_dataframe()
            self.export_data_into_feature_stored(dataframe)
            self.split_data_as_train_test(dataframe)
            Data_Ingestion_Artifacts=DataIngestionArtifacts(
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )
            return Data_Ingestion_Artifacts
        
        except Exception as e:
            raise FoodDeliveryTimePredictionException(e,sys)
