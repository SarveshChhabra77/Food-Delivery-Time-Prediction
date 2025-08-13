from FoodDeliveryTimePrediction.Logging.logger import logging
from FoodDeliveryTimePrediction.Exception.exception import FoodDeliveryTimePredictionException
from FoodDeliveryTimePrediction.Entity.config_entity import TrainingPipelineConfig,DataIngestionConfig
from FoodDeliveryTimePrediction.Data_Ingestion.data_ingestion import DataIngestion
import sys



if __name__=='__main__':
    
    try: 
        training_pipeline_config=TrainingPipelineConfig()
        
        data_ingestion_config=DataIngestionConfig(training_pipeline_config)
        
        data_ingestion=DataIngestion(data_ingestion_config=data_ingestion_config)
        
        logging.info('Initiating Data Ingestion')
        data_ingestion_artifacts=data_ingestion.initiate_data_ingestion()
        logging.info('Data Ingestion Complted')
        
        print(data_ingestion_artifacts)
        
    except Exception as e:
        raise FoodDeliveryTimePredictionException(e,sys)