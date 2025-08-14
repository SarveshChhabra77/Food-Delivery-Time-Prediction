from FoodDeliveryTimePrediction.Logging.logger import logging
from FoodDeliveryTimePrediction.Exception.exception import FoodDeliveryTimePredictionException
from FoodDeliveryTimePrediction.Entity.config_entity import TrainingPipelineConfig,DataIngestionConfig,DataValidationConfig,DataTransformationConfig
from FoodDeliveryTimePrediction.Src.Data_Transformation.data_transformation import DataTransformation
from FoodDeliveryTimePrediction.Src.Data_Ingestion.data_ingestion import DataIngestion
from FoodDeliveryTimePrediction.Src.Data_Validation.data_validation import DataValidation
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
        
        data_validation_config=DataValidationConfig(training_pipeline_config)
        
        data_validation=DataValidation(data_ingestion_artifacts,data_validation_config)
        
        logging.info("Initiating Data Validation")
        dataValidation_artifact=data_validation.initiate_data_validation()
        logging.info('Data Validation Completed')
        print(dataValidation_artifact)
        
        data_transformation_config=DataTransformationConfig(training_pipeline_config)

        data_transformation=DataTransformation(data_validation_artifact=dataValidation_artifact,data_transformation_config=data_transformation_config)
        
        logging.info('Initiating Data Tranformation')
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        logging.info('data Transformation complted')
        print(data_transformation_artifact)
        
        
    except Exception as e:
        raise FoodDeliveryTimePredictionException(e,sys)