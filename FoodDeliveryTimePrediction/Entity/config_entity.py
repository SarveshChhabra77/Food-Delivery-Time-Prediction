from datetime import datetime
import os
from FoodDeliveryTimePrediction import Constants

class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):
        timestamp=timestamp.strftime('%m_%d_%Y_%H_%M_%S')
        self.pipeline=Constants.Pipeline_Name
        self.artifact_name=Constants.Artifact_Dir
        self.artifact_dir=os.path.join(self.artifact_name,timestamp)
        self.timestamp:str=timestamp
        
class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_dir:str=os.path.join(training_pipeline_config.artifact_dir,Constants.Data_Ingestion_Dir_name)
        
        self.feature_store_file_path:str=os.path.join(self.data_ingestion_dir,Constants.Data_Ingestion_Feature_Store_Dir,Constants.File_Name)
        
        self.train_file_path:str=os.path.join(self.data_ingestion_dir,Constants.Data_Ingestion_Ingested_Dir,Constants.Train_File_Name)
        
        self.test_file_path:str=os.path.join(self.data_ingestion_dir,Constants.Data_Ingestion_Ingested_Dir,Constants.Test_File_Name)

        self.train_test_split_ratio:float=Constants.Data_Ingestion_Train_Test_Split_Ratio
        
class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        
        self.data_validation_dir=os.path.join(training_pipeline_config.artifact_dir,Constants.Data_Validation_Dir_Name)
        
        self.valid_data_dir=os.path.join(self.data_validation_dir,Constants.Data_Validation_Valid_Data_Dir)
        
        self.invalid_data_dir=os.path.join(self.data_validation_dir,Constants.Data_Validation_Invalid_Data_Dir)
        
        
        self.valid_data_train_file_path=os.path.join(self.valid_data_dir,Constants.Train_File_Name)
        self.valid_data_test_file_path=os.path.join(self.valid_data_dir,Constants.Test_File_Name)
        
        self.invalid_data_train_file_path=os.path.join(self.invalid_data_dir,Constants.Train_File_Name)
        self.invalid_data_test_file_path=os.path.join(self.invalid_data_dir,Constants.Test_File_Name)
        
        self.drift_report_file_path=os.path.join(self.data_validation_dir,Constants.Data_Validation_Drift_Report_Dir,Constants.Data_Validation_Drift_Report_File_Name)
        
class DataTransformationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir=os.path.join(training_pipeline_config.artifact_dir,Constants.DATA_TRANSFORMATION_DIR_NAME)
        
        self.transformed_train_file_path=os.path.join(self.data_transformation_dir,Constants.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,Constants.Train_File_Name.replace('csv','npy'))

        self.transformed_test_file_path=os.path.join(self.data_transformation_dir,Constants.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,Constants.Test_File_Name.replace('csv','npy'))

        self.transformed_object_file=os.path.join(self.data_transformation_dir,Constants.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,Constants.PREPROCESSING_OBJECT_FILE_NAME)