import os
import numpy as np


"Defining the training pipeline constants"

Target_Column:str='Delivery_Time_min'
Pipeline_Name:str=''
Artifact_Dir:str='Artifacts'
File_Name:str='Food_Delivery_Times.csv'
Train_File_Name:str='Train.csv'
Test_File_Name:str='Test.csv'
Saved_Model_Dir:str=os.path.join('saved_models')
Model_File_Name:str='model.pkl'


"Defining the data ingestion constant names starting with data ingestion"
Data_Ingestion_Collection_Name:str=''
Data_Ingestion_Dir_name:str='data_ingestion'
Data_Ingestion_Feature_Store_Dir:str='feature_stored'
Data_Ingestion_Ingested_Dir:str='ingested'
Data_Ingestion_Train_Test_Split_Ratio:float=0.2
