from FoodDeliveryTimePrediction.Logging.logger import logging
from FoodDeliveryTimePrediction.Exception.exception import FoodDeliveryTimePredictionException
from FoodDeliveryTimePrediction.Entity.config_entity import DataTransformationConfig
from FoodDeliveryTimePrediction.Entity.artifacts_entity import DataValidationArtifacts,DataTransformationArtifacts
from FoodDeliveryTimePrediction.Constants import Target_Column
from FoodDeliveryTimePrediction.Utils.main_utils import save_numpy_array_data,save_obj
import os
import sys
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer



class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifacts,data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact=data_validation_artifact
            self.data_transformation_config=data_transformation_config
        except Exception as e:
            raise FoodDeliveryTimePredictionException(e,sys)
    
    @staticmethod
    def read_data(file_path:str):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            FoodDeliveryTimePredictionException(e,sys)
            
    def build_preprocessor(self)->ColumnTransformer:
        
        try:
            numerical_features = ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs']
            categorical_onehot = ['Weather', 'Vehicle_Type']
            categorical_ordinal = ['Traffic_Level', 'Time_of_Day']
            
            ## Defining oridinal categories
            traffic_mapping = ['Low', 'Medium', 'High']
            time_mapping = ['Afternoon', 'Evening', 'Night', 'Morning']
            
            numerical_pipelien=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='mean')),
                    ('scaler',StandardScaler())
                ]
            )
            onehot_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('onehot',OneHotEncoder(drop='first'))
                ]
            )
            ordinal_pipeline= Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinal',OrdinalEncoder(categories=[traffic_mapping,time_mapping]))
                ]
            )
            
            
            ## combining all the pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num',numerical_pipelien,numerical_features),
                    ('onehot',onehot_pipeline,categorical_onehot),
                    ('ordinal',ordinal_pipeline,categorical_ordinal)
                ]
            )
            return preprocessor
        except Exception as e:
            FoodDeliveryTimePredictionException(e,sys)
    
    def initiate_data_transformation(self):
        try:
            logging.info('Enter inittiate_data_transformation method of DataTransformation class')

            train_df=DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            
            test_df=DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            
            input_feature_train_df=train_df.drop(columns=[Target_Column],axis=1)
            target_feature_train_df=train_df[Target_Column]
            
            input_feature_test_df=test_df.drop(columns=[Target_Column],axis=1)
            target_feature_test_df=test_df[Target_Column]
            
            
            preprocessor = self.build_preprocessor()
            
            processor_obj = preprocessor.fit(input_feature_train_df)
            
            transformed_input_train_feature=preprocessor.transform(input_feature_train_df)
            transformed_input_test_feature=preprocessor.transform(input_feature_test_df)
        
            onehot_cols = preprocessor.named_transformers_['onehot']['onehot'].get_feature_names_out(['weather','Vehicle_Type'])
                    
            all_cols = ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs'] + list(onehot_cols) + ['Traffic_Level_encoded', 'Time_of_day_encoded']
            
            transformed_input_train_df=pd.DataFrame(transformed_input_train_feature,columns=all_cols)    
            
            transformed_input_test_df=pd.DataFrame(transformed_input_test_feature,columns=all_cols)   
            
            final_train_df=pd.concat([transformed_input_train_df,target_feature_train_df.reset_index(drop=True)],axis=1)
            final_train_df=final_train_df.to_numpy()
            
            final_test_df=pd.concat([transformed_input_test_df,target_feature_test_df.reset_index(drop=True)],axis=1)
            final_test_df=final_test_df.to_numpy()
            
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,final_train_df)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,final_test_df)

            save_obj(self.data_transformation_config.transformed_object_file,processor_obj)
            
            save_obj('final_model/preprocessor.pkl',processor_obj)
            
            ## Preparing Data Transformation Artifacts
            
            data_transformation_artifacts=DataTransformationArtifacts(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            
            
            return data_transformation_artifacts
            
        except Exception as e:
            raise FoodDeliveryTimePredictionException(e,sys)
        

