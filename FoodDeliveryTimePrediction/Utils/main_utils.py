import yaml
from FoodDeliveryTimePrediction.Logging.logger import logging
from FoodDeliveryTimePrediction.Exception.exception import FoodDeliveryTimePredictionException
import os,sys
import numpy as np
import pickle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error,mean_squared_error,r2_score
from FoodDeliveryTimePrediction.Entity.artifacts_entity import RegressionMetricArtifact

def write_yaml_file(file_path:str,content:object,replace:bool=False)->None:
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'w') as file:
            yaml.dump(content,file)
    except Exception as e:
        raise FoodDeliveryTimePredictionException(e,sys) 
    

def save_numpy_array_data(file_path:str,array:np.array):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file:
            np.save(file,array)
            
    except Exception as e:
        raise FoodDeliveryTimePredictionException(e,sys)

def save_obj(file_path:str,obj:object)->None:
    try:
        logging.info('Entered the save-object method of main_utils ')
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
        logging.info('Exited the save_object method of the main_utils')
    except Exception as e:
        raise FoodDeliveryTimePredictionException(e,sys)
    
    
def load_object(file_path:str)->object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f'The file path : {file_path} is not exist')
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise FoodDeliveryTimePredictionException(e,sys)
    
    
def load_numpy_array_data(file_path:str)->np.array:
    try:
        if not os.path.exists(file_path):
            raise Exception(f'File path : {file_path} not exist')
        with open(file_path,'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise FoodDeliveryTimePredictionException(e,sys)
    

def evaluate_models(x_train,y_train,x_test,y_test,models,params):
    try:
        report={}
        
        for i in range(len(models)):
            model = list(models.values())[i]
            model_name=list(models.keys())[i]
            para=params[model_name]
            
            rcv=RandomizedSearchCV(model,para,cv=5,n_jobs=-1,n_iter=20)
            rcv.fit(x_train,y_train)
            
            model.set_params(**rcv.best_params_)
            model.fit(x_train,y_train)
            
            y_pred=model.predict(x_test)
            
            test_model_r2_score=r2_score(y_test,y_pred)
            
            report[model_name]=test_model_r2_score
            
            return report
        
    except Exception as e:
        raise FoodDeliveryTimePredictionException(e,sys)
            

def get_regression_score(y_true,y_pred)->RegressionMetricArtifact:
    try:
        model_r2_score=r2_score(y_true,y_pred)
        model_mse=mean_squared_error(y_true,y_pred)
        model_rmse=root_mean_squared_error(y_true,y_pred)
        
        regression_metric=RegressionMetricArtifact(
            r2_score=model_r2_score,
            mean_squared_error=model_mse,
            root_mean_squared_error=model_rmse
        )
        return regression_metric
    except Exception as e:
        raise FoodDeliveryTimePredictionException(e,sys)
    
    
class TimePredictionModel:
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor=preprocessor
            self.model=model
        except Exception as e:
            raise FoodDeliveryTimePredictionException(e,sys)
    def predict(self,x):
        try:
            x_transform=self.preprocessor.transform(x)
            y_pred=self.model.predict(x_transform)
            return x_transform
        except Exception as e:
            raise FoodDeliveryTimePredictionException(e,sys)