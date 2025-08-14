import yaml
from FoodDeliveryTimePrediction.Logging.logger import logging
from FoodDeliveryTimePrediction.Exception.exception import FoodDeliveryTimePredictionException
import os,sys
import numpy as np
import pickle

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