import yaml
from FoodDeliveryTimePrediction.Logging.logger import logging
from FoodDeliveryTimePrediction.Exception.exception import FoodDeliveryTimePredictionException
import os,sys
import numpy as np


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
    
    