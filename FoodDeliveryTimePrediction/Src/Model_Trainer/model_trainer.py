import os
import sys
from FoodDeliveryTimePrediction.Logging.logger import logging
from FoodDeliveryTimePrediction.Exception.exception import FoodDeliveryTimePredictionException
from FoodDeliveryTimePrediction.Entity.artifacts_entity import RegressionMetricArtifact
from FoodDeliveryTimePrediction.Entity.artifacts_entity import DataTransformationArtifacts,ModelTrainerArtifacts
from FoodDeliveryTimePrediction.Entity.config_entity import ModelTrainerConfig
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from FoodDeliveryTimePrediction.Utils.main_utils import save_obj,load_numpy_array_data,evaluate_models,load_object,get_regression_score,TimePredictionModel
import mlflow

class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifacts):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifacts=data_transformation_artifact
        except Exception as e:
            raise FoodDeliveryTimePredictionException(e,sys)
    
    def track_mlflow(self,best_model,regression_metrics:RegressionMetricArtifact):
        
        try:
            with mlflow.start_run():
                r2_score=regression_metrics.r2_score
                mean_squared_error=regression_metrics.mean_squared_error
                root_mean_squared_error=regression_metrics.root_mean_squared_error
                # Logs each metric to MLflow under this run. These will be visible in the MLflow UI.
                mlflow.log_metric('r2_score',r2_score)
                mlflow.log_metric('mean_squared_error',mean_squared_error)
                mlflow.log_metric('root_mean_squared_error',root_mean_squared_error)
                
                # Logs the trained model under the name 'model'. MLflow stores the model in a format that you can load later for prediction or deployment.
                mlflow.sklearn.log_model(best_model,'model')
                
        except Exception as e:
            raise FoodDeliveryTimePredictionException(e,sys)
        
    def model_train(self,x_train,y_train,x_test,y_test):
        try:
            
            models = {
                'DecisionTree': DecisionTreeRegressor(),
                'RandomForest': RandomForestRegressor(),
                'AdaBoost': AdaBoostRegressor(),
                'GradientBoosting': GradientBoostingRegressor(),
                'XGBoost': XGBRegressor(),
                'CatBoost': CatBoostRegressor(verbose=0)
            }
            param_grids = {
                "DecisionTree": {
                    "max_depth": [None, 5, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": [None, "sqrt", "log2"]
                },
                
                "RandomForest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2"]
                },
                
                "AdaBoost": {
                    "n_estimators": [50, 100, 200, 300],
                    "learning_rate": [0.001, 0.01, 0.1, 0.5, 1]
                },
                
                "GradientBoosting": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.001, 0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                
                },
                
                "XGBoost": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.001, 0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7, 10],

                },
                
                'CatBoost' : {
                    "depth": [7, 8, 9],
                    "learning_rate": [0.02, 0.03, 0.04],
                    "iterations": [300, 500, 800],   # try longer training
                    "l2_leaf_reg": [5, 6, 7, 8],
                    "bagging_temperature": [0.3, 0.5, 0.7],
                    "random_strength": [1, 2, 3],
                    "min_data_in_leaf": [5, 10, 20],
                    "rsm": [0.7, 0.8, 0.9],
                    "max_bin": [254, 512],
                    "grow_policy": ["SymmetricTree", "Lossguide"]
                }
            }
            
            model_report,tunned_models=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,params=param_grids)

            best_model_score=max(list(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            
            best_model=tunned_models[best_model_name]
            
            y_train_pred=best_model.predict(x_train)
            
            regression_train_metrics=get_regression_score(y_train,y_train_pred)
            
            self.track_mlflow(best_model,regression_train_metrics)
            
            y_test_pred=best_model.predict(x_test)
            
            regression_test_metrics=get_regression_score(y_test,y_test_pred)
            
            self.track_mlflow(best_model,regression_test_metrics)
            
            preprocessor=load_object(self.data_transformation_artifacts.transformed_object_file_path)
            
            model_dir_path=os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)

            model=TimePredictionModel(preprocessor,best_model)
            
            save_obj(self.model_trainer_config.trained_model_file_path,model)
            
            save_obj('final_model/model.pkl',best_model)
            
            model_trainer_artifact=ModelTrainerArtifacts(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=regression_train_metrics,
                test_metric_artifacts=regression_test_metrics,
                model_name=best_model_name
            )
            
            logging.info(f'Model Trainer Artifacts : {model_trainer_artifact}')

            return model_trainer_artifact

        except Exception as e:
            raise FoodDeliveryTimePredictionException(e,sys)
        
    def initiate_model_trainer(self)->ModelTrainerArtifacts:
        try:
            train_file_path=self.data_transformation_artifacts.transformed_train_file_path
            test_file_path=self.data_transformation_artifacts.transformed_test_file_path
            
            ## loading training and testing arrays
            train_arr=load_numpy_array_data(train_file_path)
            test_arr=load_numpy_array_data(test_file_path)

            X_train,X_test,y_train,y_test=(
                train_arr[:,:-1],
                test_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,-1]
            )
            
            model_trainer_artifact=self.model_train(X_train,y_train,X_test,y_test)
            
            return model_trainer_artifact
            

        except Exception as e:
            raise FoodDeliveryTimePredictionException(e,sys)
            
