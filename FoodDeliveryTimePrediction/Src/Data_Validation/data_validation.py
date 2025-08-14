from FoodDeliveryTimePrediction.Logging.logger import logging
from FoodDeliveryTimePrediction.Exception.exception import FoodDeliveryTimePredictionException
from FoodDeliveryTimePrediction.Entity.artifacts_entity import DataIngestionArtifacts, DataValidationArtifacts
from FoodDeliveryTimePrediction.Entity.config_entity import DataValidationConfig
import os
import pandas as pd
import sys
from typing import Dict, Any, List, Union
from Data_Schema import SCHEMA
from scipy.stats import ks_2samp
from FoodDeliveryTimePrediction.Utils.main_utils import write_yaml_file
from Data_Schema import SCHEMA

class DataValidation:
    """
    Class responsible for validating ingested data before further processing.
    Includes schema validation, missing value checks, and dataset drift detection.
    """

    def __init__(self, data_ingestion_artifact: DataIngestionArtifacts, data_validation_config: DataValidationConfig):
        """
        Constructor to initialize data ingestion artifacts and data validation config.
        """
        try:
            self.data_ingestion_artifacts = data_ingestion_artifact
            self.data_validation_config = data_validation_config
        except Exception as e:
            raise FoodDeliveryTimePredictionException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Reads CSV data into a pandas DataFrame.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise FoodDeliveryTimePredictionException(e, sys)

    def validate_dataframe_schema(self,dataframe: pd.DataFrame, schema: Dict[str, Union[Any, List[Any]]]) -> bool:
        """
        Validates that the DataFrame matches the expected schema:
        - Column names match exactly
        - Data types match expected types
        """
        try:
            logging.info('Starting schema validation...')

            # --- Step 1: Validate column names ---
            expected_columns = set(schema.keys())
            dataframe_columns = set(dataframe.columns)

            if expected_columns != dataframe_columns:
                missing_columns = expected_columns - dataframe_columns
                extra_columns = dataframe_columns - expected_columns

                if missing_columns:
                    logging.info(f"Validation FAILED: Missing columns: {missing_columns}")
                if extra_columns:
                    logging.info(f"Validation FAILED: Extra columns found: {extra_columns}")
                return False

            # --- Step 2: Validate column data types ---
            for column, expected_dtypes in schema.items():
                # Ensure expected_dtypes is always treated as a list
                if not isinstance(expected_dtypes, list):
                    expected_dtypes = [expected_dtypes]

                actual_dtype = dataframe[column].dtype
                # Check if actual dtype matches any of the allowed dtypes
                if not any(pd.api.types.is_dtype_equal(actual_dtype, dt) for dt in expected_dtypes):
                    logging.info(
                        f"Validation FAILED: Column '{column}' has dtype '{actual_dtype}', "
                        f"expected one of {expected_dtypes}"
                    )
                    raise False

            logging.info("Validation PASSED: DataFrame schema is correct.")
            return True

        except Exception as e:
            raise FoodDeliveryTimePredictionException(e, sys)

    def check_missing_values(self,dataframe: pd.DataFrame, threshold: float = 0.5):
        """
        Checks if any column has more than the allowed percentage of missing values.
        """
        try:
            missing_data_report = {}
            for column in dataframe.columns:
                missing_percentage = dataframe[column].isnull().mean()
                if missing_percentage > 0:
                    missing_data_report[column] = missing_percentage

            # Check if any column exceeds the threshold
            if any(value > threshold for value in missing_data_report.values()):
                logging.info(f"Validation FAILED: Columns with more than {threshold*100}% missing values found.")
                logging.info(missing_data_report)
                return False
            else:
                logging.info("Validation PASSED: Missing value check successful.")
                return True
        except Exception as e:
            raise FoodDeliveryTimePredictionException(e, sys)

    def detect_dataset_drift(self, base_df: dict, current_df: dict, threshold=0.05) -> bool:
        """
        Detects dataset drift between base (train) data and current (test) data.
        Uses Kolmogorov-Smirnov test to compare distributions for each column.
        If p-value <= threshold → drift detected.
        """
        try:
            status = True  # Will be False if drift is detected
            report = {}

            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]

                # Compare distributions of two samples
                is_sample_dist = ks_2samp(d1, d2)

                # If p-value <= threshold → drift found
                if threshold <= is_sample_dist.pvalue:
                    is_found = False # no drift
                else:
                    is_found = True # drift found
                    status = False

                # Save results for the column
                report.update({
                    column: {
                        'p_value': float(is_sample_dist.pvalue),
                        'drift_status': is_found
                    }
                })

                # Save drift report as YAML file
                drift_report_filepath = self.data_validation_config.drift_report_file_path
                
                dir_path=os.path.dirname(drift_report_filepath)
                
                os.makedirs(dir_path, exist_ok=True)
                
                write_yaml_file(file_path=drift_report_filepath, content=report)

                return status

        except Exception as e:
            raise FoodDeliveryTimePredictionException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifacts:
        """
        Runs the full data validation process:
        1. Reads train and test data
        2. Validates schema
        3. Checks missing values
        4. Detects dataset drift
        5. Saves valid train/test files
        6. Returns DataValidationArtifacts object
        """
        try:
            # --- Step 1: Read train & test datasets ---
            train_file_path = self.data_ingestion_artifacts.train_file_path
            test_file_path = self.data_ingestion_artifacts.test_file_path

            train_df = self.read_data(train_file_path)
            test_df = self.read_data(test_file_path)

            # --- Step 2: Schema validation ---
            status = self.validate_dataframe_schema(train_df,SCHEMA)
            if not status:
                logging.info('Train dataframe does not contain all required columns.')

            status = self.validate_dataframe_schema(test_df,SCHEMA)
            if not status:
                logging.info('Test dataframe does not contain all required columns.')

            # --- Step 3: Missing values check ---
            status = self.check_missing_values(train_df)
            if not status:
                logging.info('Train dataframe contains excessive missing data.')

            status = self.check_missing_values(test_df)
            if not status:
                logging.info('Test dataframe contains excessive missing data.')

            # --- Step 4: Drift detection ---
            status = self.detect_dataset_drift(base_df=train_df, current_df=test_df)

            # --- Step 5: Save valid datasets ---
            dir_path = os.path.dirname(self.data_validation_config.valid_data_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_df.to_csv(self.data_validation_config.valid_data_train_file_path, index=False, header=True)
            test_df.to_csv(self.data_validation_config.valid_data_test_file_path, index=False, header=True)

            # --- Step 6: Create artifact object ---
            data_validation_artifacts = DataValidationArtifacts(
                validation_status=status,
                valid_train_file_path=self.data_validation_config.valid_data_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_data_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
            return data_validation_artifacts

        except Exception as e:
            raise FoodDeliveryTimePredictionException(e, sys)
