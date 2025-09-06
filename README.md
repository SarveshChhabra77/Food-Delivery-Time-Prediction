Overview

This project predicts estimated food delivery time (in minutes) from order and context features using a training pipeline (ingestion → validation → transformation → modeling) and a ready‑to‑use Streamlit app for interactive inference.
The repository includes reproducible notebooks, a modular Python package, persisted artifacts (preprocessor and model), and experiment tracking via MLflow.

Key features
End‑to‑end ML pipeline: data ingestion, schema validation, missing‑value checks, drift detection, feature engineering, model selection, and artifact persistence.

Robust preprocessing: One‑Hot for nominal categorical features, Ordinal encoding with explicit orders for traffic and time of day, and scaling for numeric columns.

Model selection with hyperparameter search across DecisionTree, RandomForest, AdaBoost, GradientBoosting, XGBoost, and CatBoost, selecting the best by R² and logging metrics to MLflow.

Interactive inference app built with Streamlit that loads final preprocessor and model from the final_model directory.

Project structure
The repository is organized to separate configuration, pipeline stages, utilities, notebooks, data, and final artifacts.

text
Food-Delivery-Time-Prediction-main/
├─ app.py                              # Streamlit app for inference [run]
├─ main.py                             # Orchestrates the full training pipeline
├─ requirements.txt                    # Python dependencies
├─ setup.py                            # Packaging metadata
├─ Data/
│  ├─ RawData/Food_Delivery_Times.csv  # Raw dataset
│  └─ Processed/delivery_data_cleaned.csv
├─ Data_Schema/__init__.py             # SCHEMA and Required_Columns dicts
├─ final_model/
│  ├─ model.pkl                        # Best trained regressor
│  └─ preprocessor.pkl                 # Fitted preprocessing pipeline
├─ FoodDeliveryTimePrediction/
│  ├─ Constants/__init__.py            # All constants and paths
│  ├─ Entity/                          # Config and artifact dataclasses
│  ├─ Exception/exception.py           # Custom exception
│  ├─ Logging/logger.py                # Centralized logging
│  ├─ Src/                             # Pipeline steps
│  │  ├─ Data_Ingestion/data_ingestion.py
│  │  ├─ Data_Validation/data_validation.py
│  │  ├─ Data_Transformation/data_transformation.py
│  │  └─ Model_Trainer/model_trainer.py
│  └─ Utils/main_utils.py              # I/O, search, metrics, wrapper model
├─ NoteBook/                           # Data cleaning, EDA, and training notebooks
└─ mlruns/                             # MLflow runs and model tracking artifacts
Data schema
Required feature columns and expected types are defined in Data_Schema/init.py as Required_Columns.

Features: Distance_km (float), Weather (string), Traffic_Level (string: Low/Medium/High), Time_of_Day (string: Morning/Afternoon/Evening/Night), Vehicle_Type (string: Bike/Scooter/Car), Preparation_Time_min (float/int).

Target: Delivery_Time_min (float/int).

How the pipeline works
Data Ingestion reads the raw CSV and splits train/test per configured ratio, persisting feature‑store and split files under timestamped Artifacts.

Data Validation enforces schema conformance, checks missing‑value thresholds per column, and performs Kolmogorov–Smirnov based dataset drift detection with a YAML drift report.

Data Transformation fits a preprocessing ColumnTransformer that scales numerics, one‑hot encodes Weather and Vehicle_Type, and ordinal‑encodes Traffic_Level and Time_of_Day using explicit category orders, and saves preprocessed NumPy arrays and the fitted preprocessor.

Model Trainer tunes and evaluates multiple regressors with RandomizedSearchCV, selects the best by R², logs metrics to MLflow, persists a TimePredictionModel wrapper and also saves best raw model and preprocessor into final_model for the app.

Installation
Ensure Python and pip are available, then install dependencies from requirements.txt.

It is recommended to use a virtual environment for isolation.

Commands

text
# From the repo root
pip install -r requirements.txt
Training
The complete training pipeline is orchestrated by main.py and will create timestamped Artifacts containing each stage’s outputs.
The run logs are stored under logs/, and MLflow run metadata and models live under mlruns/.

Commands

text
# From the repo root
python main.py
Outputs

Artifacts/<timestamp>/data_ingestion/...: feature store CSV, train/test splits.

Artifacts/<timestamp>/data_validation/...: validated train/test and drift report YAML.

Artifacts/<timestamp>/data_transformation/...: transformed train/test arrays and preprocessing object.

Artifacts/<timestamp>/model_trainer/...: trained wrapper model and metrics.

final_model/model.pkl and final_model/preprocessor.pkl are also written for the Streamlit app.

Inference app (Streamlit)
The Streamlit UI in app.py loads final_model/preprocessor.pkl and final_model/model.pkl and provides inputs for features to predict delivery time in minutes.
Interactive inputs include distance, weather, traffic level, time of day, vehicle type, and preparation time, and clicking Predict Delivery Time returns an estimated time.

Commands

text
# From the repo root
streamlit run app.py
Configuration and constants
All directory names, file names, train/test split ratio, expected accuracy thresholds, and preprocessing object names are centralized in FoodDeliveryTimePrediction/Constants/init.py.
Artifact and step‑level configs are instantiated via dataclasses in FoodDeliveryTimePrediction/Entity/config_entity.py and are passed into each pipeline stage.

Experiment tracking
The Model Trainer logs metrics (r2_score, mean_squared_error, root_mean_squared_error) and the trained model to MLflow, which organizes runs under mlruns/ for offline inspection.
This allows comparing models selected via hyperparameter search and preserving model artifacts for future evaluation.

Notebooks
Three notebooks are included to illustrate cleaning, exploratory data analysis, and trainer experimentation: 1‑DataCleaning.ipynb, 2‑EDA.ipynb, and 3‑ModelTrainer.ipynb.
They demonstrate basic imputations, encodings, visualizations, and model selection findings (with CatBoost favored for non‑linear patterns in an example workflow).

Technical details
Preprocessing: SimpleImputer(mean) + StandardScaler for numeric; SimpleImputer(most_frequent) + OneHotEncoder(drop='first') for Weather and Vehicle_Type; OrdinalEncoder with explicit lists for Traffic_Level and Time_of_Day.

Models searched: DecisionTreeRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, XGBRegressor, CatBoostRegressor with tuned grids via RandomizedSearchCV.

Metrics: R², MSE, RMSE for both training and test predictions, encapsulated via RegressionMetricArtifact.

Data paths and portability notes
Data Ingestion currently reads from a hardcoded Windows path (D:\Food Delivery Time Prediction\Data\RawData\Food_Delivery_Times.csv), which should be pointed to the repository’s Data/RawData/Food_Delivery_Times.csv for portability.

Constants.Schema_file_path is set to a Windows path but schema is actually imported from Data_Schema/init.py; verify local paths before running on non‑Windows environments.

How to extend
Add new features or categorical levels by updating Data_Schema.Required_Columns and the DataTransformation encoder mappings and pipelines accordingly.

Add new models or tuning grids by editing models and param_grids in Model_Trainer/model_trainer.py and re‑running main.py.

Requirements
All necessary packages are pinned in requirements.txt, including Streamlit for the app, scikit‑learn, XGBoost, CatBoost for training, and MLflow for experiment tracking.
Install from this file to replicate the development environment and ensure compatible versions.

Troubleshooting
If app.py errors on model loading, ensure final_model/model.pkl and final_model/preprocessor.pkl exist (retrain with python main.py if needed).

If training fails on data read or schema validation, confirm Data/RawData/Food_Delivery_Times.csv exists and matches Data_Schema.Required_Columns.

Author 
Author: Sarvesh Chhabra (from setup.py metadata).

Quick start
Install dependencies: pip install -r requirements.txt.

Train the pipeline: python main.py.

Launch the app: streamlit run app.py.


Invite your team
Deeper app integrations and secure team collaboration
Food-Delivery-Time
