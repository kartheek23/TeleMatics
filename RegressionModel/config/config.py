import pathlib
import pandas as pd
import os
parDir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
import sys
sys.path.append(parDir)

#PACKAGE_ROOT = pathlib.Path(RegressionModel.__file__).resolve().parent
DATASET_DIR = parDir + '\\DataSets'

TRAIN_MODEL_DIR = parDir + '\\trained_models'

PIPELINE_SAVE_FILE_NAME = "model_v1"

#Features or Independent variables we want to use
FEATURES = ['Accuracy','Bearing',
            'acceleration_x','acceleration_y',
            'acceleration_z','second','Speed']

#Target or Dependent variable we want to predict
TARGET = ['label']

#Training data set file
TRAINING_DATA_FILE = '\\MergedDataset\\Final.csv'

#Test Data File for prediction
TEST_DATA_FOLDER = DATASET_DIR + '\\TestFileUpload'

#Test Predictions folder
TEST_PREDICTION_FOLDER = DATASET_DIR + '\\Predictions'
