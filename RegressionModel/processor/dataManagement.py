import pandas as pd
import os
from sklearn.externals import joblib
import cloudpickle
from sklearn.pipeline import Pipeline
import xgboost
parDir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
import sys
sys.path.append(parDir)
import dask.dataframe as dd
from config import config

def load_dataset(*, file_name: str
                 ) -> dd.DataFrame:
    _data = pd.read_csv(f'{config.DATASET_DIR}{file_name}')
    return _data

def save_pipeline(*,pipeline_to_persist) -> None:
	''' Persist the Pipeline'''
	save_file_name = f'{config.PIPELINE_SAVE_FILE_NAME}.pkl'
	save_path = config.TRAIN_MODEL_DIR / save_file_name
	joblib.dump(pipeline_to_persist,save_path)

def load_pipeline() -> Pipeline:
	'''Load a persisted pipeline'''
	save_file_name = f'{config.PIPELINE_SAVE_FILE_NAME}.pkl'
	save_path = config.TRAIN_MODEL_DIR + "\\" + save_file_name
	print(save_path)
	with open(save_path,mode='rb') as model_f:
    		model = cloudpickle.load(model_f)
	return model