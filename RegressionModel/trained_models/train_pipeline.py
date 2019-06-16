import os
parDir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
import sys
sys.path.append(parDir)
from sklearn.model_selection import train_test_split

from config import config
from processor import dataManagement
import pipeline
import pandas as pd

def run_training() -> None:
	"""Train the model. """
	""" Read Training data"""
	data = dataManagement.load_dataset(file_name=config.TRAINING_DATA_FILE)
	#Divide train and test
	X_train,X_test,y_train,y_test=train_test_split(
		pd.DataFrame(data[config.FEATURES]),pd.DataFrame(data[config.TARGET]),
		test_size=0.1,
		random_state=0)

	model = pipeline.tele_pipe.fit(X_train,y_train)
	dataManagement.save_pipeline(model)
	

if __name__ == '__main__':
    run_training()