import os
import sys
from xgboost.sklearn import XGBClassifier
parDir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
sys.path.append(parDir)
from processor import dataManagement
from config import config
from trained_models import predict
import pandas as pd
import numpy as np


def make_prediction(inputData):
    predictions = pd.DataFrame(_predict_pipe.predict(inputData[config.FEATURES]))
    predictions_proba = pd.DataFrame(_predict_pipe.predict_proba(inputData[config.FEATURES]))
    predictions.columns = config.TARGET
    print(metrics.roc_auc_score(inputData[config.TARGET],predictions_proba))
    print(classification_report(inputData[config.TARGET],predictions))
    print(confusion_matrix(inputData[config.TARGET],predictions)
    predictions.to_csv(config.TEST_PREDICTION_FOLDER + "/new_predictions.csv")
    predictions_proba.to_csv(config.TEST_PREDICTION_FOLDER + "/new_pred_proba.csv")
    print("Please check the predictions folder for the Predictions of the holdout dataset.")

def readFilesFromFolder(dirToRead):
    files = []
    #r=root,d=directories,f=file
    for r,d,f in os.walk(dirToRead):
        for file in f:
            if '.csv' in file:
                files.append(os.path.join(r,file))
    return files

def concatAllFilesInDir(file,inputCols):
    np_array_list = []
    if(len(file)>1):
        for f in file:
            df = pd.read_csv(f, index_col=None, header=0)
            np_array_list.append(df.as_matrix())
        comb_np_array = np.vstack(np_array_list)
        big_frame = pd.DataFrame(comb_np_array)
        big_frame.columns = inputCols
    else:
        for f in file:
            big_frame = pd.DataFrame(pd.read_csv(f))
    return big_frame

if __name__ == '__main__':
    _predict_pipe = dataManagement.load_pipeline()
    '''First read Files from Folder and then get the required dataFiles'''
    file=readFilesFromFolder(config.TEST_DATA_FOLDER)
    inputData=concatAllFilesInDir(file,config.FEATURES)
    make_prediction(inputData)