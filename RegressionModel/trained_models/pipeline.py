import os
parDir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
import sys
sys.path.append(parDir)
from processor import preprocessor as pp
from config import config
from sklearn.pipeline import Pipeline
from xgboost.sklearn import XGBClassifier

tele_pipe = Pipeline([
    #('numericalImputer',pp.NumericalImputer(variables = config.FEATURES)),
	#('scaler',pp.MinMaxScaling(variables=config.FEATURES))
    ('xgb',XGBClassifier(
 learning_rate=0.3,
 n_estimators=200,
 max_depth=20,
 verbosity=2,
 min_child_weight=1,
 gamma=0,
 subsample=0.5,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 scoring="roc_auc",
 n_jobs=-1,
 reg_alpha=1,
 ))])