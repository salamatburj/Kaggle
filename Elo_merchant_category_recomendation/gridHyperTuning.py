import pandas as pd
import numpy as np
from datetime import datetime
import lightgbm as lgb
import xgboost as xgb
import fire
from sklearn.model_selection import ParameterGrid
import pickle

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    else:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


def optimize_xgb(data, target, log_pickle):
    trn_data = xgb.DMatrix(data, label=target)

    param_range = {
        'max_depth': range(4, 10),
        'min_child_weight': range(10, 150, 30),
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }

    start_time = timer()
    
    log = []
    for param in ParameterGrid(param_range):
        param.update({
            'objective': 'reg:linear',
            'booster': 'gbtree',
            'eval_metric': 'rmse',
            'learning_rate': 0.01,
            'silent': 1, 
        })
        xgb_cv = xgb.cv(param, trn_data, 10000, nfold=4, metrics=('rmse'), early_stopping_rounds=600, verbose_eval=200)
        
        log.append((xgb_cv.iloc[:, -1].idxmin(), xgb_cv.iloc[:, -1].min(), param))

        with open(log_pickle, 'wb') as f:
            pickle.dump(log, f)

    timer(start_time)

def main(train_csv, target_txt, log_pickle):
    train = pd.read_csv(train_csv, index_col=0)
    target = np.loadtxt(target_txt)
    optimize_xgb(train.iloc[:, 2:].values, target, log_pickle)

if __name__ == "__main__":
    fire.Fire(main)