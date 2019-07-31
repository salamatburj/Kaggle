import pandas as pd
import numpy as np
import datetime
import lightgbm as lgb
import fire

### helper funcs for convert

def binarize(df):
    for col in ['authorized_flag', 'category_1']:
        df[col] = df[col].map({'Y':1, 'N':0})
    return df

def read_data(input_file):
    df = pd.read_csv(input_file)
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days
    return df

def agg_trans(history):
    history.purchase_date = pd.DatetimeIndex(history.purchase_date).astype(np.int) * 1e-9
    agg_func = {
        'category_1': ['sum', 'mean'],
        'category_2_1': ['mean'],
        'category_2_2': ['mean'],
        'category_2_3': ['mean'],
        'category_2_4': ['mean'],
        'category_2_5': ['mean'],
        'category_3_A': ['mean'],
        'category_3_B': ['mean'],
        'category_3_C': ['mean'],
        'merchant_id': ['nunique'],
        'merchant_category_id': ['nunique'],
        'state_id': ['nunique'],
        'city_id': ['nunique'],
        'subsector_id': ['nunique'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
        'installments': ['sum', 'mean', 'max', 'min', 'std'],
        'purchase_month': ['mean', 'max', 'min', 'std'],
        'purchase_date': [np.ptp, 'min', 'max'],
        'month_lag': ['min', 'max'],
        'month_diff': ['mean'],
        'weekend': ['sum', 'mean'],
    }
    for col in ['month','hour','weekofyear','dayofweek','year']:
        agg_func[col] = ['nunique']
    
    
    agg_history = history.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['_'.join(col).strip() for col in agg_history.columns.values]
    df = history.groupby('card_id').size()
    df.name = 'transactions_count'
    agg_history = agg_history.join(df)

    return agg_history

def agg_per_month(history):
    grouped = history.groupby(['card_id', 'month_lag'])

    agg_func = {
            'purchase_amount': ['count', 'sum', 'mean', 'min', 'max', 'std'],
            'installments': ['count', 'sum', 'mean', 'min', 'max', 'std'],
            }

    intermediate_group = grouped.agg(agg_func)
    intermediate_group.columns = ['_'.join(col).strip() for col in intermediate_group.columns.values]
    intermediate_group.reset_index(inplace=True)

    final_group = intermediate_group.groupby('card_id').agg(['mean', 'std'])
    final_group.columns = ['_'.join(col).strip() for col in final_group.columns.values]
    
    return final_group


def convert(train_csv, test_csv, hist_csv, new_csv, out_prefix, nrows=None):
    # read train/test
    train = read_data(train_csv)
    test = read_data(test_csv)
    target = train.target.values
    train.drop('target', axis=1, inplace=True)

    # read hist/new
    hist_trans = pd.read_csv(hist_csv, parse_dates=['purchase_date'], nrows=nrows)
    new_trans = pd.read_csv(new_csv, parse_dates=['purchase_date'], nrows=nrows)
    hist_trans = binarize(hist_trans)
    new_trans = binarize(new_trans)

    # fill missing values
    hist_trans.category_2 = hist_trans.category_2.fillna(0).astype(int)
    new_trans.category_2 = new_trans.category_2.fillna(0).astype(int)
    hist_trans.category_3 = hist_trans.category_3.fillna('A')
    new_trans.category_3 = new_trans.category_3.fillna('A')
    hist_trans.merchant_id = hist_trans.merchant_id.fillna('M_ID_00a6ca8a8a')
    new_trans.merchant_id = new_trans.merchant_id.fillna('M_ID_00a6ca8a8a')

    # add datetime features
    for df in [hist_trans, new_trans]:
        df['year'] = df.purchase_date.dt.year
        df['weekofyear'] = df.purchase_date.dt.weekofyear
        df['month'] = df.purchase_date.dt.month
        df['dayofweek'] = df.purchase_date.dt.dayofweek
        df['weekend'] = (df.purchase_date.dt.weekday >= 5).astype(int)
        df['hour'] = df.purchase_date.dt.hour
        df['month_diff'] = (datetime.datetime.today() - df.purchase_date).dt.days // 30
        df['month_diff'] += df.month_lag


    # feature engineering
    hist_trans = pd.get_dummies(hist_trans, columns=['category_2', 'category_3'])
    new_trans = pd.get_dummies(new_trans, columns=['category_2', 'category_3'])
    
    auth_mean = hist_trans.groupby('card_id').agg({'authorized_flag': ['sum', 'mean']})
    auth_mean.columns = ['_'.join(col).strip() for col in auth_mean.columns]

    authed_trans = hist_trans[hist_trans.authorized_flag == 1]
    hist_trans = hist_trans[hist_trans.authorized_flag == 0]

    hist_trans['purchase_month'] = hist_trans['purchase_date'].dt.month
    authed_trans['purchase_month'] = authed_trans['purchase_date'].dt.month
    new_trans['purchase_month'] = new_trans['purchase_date'].dt.month

    hist = agg_trans(hist_trans)
    hist.columns = [f'hist_{c}' for c in hist.columns]
    authed = agg_trans(authed_trans)
    authed.columns = [f'auth_{c}' for c in authed.columns]
    new = agg_trans(new_trans)
    new.columns = [f'new_{c}' for c in new.columns]

    final_group = agg_per_month(hist_trans)

    train = train.join(hist, on='card_id')
    train = train.join(authed, on='card_id')
    train = train.join(new, on='card_id')
    train = train.join(final_group, on='card_id')
    train = train.join(auth_mean, on='card_id')
    test = test.join(hist, on='card_id')
    test = test.join(authed, on='card_id')
    test = test.join(new, on='card_id')
    test = test.join(final_group, on='card_id')
    test = test.join(auth_mean, on='card_id')
    print(train.shape, test.shape)

    np.savetxt(f'{out_prefix}.target.txt', target)
    train.to_csv(f'{out_prefix}.tr.csv', index=False)
    test.to_csv(f'{out_prefix}.te.csv', index=False)

def model(prefix, out_submit):
    target = np.loadtxt(f'{prefix}.target.txt')
    train = pd.read_csv(f'{prefix}.tr.csv', index_col=0)
    test = pd.read_csv(f'{prefix}.te.csv', index_col=0)
    features = [c for c in train.columns if c not in ['card_id', 'first_active_month']]
    categorical_feats = [c for c in features if 'feature_' in c]

    param = {'num_leaves': 31,
         'min_data_in_leaf': 150, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.005,
         "boosting": "gbdt",
         "feature_fraction": 0.8,
         "bagging_freq": 1,
         "bagging_fraction": 0.8 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         #"nthread": 4,
         #"verbosity": -1
    }

    tr_data = lgb.Dataset(train[features], label=target, categorical_feature=categorical_feats)

    clf = lgb.train(param, tr_data, 3000, verbose_eval=200)
    y_pred = clf.predict(test[features], num_iteration=clf.best_iteration)

    pd.DataFrame({
        'card_id': test.card_id,
        'target': y_pred
    }).to_csv(out_submit, index=False)


def outlier(prefix, out_pred):
    target = np.loadtxt(f'{prefix}.target.txt')
    target_outlier = (target < -33).astype(int)
    train = pd.read_csv(f'{prefix}.tr.csv', index_col=0)
    test = pd.read_csv(f'{prefix}.te.csv', index_col=0)
    features = [c for c in train.columns if c not in ['card_id', 'first_active_month']]
    categorical_feats = [c for c in features if 'feature_' in c]

    param = {'num_leaves': 31,
         'min_data_in_leaf': 150, 
         'objective':'binary',
         'max_depth': -1,
         'learning_rate': 0.005,
         "boosting": "gbdt",
         "feature_fraction": 0.8,
         "bagging_freq": 1,
         "bagging_fraction": 0.8 ,
         "bagging_seed": 11,
         "metric": 'auc',
         "lambda_l1": 0.1,
         'scale_pos_weight': 15
         #"nthread": 4,
         #"verbosity": -1
    }

    tr_data = lgb.Dataset(train[features], label=target_outlier, categorical_feature=categorical_feats)

    clf = lgb.train(param, tr_data, 1000, verbose_eval=200)
    y_pred = clf.predict(test[features], num_iteration=clf.best_iteration)

    pd.DataFrame({
        'card_id': test.card_id,
        'target': y_pred
    }).to_csv(out_submit, index=False)
    
def model_bk(prefix, out_submit):
    target = np.loadtxt(f'{prefix}.target.txt')
    train = pd.read_csv(f'{prefix}.tr.csv')
    test = pd.read_csv(f'{prefix}.te.csv')
    features = [c for c in train.columns if c not in ['card_id', 'first_active_month']]
    categorical_feats = [c for c in features if 'feature_' in c]

    lgbr = LGBMRegressor(n_estimators=1000, n_jobs=32)
    lgbr.fit(train[features], target)

    y_pred = lgbr.predict(test[features])

    pd.DataFrame({
        'card_id': test.card_id,
        'target': y_pred
    }).to_csv(out_submit, index=False)

def main():
    pass


if __name__ == "__main__":
    fire.Fire({
        'convert': convert,
        'model': model
    })
