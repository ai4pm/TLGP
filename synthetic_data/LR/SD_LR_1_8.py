from numpy.random import seed
import pandas as pd
import random as rn
import os,sys
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from TLGP.TL_PRS import TL_PRS_scr

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1000)

seed(11111)
os.environ['PYTHONHASHSEED'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"
rn.seed(11111)

def independent_learning(data):
    train_data, val_data, test_data = data
    X_test, Y_test = test_data
    model = LogisticRegression(solver='newton-cg', C=0.5)
    model.fit(train_data[0], train_data[1])

    x_test_scr = np.round(model.predict_proba(X_test), decimals=3)[:,1]
    AUC = roc_auc_score(Y_test, x_test_scr)
    return AUC

def mixture_learning(Aggr, minor='AA'):
    train_data, val_data, test_data = Aggr
    X_test, Y_test, R_test = test_data

    model = LogisticRegression(solver='newton-cg', C=0.5)
    model.fit(train_data[0], train_data[1])
    x_test_scr = np.round(model.predict_proba(X_test)[:,1], decimals=3)

    A_AUC = roc_auc_score(Y_test, x_test_scr)
    Y_EA, scr_EA = Y_test[R_test == 'EUR'], x_test_scr[R_test == 'EUR']
    Y_minor, scr_minor = Y_test[R_test == minor], x_test_scr[R_test == minor]
    EA_AUC, AA_AUC = roc_auc_score(Y_EA, scr_EA), roc_auc_score(Y_minor, scr_minor)
    return A_AUC, EA_AUC, AA_AUC

def naive_transfer(EA, Minor):
    EA_train_data, EA_val_data, EA_test_data = EA
    Minor_train_data, Minor_val_data, Minor_test = Minor
    Minor_test_X, Minor_test_Y = Minor_test
    model = LogisticRegression(solver='newton-cg', C=0.5)

    model.fit(EA_train_data[0], EA_train_data[1])
    Minor_test_scr = np.round(model.predict_proba(Minor_test_X), decimals=3)[:,1]
    Naive_AUC = roc_auc_score(Minor_test_Y, Minor_test_scr)
    return Naive_AUC

def transfer(EA, Minor, k=500, batch=50, lr=0.005):
    EA_train_data, EA_val_data, EA_test_data = EA
    Minor_train_data, Minor_val_data, Minor_test = Minor
    X_test, Y_test = Minor_test

    X_s, Y_s = EA_train_data
    X_t, Y_t = Minor_train_data

    scr = TL_PRS_scr(0, X_s, X_t, Y_s, Y_t, k, X_test, batch=batch, lr=lr)
    auc = roc_auc_score(Y_test, scr)
    return auc

def get_data(file, seed=0, minor='AMR'):
    A = loadmat(file)
    R = A['R']
    Y = A['Y'][0].astype('int32')
    X = A['X'].astype('float32')

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    df = pd.DataFrame(X)
    df['R'], df['Y'] = R, Y
    df['YR'] = df['Y'].map(str) + df['R']

    train, test = train_test_split(df, test_size=0.2, random_state=11, shuffle=True, stratify=df['YR'])
    Y_train, R_train = train['Y'].values, train['R'].values
    train = train.drop(columns=['Y', 'R', 'YR'])
    X_train = train.values

    val_samples, test_samples = train_test_split(test, test_size=0.5, random_state=seed, shuffle=True, stratify=test['YR'])
    Y_val, R_val = val_samples['Y'].values, val_samples['R'].values
    Y_test, R_test = test_samples['Y'].values, test_samples['R'].values
    val_samples = val_samples.drop(columns=['Y', 'R', 'YR'])
    test_samples = test_samples.drop(columns=['Y', 'R', 'YR'])
    X_val, X_test = val_samples.values, test_samples.values

    train_data = (X_train, Y_train)
    val_data = (X_val, Y_val)
    test_data = (X_test, Y_test, R_test)

    EA_X_train, EA_Y_train = X_train[R_train == 'EUR'], Y_train[R_train == 'EUR']
    EAA_X_train, EAA_Y_train = X_train[R_train == minor], Y_train[R_train == minor]
    EA_X_val, EA_Y_val = X_val[R_val == 'EUR'], Y_val[R_val == 'EUR']
    EAA_X_val, EAA_Y_val = X_val[R_val == minor], Y_val[R_val == minor]
    EA_X_test, EA_Y_test = X_test[R_test == 'EUR'], Y_test[R_test == 'EUR']
    EAA_X_test, EAA_Y_test = X_test[R_test == minor], Y_test[R_test == minor]

    EA_train_data = (EA_X_train, EA_Y_train)
    EA_val_data = (EA_X_val, EA_Y_val)
    EA_test_data = (EA_X_test, EA_Y_test)

    EAA_train_data = (EAA_X_train, EAA_Y_train)
    EAA_val_data = (EAA_X_val, EAA_Y_val)
    EAA_test_data = (EAA_X_test, EAA_Y_test)

    Aggr = [train_data, val_data, test_data]
    EA = [EA_train_data, EA_val_data, EA_test_data]
    EAA = [EAA_train_data, EAA_val_data, EAA_test_data]

    return [Aggr, EA, EAA]

Map_Minor = ['AMR', 'SAS', 'EAS', 'AFR']*4

def run_SD(SD):
    res = []
    for seed in range(20):
        Aggr, EA, Minor = get_data("../SD{}.mat".format(SD), seed=seed, minor=Map_Minor[SD-1])
        Mix0, Mix1, Mix2 = mixture_learning(Aggr, minor=Map_Minor[SD-1])
        ind_1 = independent_learning(EA)
        ind_2 = independent_learning(Minor)
        naive = naive_transfer(EA, Minor)
        tl2 = transfer(EA, Minor, k=500, batch=50, lr=0.005)
        row = [Mix0, Mix1, Mix2, ind_1, ind_2, naive, tl2]
        print(row, seed)
        res.append(row)

    df_res = pd.DataFrame(res, columns=['Mix0_LR', 'Mix1_LR', 'Mix2_LR', 'Ind1_LR', 'Ind2_LR', 'NT_LR', 'TL_LR'])
    df['SD'] = SD
    print(df_res)


def main():
    arguments = sys.argv
    print(arguments)
    for SD in range(1,10):
        run_SD(SD)

if __name__ == '__main__':
    main()
