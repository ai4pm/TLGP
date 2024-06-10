
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, average_precision_score
import numpy as np
from scipy.stats import norm
from sklearn.metrics import precision_score, confusion_matrix

def get_auc(y_true, y_pred_scr):
    return roc_auc_score(y_true, y_pred_scr)

def get_aupr(y_true, y_pred_scr):
    au_pr_score = average_precision_score(y_true, y_pred_scr)
    return au_pr_score

def get_opt_cutpoint(y_true, y_pred_scr):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_scr)
    youden_index = tpr - fpr
    optimal_threshold = thresholds[np.argmax(youden_index)]
    return optimal_threshold

def get_ppv(y_true, y_pred_scr):
    optimal_threshold = get_opt_cutpoint(y_true, y_pred_scr)
    y_pred = y_pred_scr > optimal_threshold
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    return ppv

def get_npv(y_true, y_pred_scr):
    optimal_threshold = get_opt_cutpoint(y_true, y_pred_scr)
    y_pred = y_pred_scr > optimal_threshold

    conf_matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    return npv

def get_Tjur_R2(y_true, y_pred_scr):
    R2 = np.mean(y_pred_scr[y_true==1]) - np.mean(y_pred_scr[y_true==0])
    return R2

def get_sensitivity(y_true, y_pred_scr):
    optimal_threshold = get_opt_cutpoint(y_true, y_pred_scr)
    y_pred = y_pred_scr > optimal_threshold
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    return sensitivity

def get_specificity(y_true, y_pred_scr):
    optimal_threshold = get_opt_cutpoint(y_true, y_pred_scr)
    y_pred = y_pred_scr > optimal_threshold
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity

Map = {'auc':get_auc, 'aupr':get_aupr, 'ppv':get_ppv, 'npv':get_npv, 'R2':get_Tjur_R2,
       'sensitivity':get_sensitivity, 'specificity':get_specificity}

def get_performance(y_true, y_pred_scr, metric='auc'):
    return Map[metric](y_true, y_pred_scr)

