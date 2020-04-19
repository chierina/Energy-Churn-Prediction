#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:20:03 2020

@author: naitochieri
"""
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score, f1_score, precision_score, confusion_matrix
from sklearn.base import ClassifierMixin, clone


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'



def quant_dup(list):
    my_set = set(list)
    dup = 0
    for i in my_set:
        count = 0
        
        for x in list:
            if x == i:
                count += 1
                if count == 2:
                    dup += 1
                    break
                return dup

    
def compare_nan_array(func, a, thresh):
    out = ~np.isnan(a)
    out[out] = func(a[out], thresh)
    return out


def one_minus_roc(X, y, est):
    pred = est.predict_proba(X)[:, 1]
    return 1-roc_auc_score(y, pred)

    
def feature_importance(model, x_col):
    X_columns = x_col
    plt.figure(figsize=(15, 15))
    ordering = np.argsort(model.feature_importances_)[::1][10:]
    importances = model.feature_importances_[ordering]
    feature_names = X_columns[ordering]
    x = np.arange(len(feature_names))
    width = 0.2
    plt.barh(feature_names, importances, width)
    plt.title("Feature Importance" ,fontsize=20)
    plt.xlabel("importances" ,fontsize=18)
    plt.ylabel("features" ,fontsize=18)
    plt.rc('ytick', labelsize=15) 


class EarlyStopping(ClassifierMixin):
    def __init__(self, estimator, max_n_estimators, scorer,
                 n_min_iterations=30, scale=1.02):
        self.estimator = estimator
        self.max_n_estimators = max_n_estimators
        self.scorer = scorer
        self.scale = scale
        self.n_min_iterations = n_min_iterations
    
    def _make_estimator(self, append=True):
        estimator = clone(self.estimator)
        estimator.n_estimators = 1
        estimator.warm_start = True
        return estimator
    
    def fit(self, X, y):
        est = self._make_estimator()
        self.scores_ = []

        for n_est in range(1, self.max_n_estimators+1):
            est.n_estimators = n_est
            est.fit(X,y)
            
            score = self.scorer(est)
            self.estimator_ = est
            self.scores_.append(score)

            if (n_est > self.n_min_iterations and
                score > self.scale*np.min(self.scores_)):
                return self

        return self

    
def stop_early(classifier, **kwargs):
    n_iterations = classifier.n_estimators
    early = EarlyStopping(classifier, 
                          max_n_estimators=n_iterations,
                          # fix the dataset used for testing by currying
                          scorer=partial(one_minus_roc, X_test, y_test),
                          **kwargs)
    early.fit(X_train, y_train)
    plt.figure(figsize=(12,8))
    plt.plot(np.arange(1, len(early.scores_)+1), early.scores_)
    plt.title("Learning curve by n_estimators" ,fontsize=18)
    plt.xlabel("number of estimators" ,fontsize=18)
    plt.ylabel("1 - area under ROC" ,fontsize=18)
    
    
    
def deviance_plot(est, X_test, y_test, ax=None, label='', train_color='#2c7bb6', test_color='#d7191c', alpha=1.0):
    test_dev = np.empty(n_estimators)
    for i, pred in enumerate(est.staged_predict(X_test)):
        test_dev[i] = est.loss_(y_test, pred)
        if ax is None:
            fig = plt.figure(figsize=(8, 5))
            ax = plt.gca()
            ax.plot(np.arange(n_estimators) + 1, test_dev, color=test_color, label='Test %s' % label,linewidth=2, alpha=alpha)
            ax.plot(np.arange(n_estimators) + 1, est.train_score_, color=train_color, label='Train %s' % label, linewidth=2, alpha=alpha)
            ax.set_ylabel('Error')
            ax.set_xlabel('n_estimators')
            ax.set_ylim((0, 2))
            return test_dev, ax 
        
        
def _cross_entropy_like_loss(model, input_data, targets, num_estimators):
    loss = np.zeros((num_estimators, 1))
    for index, predict in enumerate(model.staged_predict_proba(input_data)):
        loss[index, :] = -np.sum(np.log([predict[sample_num, class_num-1]
                                         for sample_num, class_num in enumerate(targets)])) 
        print(f'ce loss {index}:{loss[index, :]}')
    return loss


def log_loss_plot(est, X_train, y_train, X_test, y_test, n_estimators):
    tr_loss_ce = _cross_entropy_like_loss(est, X_train, y_train, n_estimators)
    test_loss_ce = _cross_entropy_like_loss(est, X_test, y_test, n_estimators)
    plt.figure(12,8)
    plt.plot(np.arange(n_estimators) + 1, tr_loss_ce, '-r', label='training_loss_ce')
    plt.plot(np.arange(n_estimators) + 1, test_loss_ce, '-b', label='val_loss_ce')
    plt.ylabel('Error')
    plt.xlabel('num_components')
    plt.legend(loc='upper right')
    
    