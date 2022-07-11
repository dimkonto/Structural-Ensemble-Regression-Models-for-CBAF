# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 13:34:05 2022

@author: jimak
"""

import pandas as pd
import time
import collections
from sys import argv
from sys import exit
import xlsxwriter
from pandas import ExcelWriter
from pandas import ExcelFile
from epftoolbox.evaluation import MASE

import numpy as np
from numpy import array
from numpy import split
from sklearn import preprocessing
import scipy
from tabulate import tabulate
from matplotlib import pyplot as pp
import datetime
import statistics
import math
from math import log,sqrt
from scipy import stats
from scipy.signal import find_peaks
from scipy.stats import loguniform
from mlxtend.evaluate import bias_variance_decomp

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import SplineTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, HuberRegressor, Lars, Lasso, RANSACRegressor, TheilSenRegressor, BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import VotingRegressor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import LocalOutlierFactor
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD,Adam
from tslearn.utils import to_time_series
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.metrics import soft_dtw

import seaborn as sns
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from keras.callbacks import TensorBoard

from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm
import statsmodels.formula.api as smf
from itertools import product
from scipy.optimize import differential_evolution
from dtw import dtw,accelerated_dtw
from scipy import signal

#METHODS
#def 

def Diff(li1, li2):
    return list(set(li1) - set(li2)) + list(set(li2) - set(li1))

def calc_soft_dtw(col1,col2):
    timeseries1=to_time_series(col1)
    timeseries2=to_time_series(col2)
    soft_score=soft_dtw(timeseries1, timeseries2, gamma=.1)
    return soft_score

def lagfeatures(data,featurename, nlags):
    for i in range(nlags):
        shiftpos = i+1
        data[str(featurename)+'lag_'+str(shiftpos)] = data[featurename].shift(shiftpos)
    return data

def calculate_mape(Y_actual, Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

def model_evaluation(actual_inverted, predicted_inverted, experiment):
    mape_dnn = calculate_mape(actual_inverted, predicted_inverted)
    mse_dnn = mean_squared_error(actual_inverted, predicted_inverted)
    rmse_dnn = math.sqrt(mse_dnn)
    mae_dnn = mean_absolute_error(actual_inverted, predicted_inverted)

    print(mape_dnn, mse_dnn, rmse_dnn, mae_dnn)
    
    #if experiment!=None:
    
    return mape_dnn, mse_dnn, rmse_dnn, mae_dnn

def feature_ranking(X,y):
    model = LinearRegression()
    # fit the model
    model.fit(X, y)
    # get importance
    importance = model.coef_
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    pp.bar([x for x in range(len(importance))], importance)
    pp.show()
    
    model = DecisionTreeRegressor()
    # fit the model
    model.fit(X, y)
    # get importance
    importance = model.feature_importances_
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
        # plot feature importance
    pp.bar([x for x in range(len(importance))], importance)
    pp.show()
    
    model = RandomForestRegressor()
    # fit the model
    model.fit(X, y)
    # get importance
    importance = model.feature_importances_
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    pp.bar([x for x in range(len(importance))], importance)
    pp.show()
    
    model = XGBRegressor()
    # fit the model
    model.fit(X, y)
    # get importance
    importance = model.feature_importances_
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    pp.bar([x for x in range(len(importance))], importance)
    pp.show()

def select_most_important(X_train, y_train, X_test):
	# configure to select a subset of features
	fs = SelectFromModel(LinearRegression(), max_features=10)
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

#DEFINE LIST OF 11 ESTIMATORS
def get_models():
    models = list()
    models.append(XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8))
    models.append(LinearRegression())
    models.append(make_pipeline(StandardScaler(), LinearSVR(random_state=0, tol=1e-5)))
    models.append(make_pipeline(StandardScaler(),SGDRegressor(max_iter=1000, tol=1e-3)))
    models.append(HuberRegressor())
    models.append(Lars(n_nonzero_coefs=1, normalize=False))
    models.append(make_pipeline(StandardScaler(),Lasso(max_iter=2000 ,tol=1e-5, alpha=0.1)))
    models.append(Ridge(alpha=1.0))
    models.append(TheilSenRegressor(random_state=0))
    models.append(BayesianRidge())
    models.append(KNeighborsRegressor(n_neighbors=4))
    #models.append(MLPRegressor(random_state=1, max_iter=500, early_stopping=True)) #extra model to check MLP
    return models

#DEFINE FUNCTION THAT DOES HYPERPARAMETER TUNING RANDOMIZEDSEARCHCV BASED ON ESTINDEX -> TAKE PARAMS AND RETURN BEST ESTIMATOR TO REPLACE THE DICTIONARY ENTRY
def tune_models(model_list, X_train, y_train, tscv):
    #optimal_model_list = []
    
    for index, model in enumerate(model_list):
        print(model[0], model[1])
        if model[0] == 'est 0': #XGBOOST
            print('True')
            
            xgb_params = {
             'learning_rate' : [0.05,0.10,0.15,0.20],
             'max_depth' : [ 4, 5, 6, 8],
             'min_child_weight' : [ 1, 3, 5, 7 ],
             'gamma': [ 0.0, 0.1, 0.2 , 0.3],
             'colsample_bytree' : [ 0.4, 0.5 , 0.7, 0.8 ]
            }
            
            reg = model[1]
            #print(reg.get_params().keys())
            randomSearch = RandomizedSearchCV(estimator=reg, n_jobs=-1,
            	cv=tscv, param_distributions=xgb_params,
            	scoring="neg_mean_squared_error")
            searchResults = randomSearch.fit(X_train, y_train)
            #print("[INFO] Evaluating best XGB")
            bestModel = searchResults.best_estimator_
            #print(bestModel)
            optimal_model_list = list(model)
            optimal_model_list[1] = bestModel
            model = tuple(optimal_model_list)
            model_list[index] = model
            
        if model[0] == 'est 2': #LinearSVR
            tolerance = loguniform(1e-6,1e-3)
            C = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]
            svr_params = dict(linearsvr__tol=tolerance, linearsvr__C=C)
            
            reg = model[1]
            print(reg.get_params().keys())
            randomSearch = RandomizedSearchCV(estimator=reg, n_jobs=-1,
            	cv=tscv, param_distributions=svr_params,
            	scoring="neg_mean_squared_error")
            searchResults = randomSearch.fit(X_train, y_train)
            print("[INFO] Evaluating best SVR")
            bestModel = searchResults.best_estimator_
            #print(bestModel)
            optimal_model_list = list(model)
            optimal_model_list[1] = bestModel
            model = tuple(optimal_model_list)
            model_list[index] = model
        
        if model[0] == 'est 3': #SGD
            sgd_params = {
             'sgdregressor__learning_rate' : ['constant', 'optimal', 'invscaling', 'adaptive'] ,
             'sgdregressor__alpha' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
             'sgdregressor__max_iter' : [1000, 2000, 3000, 4000],
             'sgdregressor__loss' : ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
             'sgdregressor__tol': loguniform(1e-6,1e-3),
             'sgdregressor__eta0' : [0.01,0.05,0.1,0.2],
             'sgdregressor__penalty' : ['l1', 'l2', 'elasticnet']
            }

            reg = model[1]
            print(reg.get_params().keys())
            randomSearch = RandomizedSearchCV(estimator=reg, n_jobs=-1,
            	cv=tscv, param_distributions=sgd_params,
            	scoring="neg_mean_squared_error")
            searchResults = randomSearch.fit(X_train, y_train)
            print("[INFO] Evaluating best SGD")
            bestModel = searchResults.best_estimator_
            #print(bestModel)
            optimal_model_list = list(model)
            optimal_model_list[1] = bestModel
            model = tuple(optimal_model_list)
            model_list[index] = model
            
        if model[0] == 'est 4':    #HUBER
            huber_params = {
             'max_iter' : [100, 500, 1000] ,
             'alpha' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
             'epsilon' : [1.05, 1.15, 1.25, 1.35, 1.45, 1.55],
             'tol': loguniform(1e-6,1e-3)
            }
            
            reg = model[1]
            print(reg.get_params().keys())
            randomSearch = RandomizedSearchCV(estimator=reg, n_jobs=-1,
            	cv=tscv, param_distributions=huber_params,
            	scoring="neg_mean_squared_error")
            searchResults = randomSearch.fit(X_train, y_train)
            print("[INFO] Evaluating best HUBER")
            bestModel = searchResults.best_estimator_
            print(bestModel)
            optimal_model_list = list(model)
            optimal_model_list[1] = bestModel
            model = tuple(optimal_model_list)
            model_list[index] = model
            
        if model[0] == 'est 5': #LARS
           lars_params = {
            'n_nonzero_coefs' : [1, 10, 20, 30, 50, 100, 200, 300, 400, 500] ,
            'normalize' : [False],
           }


           reg = model[1]
           print(reg.get_params().keys())
           randomSearch = RandomizedSearchCV(estimator=reg, n_jobs=-1,
           	cv=tscv, param_distributions=lars_params,
           	scoring="neg_mean_squared_error")
           searchResults = randomSearch.fit(X_train, y_train)
           print("[INFO] Evaluating best LARS")
           bestModel = searchResults.best_estimator_
           print(bestModel)
           optimal_model_list = list(model)
           optimal_model_list[1] = bestModel
           model = tuple(optimal_model_list)
           model_list[index] = model
        
        if model[0] == 'est 6': #LASSO
            lasso_params = {
             'lasso__max_iter' : [1000, 2000, 3000, 4000] ,
             'lasso__alpha' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
             'lasso__tol': loguniform(1e-6,1e-3)
            }

            reg = model[1]
            print(reg.get_params().keys())
            randomSearch = RandomizedSearchCV(estimator=reg, n_jobs=-1,
            	cv=tscv, param_distributions=lasso_params,
            	scoring="neg_mean_squared_error")
            searchResults = randomSearch.fit(X_train, y_train)
            print("[INFO] Evaluating best LASSO")
            bestModel = searchResults.best_estimator_
            print(bestModel)
            optimal_model_list = list(model)
            optimal_model_list[1] = bestModel
            model = tuple(optimal_model_list)
            model_list[index] = model
            #exit()
            
        if model[0] == 'est 7': #Ridge
            ridge_params = {
             'alpha' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 500, 1000, 1500]
            }

            reg = model[1]
            print(reg.get_params().keys())
            randomSearch = RandomizedSearchCV(estimator=reg, n_jobs=-1,
            	cv=tscv, param_distributions=ridge_params,
            	scoring="neg_mean_squared_error")
            searchResults = randomSearch.fit(X_train, y_train)
            print("[INFO] Evaluating best LASSO")
            bestModel = searchResults.best_estimator_
            print(bestModel)
            optimal_model_list = list(model)
            optimal_model_list[1] = bestModel
            model = tuple(optimal_model_list)
            model_list[index] = model
        
        if model[0] == 'est 9': #BayesianRidge
            bayesianridge_params = {
             'alpha_1' : loguniform(1e-8,1e-3),
             'alpha_2' : loguniform(1e-8,1e-3),
             'lambda_1' : loguniform(1e-8,1e-3),
             'lambda_2' : loguniform(1e-8,1e-3)
            }

            reg = model[1]
            print(reg.get_params().keys())
            randomSearch = RandomizedSearchCV(estimator=reg, n_jobs=-1,
            	cv=tscv, param_distributions=bayesianridge_params,
            	scoring="neg_mean_squared_error")
            searchResults = randomSearch.fit(X_train, y_train)
            print("[INFO] Evaluating best LASSO")
            bestModel = searchResults.best_estimator_
            print(bestModel)
            optimal_model_list = list(model)
            optimal_model_list[1] = bestModel
            model = tuple(optimal_model_list)
            model_list[index] = model
        
        if model[0] == 'est 10': #KNR
            knr_params = {
                'leaf_size' : list(range(1,50)) ,
                'n_neighbors' : list(range(1,30)) ,
                'p': [1,2]
            }

            reg = model[1]
            print(reg.get_params().keys())
            randomSearch = RandomizedSearchCV(estimator=reg, n_jobs=-1,
            	cv=tscv, param_distributions=knr_params,
            	scoring="neg_mean_squared_error")
            searchResults = randomSearch.fit(X_train, y_train)
            print("[INFO] Evaluating best LASSO")
            bestModel = searchResults.best_estimator_
            print(bestModel)
            optimal_model_list = list(model)
            optimal_model_list[1] = bestModel
            model = tuple(optimal_model_list)
            model_list[index] = model
        
        
    
    return model_list


def stpeak_estimator_info(X_train, y_train):
    
    X_seg_train, X_seg_test = X_train[:int(0.8*X_train.shape[0]),:], X_train[int(0.8*X_train.shape[0]):,:]
    y_seg_train, y_seg_test = y_train[:int(0.8*y_train.shape[0])], y_train[int(0.8*y_train.shape[0]):]
    
    print("PEAK ESTIMATOR SHAPES")
    print(X_seg_train.shape, X_seg_test.shape)
    print(y_seg_train.shape, y_seg_test.shape)
    
    peaks, properties = find_peaks(y_seg_test, height=0)
    #peak indices
    print(peaks, peaks.shape)
    #peak values y_seg_test[peaks] or properties["peak_heights"]
    
    #FIND PREDICTOR BODY
    prediction_vectors = list()
    peak_indices_vectors = list()
    peak_values_vectors = list ()
    for model in get_models():
        model.fit(X_seg_train,y_seg_train)
        yhat = model.predict(X_seg_test)
        prediction_vectors.append(yhat)
        #INSERT INDICES AND VALUES OF PEAKS FOR EACH ESTIMATOR
        pred_peaks, pred_properties = find_peaks(yhat, height=0)
        peak_indices_vectors.append(pred_peaks)
        peak_values_vectors.append(yhat[pred_peaks])
        
    
    min_mae = float('inf')
    min_index = -1
    info_list = list()
    for index, prediction in enumerate(prediction_vectors):
        #index = prediction.id
        print(index)
        #FOR PREDICTOR BODY
        mape_st, mse_st, rmse_st, mae_st = model_evaluation(y_seg_test,prediction, experiment=None)
        
        if mae_st<min_mae:
            min_mae = mae_st
            min_index = index
                
    print("RESULTING PREDICTOR BODY: ", min_index, min_mae) # 1] INFO: INDEX FOR BEST OVERALL PREDICTOR
    #info_list.append(min_index)
    
    
    print("PEAK INTERSECTIONS")
    max_intercept=0
    max_intercept_index= -1
    
    for peak_index, peak_prediction_indices in enumerate(peak_indices_vectors):
        
        #print(peak_indices_vectors[peak_index])
        
        if np.intersect1d(peaks,peak_prediction_indices).shape[0]>max_intercept:
            max_intercept = np.intersect1d(peaks,peak_prediction_indices).shape[0]
            max_intercept_index = peak_index
        #print(np.intersect1d(peaks,peak_prediction_indices), np.intersect1d(peaks,peak_prediction_indices).shape)
        
        
    print("BEST PEAK-MATCHING INDICE PREDICTOR: ", max_intercept_index, max_intercept) # 2] INFO: INDEX FOR MAX INTERCEPT INDICES
    #info_list.append(max_intercept_index)
    
    min_mae_p = float('inf')
    min_index_p = -1
    min_mae_ony = float('inf')
    min_index_ony = -1
    min_mae_exy = float('inf')
    min_index_exy = -1
    mae_peaks = list()
    
    #PERSPECTIVE EVALUATION FOR PEAKS
    for peak_index, peak_prediction_indices in enumerate(peak_indices_vectors):
        
        predictor = prediction_vectors[peak_index]
        
        #intersection_indices = np.intersect1d(peaks,peak_prediction_indices)
        #FOR EVALUATION ON PEAKS AS DETECTED BY ESTIMATORS
        value_series = peak_values_vectors[peak_index]
        print(y_seg_test[peak_prediction_indices], value_series)
        print(y_seg_test[peak_prediction_indices].shape, value_series.shape)
        
        mape_p, mse_p, rmse_p, mae_p = model_evaluation(y_seg_test[peak_prediction_indices],value_series, experiment=None)
        mae_peaks.append(mae_p)
        if mae_p<min_mae_p:
            min_mae_p = mae_p
            min_index_p = peak_index
        
        #FOR EVALUATION ON TOTAL PEAKS FOUND ON Y
        
        mape_ony, mse_ony, rmse_ony, mae_ony = model_evaluation(y_seg_test[peaks],predictor[peaks], experiment=None)
        #mae_peaks.append(mae_p)
        if mae_ony<min_mae_ony:
            min_mae_ony = mae_ony
            min_index_ony = peak_index
        
        #FOR EVALUATION OF PEAKS FOUND ONLY IN Y AND NOT FOUND IN PREDS
        peaks_only_y = Diff(peaks, peak_prediction_indices) #A-B sets
        mape_exy, mse_exy, rmse_exy, mae_exy = model_evaluation(y_seg_test[peaks_only_y],predictor[peaks_only_y], experiment=None)
        if mae_exy<min_mae_exy:
            min_mae_exy = mae_exy
            min_index_exy = peak_index
        
        
    print("BEST PEAK VALUE PERFORMANCE ON PREDICTOR INDICES: ", min_index_p, min_mae_p) # 3] INFO: INDEX FOR BEST PEAK PERFORMANCE
    info_list.append(min_index_p)
    
    print("BEST TOTAL PEAK VALUE PERFORMANCE: ", min_index_ony, min_mae_ony) # 3] INFO: INDEX FOR BEST PEAK PERFORMANCE
    info_list.append(min_index_ony)
    
    print("BEST PERFORMANCE ON PEAKS EXCLUSIVE TO Y: ", min_index_exy, min_mae_exy) # 3] INFO: INDEX FOR BEST PEAK PERFORMANCE
    info_list.append(min_index_exy)
    
    #info_list = list(dict.fromkeys(info_list)) #Remove duplicates if needed
    print(info_list)
    
    #PERSPECTIVE EVALUATION FOR NON-PEAKS
    all_indices = np.arange(y_seg_test.shape[0])
    non_peak_indices = Diff(all_indices, peaks)
    min_mae_nonp = float('inf')
    min_index_nonp = -1
    min_mae_nonp_est = float('inf')
    min_index_nonp_est = -1
    min_mae_nonp_exy = float('inf')
    min_index_nonp_exy = -1
    info_list_nonp = list()
    for index, prediction in enumerate(prediction_vectors):
        
        #FOR EVALUATION ON TOTAL NON-PEAKS FOUND ON Y
        mape_nonp, mse_nonp, rmse_nonp, mae_nonp = model_evaluation(y_seg_test[non_peak_indices],prediction[non_peak_indices], experiment=None)
        if mae_nonp<min_mae_nonp:
            min_mae_nonp = mae_nonp
            min_index_nonp = index
            
        #NON PEAKS FOUND IN ESTIMATORS
        nonp_est_indices = Diff(all_indices,peak_indices_vectors[index])
        mape_nonp_est, mse_nonp_est, rmse_nonp_est, mae_nonp_est = model_evaluation(y_seg_test[nonp_est_indices],prediction[nonp_est_indices], experiment=None)
        if mae_nonp_est<min_mae_nonp_est:
            min_mae_nonp_est = mae_nonp_est
            min_index_nonp_est = index
        
        #NON PEAKS FOUND ONLY IN Y AND NOT IN ESTIMATORS
        non_peaks_only_y = Diff(non_peak_indices, nonp_est_indices)
        mape_nonp_exy, mse_nonp_exy, rmse_nonp_exy, mae_nonp_exy = model_evaluation(y_seg_test[non_peaks_only_y],prediction[non_peaks_only_y], experiment=None)
        if mae_nonp_exy<min_mae_nonp_exy:
            min_mae_nonp_exy = mae_nonp_exy
            min_index_nonp_exy = index
        
    
    print("BEST PERFORMANCE ON TOTAL NON PEAKS DETECTED ON Y: ", min_index_nonp, min_mae_nonp) # 3] INFO: INDEX FOR BEST PEAK PERFORMANCE
    info_list_nonp.append(min_index_nonp)
    
    print("BEST PERFORMANCE ON NON PEAKS AS DETECTED BY ESTIMATORS: ", min_index_nonp_est, min_mae_nonp_est) # 3] INFO: INDEX FOR BEST PEAK PERFORMANCE
    info_list_nonp.append(min_index_nonp_est)
    
    print("BEST PERFORMANCE ON NON PEAKS EXCLUSIVELY DETECTED ON Y: ", min_index_nonp_exy, min_mae_nonp_exy) # 3] INFO: INDEX FOR BEST PEAK PERFORMANCE
    info_list_nonp.append(min_index_nonp_exy)
    
    print(info_list_nonp)
    """
    #### TEST PERFORMANCE OF RESULT FUSION [WILL NOT BE INCLUDED IN FINAL CODE]
    final_estimator_values = prediction_vectors[min_index]
    best_peak_estimator = prediction_vectors[min_index_ony]
    
    best_nonpeak_estimator = prediction_vectors[min_index_nonp]
    
    finalest_peaks, finalest_properties = find_peaks(final_estimator_values, height=0)
    finalest_nonpeaks, finalest_nonpeak_properties = find_peaks(best_nonpeak_estimator, height=0)
    bestest_peaks, bestest_peaks_properties = find_peaks(best_peak_estimator, height=0)
    
    finalest_nonpeaks = Diff(all_indices, finalest_peaks)
    finalest_best_nonpeaks = Diff(all_indices, finalest_nonpeaks)
    
    #VALUE REPLACEMENT
    #replace common peaks
    peak_replacement_indices = np.intersect1d(finalest_peaks,bestest_peaks)
    final_estimator_values[peak_replacement_indices] = best_peak_estimator[peak_replacement_indices]
    
    #replace common nonpeaks
    non_peak_replacement_indices = np.intersect1d(finalest_nonpeaks,finalest_best_nonpeaks)
    final_estimator_values[non_peak_replacement_indices] = best_nonpeak_estimator[non_peak_replacement_indices]
    
    mape_final, mse_final, rmse_final, mae_final = model_evaluation(y_seg_test,final_estimator_values, experiment=None)
    print(mape_final, mse_final, rmse_final, mae_final)
    """
    
    #SEND INFO ABOUT BEST OVERALL ESTIMATOR, LIST OF PEAK ESTIMATORS, LIST OF NONPEAK ESTIMATORS TO APPLY
    return min_index, info_list, info_list_nonp
    

def stpeak_estimator_apply(X_train, X_test, y_train, y_test,overall_best, peak_list, nonpeak_list):
    print(overall_best, peak_list, nonpeak_list)
    
    """
    #always enable for experiments, disable for testing operations
    prediction_vectors = list()
    peak_indices_vectors = list()
    peak_values_vectors = list ()
    nonpeak_indices_vectors = list()
    nonpeak_values_vectors = list()
    for model in get_models():
        model.fit(X_train,y_train)
        yhat = model.predict(X_test)
        mape_model, mse_model, rmse_model, mae_model = model_evaluation(y_test,yhat, experiment=None)
        prediction_vectors.append(yhat)
        #INSERT INDICES AND VALUES OF PEAKS FOR EACH ESTIMATOR
        pred_peaks, pred_properties = find_peaks(yhat, height=0)
        peak_indices_vectors.append(pred_peaks)
        peak_values_vectors.append(yhat[pred_peaks])
    """
    
    #STRATEGY #1 SINGLE FUSION MODEL OF JOINT LIST OF ESTIMATORS
    fusion_list = peak_list + nonpeak_list #Initially considering all ests for frequencies
    
    #find frequencies for weighting
    frequency = collections.Counter(fusion_list)
    print(dict(frequency))
    
    #CHOOSE DIFFERENT FUSION LIST OF ESTIMATORS
    #fusion_list = list(dict.fromkeys(nonpeak_list)) #performs better
    #fusion_list = list(dict.fromkeys(peak_list))
    fusion_list = list(dict.fromkeys(fusion_list))
    
    print(fusion_list)
    
    #DERIVE WEIGHTS
    freq_weights = []
    for index, fusion_label in enumerate(fusion_list):
        for est_label, est_freq in dict(frequency).items():
            if est_label == fusion_label:
                freq_weights.append(est_freq)
                
    print(freq_weights) #Must be sorted accordingly first
    
    #SORT WEIGHTS BASED ON EST LABELS
    freq_weights = [fw for _,fw in sorted(zip(fusion_list, freq_weights))]
    print("SORTED WEIGHTS")
    print(freq_weights)
    
    #exit()
    
    ##CREATE STACKING MODEL
    level0 = list()
    level1 = LinearRegression() #OR TRY THE OVERALL BEST FOR LEVEL 1
    #level1 = MLPRegressor(random_state=1, max_iter=2000, early_stopping=True)
    for index, estimator in enumerate(get_models()):
        if (index in fusion_list):
            level0.append(('est '+str(index), estimator))
        #if index == overall_best:
            #level1 = estimator
    
    print(level0)
    print(level1)
    
    ###TRY OUTLIER DETECTION ON THE STACKED REGRESSOR
    #iso = IsolationForest(contamination=0.1) #ISOLATION FOREST ALGORITHM
    #iso = OneClassSVM(nu=0.01) #ELLIPTIC ENVELOPE
    #y_iso = iso.fit_predict(X_train)
    #mask = y_iso != -1
    #X_train, y_train = X_train[mask, :], y_train[mask]
    ### end of outlier detection layer
    
    #implement timeseries cv
    tscv = TimeSeriesSplit(n_splits=5)
    print(tscv)
    for train, test in tscv.split(X_train):
        print("%s %s" % (train, test))
        
    for train, test in tscv.split(y_train):
        print("%s %s" % (train, test))    
    
    #STRUCTURAL ENSEMBLE REGRESSORS
    
    #WITH HYPERPARAMETER TUNING
    model_list = tune_models(level0, X_train, y_train, tscv)
    print(model_list)
    #exit()
    #DEFAULT STACKING REGRESSION
    #reg = StackingRegressor(estimators=model_list,final_estimator=level1) #level0 on default, model_list on tuning
    #y_fusion = reg.fit(X_train, y_train).predict(X_test)
    #print("FUSION MODEL RESULTS")
    #mape_fusion, mse_fusion, rmse_fusion, mae_fusion = model_evaluation(y_test,y_fusion, experiment=None)
    #END OF SKLEARN STACKING REGRESSOR
    
    #STRATEGY: IMPLEMENT VOTING REGRESSOR WITH DEFAULT AND CUSTOM WEIGHT SCHEME
    
    #DEFAULT WEIGHT SCHEME VOTING
    #voting_reg = VotingRegressor(estimators=model_list)
    #y_fusion = voting_reg.fit(X_train, y_train).predict(X_test)
    #print("FUSION VOTING MODEL RESULTS")
    #mape_fusion, mse_fusion, rmse_fusion, mae_fusion = model_evaluation(y_test,y_fusion, experiment=None)
    #END OF SKLEARN VOTING BASE REGRESSOR
    
    #OCCURENCE-BASED WEIGHTED VOTING
    voting_reg = VotingRegressor(estimators=model_list, weights = freq_weights)
    y_fusion = voting_reg.fit(X_train, y_train).predict(X_test)
    print("FUSION WEIGHTED VOTING MODEL RESULTS")
    mape_fusion, mse_fusion, rmse_fusion, mae_fusion = model_evaluation(y_test,y_fusion, experiment=None)
    #END OF SKLEARN VOTING WEIGHTED REGRESSOR
    
    """
    #STRATEGY #2: BUILD A STACKING REGRESSOR MANUALLY AND TRY DIFFERENT CV STRATEGIES (E.G WITH TIME SERIES SPLIT CV)
    X_base_train, X_base_test = X_train[:int(0.8*X_train.shape[0]),:], X_train[int(0.8*X_train.shape[0]):,:]
    y_base_train, y_base_test = y_train[:int(0.8*y_train.shape[0])], y_train[int(0.8*y_train.shape[0]):]
    
    y_base_prediction = np.empty((y_base_test.shape[0],0))
    test_features = np.empty((y_test.shape[0],0))
    for index, estimator in enumerate(get_models()):
        if (index in fusion_list):
            estimator.fit(X_base_train,y_base_train)
            prediction = estimator.predict(X_base_test)
            print(prediction)
            y_base_prediction = np.column_stack((y_base_prediction, prediction))
            
            #GET X TEST DATASET OF STACKED MODEL
            test_prediction = estimator.predict(X_test)
            print(test_prediction)
            test_features = np.column_stack((test_features, test_prediction))
    
    
    
    ##INVESTIGATE EFFECT OF INTERACTIONS [did not yield better performance]
    #interactions = PolynomialFeatures(degree=3)
    #y_base_prediction = interactions.fit_transform(y_base_prediction)
    #test_features = interactions.fit_transform(test_features)
    
    #Evaluate feature importance (including interactions)
    #feature_ranking(y_base_prediction, y_base_test)
    #exit()
    #Select most important features
    #y_base_prediction, test_features, fs=select_most_important(y_base_prediction, y_base_test, test_features) #does not yield better performance
    #Experiment: Include original features [did not yield better scores]
    #y_base_prediction = np.column_stack((y_base_prediction, X_base_test))
    #test_features = np.column_stack((test_features, X_test))
    
    #Investigate B-Splines (Periodic) for the predicted features
    #level1 = make_pipeline(SplineTransformer(n_knots=32, degree=3),  Ridge(alpha=1e-3))
    #level1.fit(y_base_prediction, y_base_test)
    #y_meta_pred = level1.predict(test_features)
    
    #Investigate Isotonic Regression
    #level1 = IsotonicRegression().fit(y_base_prediction, y_base_test)
    
    #y_meta_pred = level1.predict(test_features)
    
    print(y_base_prediction) #y_base_prediction = X_train of level1 , test_features = X_test of level1
    print(y_base_prediction.shape)
    
    #TRY: CREATE LAGS FOR EACH ESTIMATOR FEATURE
    
    input_df_train = pd.DataFrame(y_base_prediction)
    input_df_test = pd.DataFrame(test_features)
    output_df_train = pd.DataFrame(y_base_test)
    output_df_test = pd.DataFrame(y_test)
    
    #Create est lags
    for col in input_df_train.columns:
        for i in range(4):
            pos = i+1
            input_df_train[str(col)+'lag '+str(pos)] = input_df_train[col].shift(pos)
            input_df_test[str(col)+'lag '+str(pos)] = input_df_test[col].shift(pos)
    
    input_df_train = input_df_train.dropna()
    input_df_test = input_df_test.dropna()
    
    output_df_train = output_df_train.iloc[4:,:]
    output_df_test = output_df_test.iloc[4:,:]
    
    print(input_df_train)
    print(input_df_test)
    print(output_df_train)
    print(output_df_test)
    
    #Feature selection WITH PCA on est lags [PCA NOT YIELDING BETTER PERFORMANCE]
    #x_scaled_train = StandardScaler().fit_transform(input_df_train.to_numpy())
    #x_scaled_test = StandardScaler().fit_transform(input_df_test.to_numpy())# standardizing the features
    #print(x_scaled_train.shape)
    #print(np.mean(x_scaled_train),np.std(x_scaled_train))
    #pca_ests = PCA(n_components=3) #n should be as many as the features of the dataset to see all components before keeping based on variance ratio
    #X_train_pca = pca_ests.fit_transform(x_scaled_train)
    #X_test_pca = pca_ests.transform(x_scaled_test)
    #print(pca_ests.explained_variance_ratio_)
    #print(pca_ests.singular_values_)
    
    #level1 = LinearRegression().fit(X_train_pca, output_df_train.to_numpy())
    #y_meta_pred=level1.predict(X_test_pca)
    #print(y_meta_pred.shape, output_df_test.to_numpy().shape)
    #print("FUSION MODEL RESULTS")
    #mape_fusion, mse_fusion, rmse_fusion, mae_fusion = model_evaluation(output_df_test.to_numpy(),y_meta_pred, experiment=None)
    
    #Find best of all lags
    #X_train_fs, X_test_fs, fs=select_most_important(input_df_train.to_numpy(), output_df_train.to_numpy(), input_df_test.to_numpy())
    #level1 = LinearRegression().fit(X_train_fs, output_df_train.to_numpy())
    #y_meta_pred=level1.predict(X_test_fs)
    #print("FUSION MODEL RESULTS")
    #mape_fusion, mse_fusion, rmse_fusion, mae_fusion = model_evaluation(output_df_test.to_numpy(),y_meta_pred, experiment=None)
    
    
    #LINES FOR NORMAL EST LAGS
    #level1 = LinearRegression().fit(input_df_train.to_numpy(), output_df_train.to_numpy())
    #y_meta_pred=level1.predict(input_df_test.to_numpy())
    #print("FUSION MODEL RESULTS")
    #mape_fusion, mse_fusion, rmse_fusion, mae_fusion = model_evaluation(output_df_test.to_numpy(),y_meta_pred, experiment=None)
    #exit()
    """
    
    """
    #IMPLEMENT CV STACKING REGRESSOR NO LAGS
    bestrmse = float('inf')
    for train, test in tscv.split(X_train):
        print("%s %s" % (train, test)) #INDICES FOR TRAIN TEST SPLIT [TRAIN SET IS DIVIDED INTO VALIDATION FOLDS ]
        xcv_base_train = X_train[train,:]
        ycv_base_train = y_train[train]
        xcv_base_test = X_train[test,:]
        ycv_base_test = y_train[test]
        ycv_base_prediction = np.empty((ycv_base_test.shape[0],0))
        test_features = np.empty((y_test.shape[0],0))
        for index, estimator in enumerate(model_list):
            if (index in fusion_list):
                estimator[1].fit(xcv_base_train,ycv_base_train)
                prediction = estimator[1].predict(xcv_base_test)
                #print(prediction)
                ycv_base_prediction = np.column_stack((ycv_base_prediction, prediction))
                
                #GET X TEST DATASET OF STACKED MODEL
                test_prediction = estimator[1].predict(X_test)
                #print(test_prediction)
                test_features = np.column_stack((test_features, test_prediction))
        
        level1 = LinearRegression().fit(ycv_base_prediction, ycv_base_test)
        y_meta_pred=level1.predict(test_features)
        print("FUSION MODEL RESULTS")
        mape_fusion, mse_fusion, rmse_fusion, mae_fusion = model_evaluation(y_test,y_meta_pred, experiment=None)
        if rmse_fusion<bestrmse:
            bestrmse = rmse_fusion
            besty_meta_pred = y_meta_pred
    
    
    y_fusion = besty_meta_pred
    #exit()
    """
    
        #level1 = LinearRegression().fit(X_train[train,:], y_base_test)
    
    #BASE LEVEL 1 LR (NO LAGS)
    #level1 = LinearRegression().fit(y_base_prediction, y_base_test)
    
    #y_meta_pred=level1.predict(test_features)
    #print("FUSION MODEL RESULTS")
    #mape_fusion, mse_fusion, rmse_fusion, mae_fusion = model_evaluation(y_test,y_meta_pred, experiment=None)
    #y_fusion = y_meta_pred
    #exit()
    
    """
    ###STRATEGY #3: UTILIZE PEAK & NON-PEAK LISTS INDIVIDUALLY
    level0 = list()
    level1 = LinearRegression() #OR TRY THE OVERALL BEST FOR LEVEL 1
    #level1 = MLPRegressor(random_state=1, max_iter=2000, early_stopping=True)
    for index, estimator in enumerate(get_models()):
        if (index in peak_list):
            level0.append(('est '+str(index), estimator))
        #if index == overall_best:
            #level1 = estimator
    
    print(level0)
    print(level1)
    
    reg = StackingRegressor(estimators=level0,final_estimator=level1) #level0 on default, model_list on tuning
    y_fusion_peak = reg.fit(X_train, y_train).predict(X_test)
    #print("FUSION MODEL RESULTS")
    #mape_fusion, mse_fusion, rmse_fusion, mae_fusion = model_evaluation(y_test,y_fusion, experiment=None)
    
    level0 = list()
    level1 = LinearRegression() #OR TRY THE OVERALL BEST FOR LEVEL 1
    #level1 = MLPRegressor(random_state=1, max_iter=2000, early_stopping=True)
    for index, estimator in enumerate(get_models()):
        if (index in nonpeak_list):
            level0.append(('est '+str(index), estimator))
        #if index == overall_best:
            #level1 = estimator
    
    print(level0)
    print(level1)
    
    reg = StackingRegressor(estimators=level0,final_estimator=level1) #level0 on default, model_list on tuning
    y_fusion_nonpeak = reg.fit(X_train, y_train).predict(X_test)
    #print("FUSION MODEL RESULTS")
    #mape_fusion, mse_fusion, rmse_fusion, mae_fusion = model_evaluation(y_test,y_fusion, experiment=None)
    
    print(y_test, y_fusion_peak, y_fusion_nonpeak)
    
    for index, estimator in enumerate(get_models()):
        if (index == overall_best):
            y_hat_overall = estimator.fit(X_train, y_train).predict(X_test)
            peaks_overall, properties = find_peaks(y_hat_overall, height=0)
            #peak indices
            print(peaks_overall, peaks_overall.shape)
            all_indices = np.arange(y_hat_overall.shape[0])
            non_peak_indices = Diff(all_indices, peaks_overall)
            
            #STRAIGHTFORWARD REPLACEMENT [yields slightly better results]
            y_hat_overall[peaks_overall] = y_fusion_peak[peaks_overall]
            y_hat_overall[non_peak_indices] = y_fusion_nonpeak[non_peak_indices]
            
            #COMPARATIVE REPLACEMENT [NOT CONSISTENTLY BETTER RESULTS]
            #for k in range(y_hat_overall.shape[0]):
            #    if k in peaks_overall:
            #        if y_hat_overall[k]<y_fusion_peak[k]:
            #            y_hat_overall[k] = y_fusion_peak[k]
                
            #    if k in non_peak_indices:
            #        if y_hat_overall[k]>y_fusion_nonpeak[k]:
            #            y_hat_overall[k] = y_fusion_nonpeak[k]
    
    print(y_hat_overall)
    mape_fusion, mse_fusion, rmse_fusion, mae_fusion = model_evaluation(y_test,y_fusion_peak, experiment=None)
    mape_fusion, mse_fusion, rmse_fusion, mae_fusion = model_evaluation(y_test,y_fusion_nonpeak, experiment=None)
    mape_fusion, mse_fusion, rmse_fusion, mae_fusion = model_evaluation(y_test,y_hat_overall, experiment=None)
        
    y_fusion=y_hat_overall
    #exit()
    """
    
    return y_fusion

def estimators(X_train, X_test, y_train, y_test):
    
    #EXPERIMENTS ON X_TRAIN AND Y_TRAIN
    
    ##SETUP STRUCTURAL PEAK ESTIMATOR [ONLY WHEN COMMENTED OUT TO TEST OTHER ESTIMATORS, y_fusion=y_test as a bypass]
    overall_best, peak_list, nonpeak_list=stpeak_estimator_info(X_train, y_train)
    y_fusion=stpeak_estimator_apply(X_train, X_test, y_train, y_test,overall_best, peak_list, nonpeak_list)
    #exit() #Halting execution for testing
    #y_fusion = y_test #bypass line for structural estimation (comment out normally)
    
    #IN CASE WE OUTPUT ONLY THE FUSION MODEL ABOVE (ELSE COMMENT OUT)
    y_hat_xgb = y_test
    y_hat_lr = y_test 
    y_hat_svr = y_test 
    y_hat_sgd = y_test
    y_hat_hub = y_test 
    y_hat_lars = y_test 
    y_hat_clf= y_test 
    y_hat_rdg= y_test 
    y_hat_ts= y_test 
    y_hat_bar= y_test 
    y_hat_neigh = y_test
    
        
    #X_train_peaks=np.full(shape=(errorset.shape[1], trainset_X.shape[1], trainset_X.shape[0]+errorset.shape[0]-1), fill_value=np.nan)
    for l in range(X_train.shape[1]):
        print("X_TRAIN LAG ", l)
        print(X_train[:,l])
        peaks, properties = find_peaks(X_train[:,l], height=0)
        #pp.plot(X_train[:,l])
        #pp.plot(peaks, X_train[:,l][peaks], "x")
        #pp.plot(np.zeros_like(X_train[:,l]), "--", color="gray")
        #pp.show()
        print(X_train[:,l].shape)
        print(properties["peak_heights"])
        print(properties["peak_heights"].shape)
        print(peaks)
        print(peaks.shape)
        print("PEAK VALUES")
        print(X_train[:,l][peaks])
    
    print("Y_TRAIN")
    print(y_train)
    peaks, properties = find_peaks(y_train, height=0)
    #pp.plot(y_train)
    #pp.plot(peaks, y_train[peaks], "x")
    #pp.plot(np.zeros_like(y_train), "--", color="gray")
    #pp.show()
    print(y_train.shape)
    print(properties["peak_heights"])
    print(properties["peak_heights"].shape)
    print(peaks)
    
    ##MLP REGRESSOR
    mlp = MLPRegressor(random_state=1, max_iter=500, early_stopping=True).fit(X_train, y_train)
    y_hat_mlp = mlp.predict(X_test)
    #pp.plot(y_test[0:300],label='Test')
    #pp.plot(y_hat_mlp[0:300], label='Predicted CL MLP')
    #pp.legend(loc = 'best')
    #pp.show()
    print('MLP CLUSTER PREDICTION EVALUATION')
    model_evaluation(y_test,y_hat_mlp, experiment=None)
    """
    
    """
    ##SETUP ESTIMATORS
    ###XGBOOST (1000 or 100 estimators for robust result)
    tscv = TimeSeriesSplit(n_splits=5)
    """
    #WITH HYPERPARAMETER OPT
    xgb_params = {
     'learning_rate' : [0.05,0.10,0.15,0.20],
     'max_depth' : [ 4, 5, 6, 8],
     'min_child_weight' : [ 1, 3, 5, 7 ],
     'gamma': [ 0.0, 0.1, 0.2 , 0.3],
     'colsample_bytree' : [ 0.4, 0.5 , 0.7, 0.8 ]
    }
    model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
    print(model.get_params().keys())
    randomSearch = RandomizedSearchCV(estimator=model, n_jobs=-1,
    	cv=tscv, param_distributions=xgb_params,
    	scoring="neg_mean_squared_error")
    searchResults = randomSearch.fit(X_train, y_train)
    print("[INFO] Evaluating best XGB")
    bestModel = searchResults.best_estimator_
    print(bestModel)
    print("R2: {:.2f}".format(bestModel.score(X_test, y_test)))
    y_hat_xgb=bestModel.predict(X_test)
    print('XGBOOST CLUSTER PREDICTION EVALUATION')
    mape, mse, rmse, mae=model_evaluation(y_test,y_hat_xgb, experiment=None)
    
    #xgb = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
    #result=xgb.fit(X_train,y_train)
    #y_hat_xgb=xgb.predict(X_test)
    #pp.plot(y_test[0:300],label='Test')
    #pp.plot(y_hat_xgb[0:300], label='Predicted CL XGB')
    #pp.legend(loc = 'best')
    #pp.show()
    
    #model_evaluation(y_test,y_hat_xgb, experiment=None)
    """
    #xg.plot_importance(xgb, height=0.9 ,max_num_features = 10)
    #pp.savefig(r'D:\Datasets\dimreduce\charts\xgbimportance.jpg',dpi=300,bbox_inches="tight")
    #pp.show()
    
    """
    ###LINEAR REGRESSION
    lr=LinearRegression().fit(X_train,y_train)
    y_hat_lr=lr.predict(X_test)
    #print(y_hat)
    print('Variance score: {}'.format(lr.score(X_test, y_test)))
    #pp.plot(y_test[0:300],label='Test')
    #pp.plot(y_hat_lr[0:300], label='Predicted CL LR')
    #pp.legend(loc = 'best')
    #pp.show()
    print('LR CLUSTER PREDICTION EVALUATION')
    model_evaluation(y_test,y_hat_lr, experiment=None)
    
    ###SUPPORT VECTOR REGRESSION
    
    tolerance = loguniform(1e-6,1e-3)
    C = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]
    svr_params = dict(linearsvr__tol=tolerance, linearsvr__C=C)

    model = make_pipeline(StandardScaler(), LinearSVR(random_state=0, tol=1e-5))
    print(model.get_params().keys())
    randomSearch = RandomizedSearchCV(estimator=model, n_jobs=-1,
    	cv=tscv, param_distributions=svr_params,
    	scoring="neg_mean_squared_error")
    searchResults = randomSearch.fit(X_train, y_train)
    print("[INFO] Evaluating best SVR")
    bestModel = searchResults.best_estimator_
    print(bestModel)
    print("R2: {:.2f}".format(bestModel.score(X_test, y_test)))
    y_hat_svr=bestModel.predict(X_test)
    #mape, mse, rmse, mae=model_evaluation(y_test,y_hat_svr, experiment=None)
    
    #regr = make_pipeline(StandardScaler(), LinearSVR(random_state=0, tol=1e-5))
    #regr.fit(X_train, y_train)
    #y_hat_svr=regr.predict(X_test)
    
    #pp.plot(y_test[0:300],label='Test')
    #pp.plot(y_hat_svr[0:300], label='Predicted CL SVR')
    #pp.legend(loc = 'best')
    #pp.show()
    print('SVR CLUSTER PREDICTION EVALUATION')
    model_evaluation(y_test,y_hat_svr, experiment=None)
    
    #exit()
    
    ###SGD Regression
    
    sgd_params = {
     'sgdregressor__learning_rate' : ['constant', 'optimal', 'invscaling', 'adaptive'] ,
     'sgdregressor__alpha' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
     'sgdregressor__max_iter' : [1000, 2000, 3000, 4000],
     'sgdregressor__loss' : ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
     'sgdregressor__tol': loguniform(1e-6,1e-3),
     'sgdregressor__eta0' : [0.01,0.05,0.1,0.2],
     'sgdregressor__penalty' : ['l1', 'l2', 'elasticnet']
    }

    model = make_pipeline(StandardScaler(),SGDRegressor(max_iter=1000, tol=1e-3))
    print(model.get_params().keys())
    randomSearch = RandomizedSearchCV(estimator=model, n_jobs=-1,
    	cv=tscv, param_distributions=sgd_params,
    	scoring="neg_mean_squared_error")
    searchResults = randomSearch.fit(X_train, y_train)
    print("[INFO] Evaluating best SGD")
    bestModel = searchResults.best_estimator_
    print(bestModel)
    print("R2: {:.2f}".format(bestModel.score(X_test, y_test)))
    y_hat_sgd=bestModel.predict(X_test)
    #mape, mse, rmse, mae=model_evaluation(y_test,y_hat_sgd, experiment=None)
    
    #reg = make_pipeline(StandardScaler(),SGDRegressor(max_iter=1000, tol=1e-3)) #COMMENTED OUT ARE VANILLA MODELS
    #reg.fit(X_train, y_train)
    
    #y_hat_sgd=reg.predict(X_test)
    
    #pp.plot(y_test[0:300],label='Test')
    #pp.plot(y_hat_sgd[0:300], label='Predicted CL SGD')
    #pp.legend(loc = 'best')
    #pp.show()
    print('SGD CLUSTER PREDICTION EVALUATION')
    model_evaluation(y_test,y_hat_sgd, experiment=None)
    
    ###HUBER Regression
    
    huber_params = {
     'max_iter' : [100, 500, 1000] ,
     'alpha' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
     'epsilon' : [1.05, 1.15, 1.25, 1.35, 1.45, 1.55],
     'tol': loguniform(1e-6,1e-3)
    }

    model = HuberRegressor()
    print(model.get_params().keys())
    randomSearch = RandomizedSearchCV(estimator=model, n_jobs=-1,
    	cv=tscv, param_distributions=huber_params,
    	scoring="neg_mean_squared_error")
    searchResults = randomSearch.fit(X_train, y_train)
    print("[INFO] Evaluating best HUBER")
    bestModel = searchResults.best_estimator_
    print(bestModel)
    print("R2: {:.2f}".format(bestModel.score(X_test, y_test)))
    y_hat_hub=bestModel.predict(X_test)
    #mape, mse, rmse, mae=model_evaluation(y_test,y_hat_hub, experiment=None)
    
    #huber = HuberRegressor().fit(X_train, y_train)
    #y_hat_hub=huber.predict(X_test)
    
    #pp.plot(y_test[0:300],label='Test')
    #pp.plot(y_hat_hub[0:300], label='Predicted CL Huber')
    #pp.legend(loc = 'best')
    #pp.show()
    print('HUBER CLUSTER PREDICTION EVALUATION')
    model_evaluation(y_test,y_hat_hub, experiment=None)
    
    ###Lars Regression
    
    lars_params = {
     'n_nonzero_coefs' : [1, 10, 20, 30, 50, 100, 200, 300, 400, 500] ,
     'normalize' : [False],
    }


    model = Lars(n_nonzero_coefs=1, normalize=False)
    print(model.get_params().keys())
    randomSearch = RandomizedSearchCV(estimator=model, n_jobs=-1,
    	cv=tscv, param_distributions=lars_params,
    	scoring="neg_mean_squared_error")
    searchResults = randomSearch.fit(X_train, y_train)
    print("[INFO] Evaluating best LARS")
    bestModel = searchResults.best_estimator_
    print(bestModel)
    print("R2: {:.2f}".format(bestModel.score(X_test, y_test)))
    y_hat_lars=bestModel.predict(X_test)
    #mape, mse, rmse, mae=model_evaluation(y_test,y_hat_lars, experiment=None)
    
    #lar=Lars(n_nonzero_coefs=1, normalize=False).fit(X_train, y_train)
    #y_hat_lars=lar.predict(X_test)
    
    #pp.plot(y_test[0:300],label='Test')
    #pp.plot(y_hat_lars[0:300], label='Predicted CL Lars')
    #pp.legend(loc = 'best')
    #pp.show()
    print('LARS CLUSTER PREDICTION EVALUATION')
    model_evaluation(y_test,y_hat_lars, experiment=None)
    
    ###LASSO REGRESSION
    
    lasso_params = {
     'lasso__max_iter' : [1000, 2000, 3000, 4000] ,
     'lasso__alpha' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
     'lasso__tol': loguniform(1e-6,1e-3)
    }

    model = make_pipeline(StandardScaler(),Lasso(max_iter=2000 ,tol=1e-5, alpha=0.1))
    print(model.get_params().keys())
    randomSearch = RandomizedSearchCV(estimator=model, n_jobs=-1,
    	cv=tscv, param_distributions=lasso_params,
    	scoring="neg_mean_squared_error")
    searchResults = randomSearch.fit(X_train, y_train)
    print("[INFO] Evaluating best LASSO")
    bestModel = searchResults.best_estimator_
    print(bestModel)
    print("R2: {:.2f}".format(bestModel.score(X_test, y_test)))
    y_hat_clf=bestModel.predict(X_test)
    #mape, mse, rmse, mae=model_evaluation(y_test,y_hat_clf, experiment=None)
    
    #clf = Lasso(alpha=0.1).fit(X_train, y_train)
    #y_hat_clf = clf.predict(X_test)
    
    #pp.plot(y_test[0:300],label='Test')
    #pp.plot(y_hat_clf[0:300], label='Predicted CL Lasso')
    #pp.legend(loc = 'best')
    #pp.show()
    print('LASSO CLUSTER PREDICTION EVALUATION')
    model_evaluation(y_test,y_hat_clf, experiment=None)
    
    ###RANSAC REGRESSOR [NOT VALID CONSENSUS SET]
    #rnsc = RANSACRegressor(random_state=0).fit(X_train, y_train)
    #y_hat_rnsc = rnsc.predict(X_test)
    
    #pp.plot(y_test[0:300],label='Test')
    #pp.plot(y_hat_rnsc[0:300], label='Predicted CL RANSAC')
    #pp.legend(loc = 'best')
    #pp.show()
    #print('RANSAC CLUSTER PREDICTION EVALUATION')
    #model_evaluation(y_test,y_hat_rnsc)
    
    ###RIDGE REGRESSOR
    
    ridge_params = {
     'alpha' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 500, 1000, 1500]
    }

    model = Ridge(alpha=1.0)
    print(model.get_params().keys())
    randomSearch = RandomizedSearchCV(estimator=model, n_jobs=-1,
    	cv=tscv, param_distributions=ridge_params,
    	scoring="neg_mean_squared_error")
    searchResults = randomSearch.fit(X_train, y_train)
    print("[INFO] Evaluating best RIDGE")
    bestModel = searchResults.best_estimator_
    print(bestModel)
    print("R2: {:.2f}".format(bestModel.score(X_test, y_test)))
    y_hat_rdg=bestModel.predict(X_test)
    #mape, mse, rmse, mae=model_evaluation(y_test,y_hat_rdg, experiment=None)
    
    #rdg = Ridge(alpha=1.0).fit(X_train, y_train)
    #y_hat_rdg = rdg.predict(X_test)
    
    #pp.plot(y_test[0:300],label='Test')
    #pp.plot(y_hat_rdg[0:300], label='Predicted CL RIDGE')
    #pp.legend(loc = 'best')
    #pp.show()
    print('RIDGE CLUSTER PREDICTION EVALUATION')
    model_evaluation(y_test,y_hat_rdg, experiment=None)
    
    ###TheilSen REGRESSION
    ts = TheilSenRegressor(random_state=0).fit(X_train, y_train)
    y_hat_ts = ts.predict(X_test)
    
    #pp.plot(y_test[0:300],label='Test')
    #pp.plot(y_hat_ts[0:300], label='Predicted CL TS')
    #pp.legend(loc = 'best')
    #pp.show()
    print('TS CLUSTER PREDICTION EVALUATION')
    model_evaluation(y_test,y_hat_ts, experiment=None)
    
    ###GAUSSIAN PROCESS REGRESSION [UNABLE TO ALLOCATE RAM]
    
    ###BAYESIAN RIDGE
    
    bayesianridge_params = {
     'alpha_1' : loguniform(1e-8,1e-3),
     'alpha_2' : loguniform(1e-8,1e-3),
     'lambda_1' : loguniform(1e-8,1e-3),
     'lambda_2' : loguniform(1e-8,1e-3)
    }

    model = BayesianRidge()
    print(model.get_params().keys())
    randomSearch = RandomizedSearchCV(estimator=model, n_jobs=-1,
    	cv=tscv, param_distributions=bayesianridge_params,
    	scoring="neg_mean_squared_error")
    searchResults = randomSearch.fit(X_train, y_train)
    print("[INFO] Evaluating best LASSO")
    bestModel = searchResults.best_estimator_
    print(bestModel)
    print("R2: {:.2f}".format(bestModel.score(X_test, y_test)))
    y_hat_bar=bestModel.predict(X_test)
    #mape, mse, rmse, mae=model_evaluation(y_test,y_hat_bar, experiment=None)
    
    #bar = BayesianRidge().fit(X_train, y_train)
    #y_hat_bar = bar.predict(X_test)
    
    #pp.plot(y_test[0:300],label='Test')
    #pp.plot(y_hat_bar[0:300], label='Predicted CL Bayesian')
    #pp.legend(loc = 'best')
    #pp.show()
    print('Bayesian CLUSTER PREDICTION EVALUATION')
    model_evaluation(y_test,y_hat_bar, experiment=None)
    
    ###KNEIGHBORS REGRESSOR
    
    knr_params = {
        'leaf_size' : list(range(1,50)) ,
        'n_neighbors' : list(range(1,30)) ,
        'p': [1,2]
    }

    model = KNeighborsRegressor(n_neighbors=4)
    print(model.get_params().keys())
    randomSearch = RandomizedSearchCV(estimator=model, n_jobs=-1,
    	cv=tscv, param_distributions=knr_params,
    	scoring="neg_mean_squared_error")
    searchResults = randomSearch.fit(X_train, y_train)
    print("[INFO] Evaluating best LASSO")
    bestModel = searchResults.best_estimator_
    print(bestModel)
    print("R2: {:.2f}".format(bestModel.score(X_test, y_test)))
    y_hat_neigh=bestModel.predict(X_test)
    #mape, mse, rmse, mae=model_evaluation(y_test,y_hat_neigh, experiment=None)
    
    #neigh = KNeighborsRegressor(n_neighbors=4).fit(X_train, y_train)
    #y_hat_neigh = neigh.predict(X_test)
    
    #pp.plot(y_test[0:300],label='Test')
    #pp.plot(y_hat_neigh[0:300], label='Predicted CL KNR')
    #pp.legend(loc = 'best')
    #pp.show()
    print('KNR CLUSTER PREDICTION EVALUATION')
    model_evaluation(y_test,y_hat_neigh, experiment=None)
    """
        
    return y_test,y_hat_xgb, y_hat_lr, y_hat_svr, y_hat_sgd, y_hat_hub, y_hat_lars, y_hat_clf, y_hat_rdg, y_hat_ts, y_hat_bar, y_hat_neigh, y_fusion 

def prep_in_out(df_dataset):
    #remove null records
    #df_dataset = df_dataset.dropna()
    #df_dataset =df_dataset.drop(columns=['Datetime'])
    #DEFINE X and y
    X=df_dataset.drop(columns=['total']).values
    y=df_dataset['total'].values
    print(X)
    print(X.shape)
    print(y)
    print(y.shape)
    
    #split dataset to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test

def get_client_groups(cl_labels,c_num):
    
    client_group_dict={}
    for j in range(c_num):
        for i in range(cl_labels.shape[0]):
            if cl_labels[i]==j:
                client_group_dict.setdefault(j,[]).append(str(i+1))
    
    return client_group_dict

def create_cluster_dataset(df_dataset_train, df_dataset_test,cluster_members):
    
    #cluster_data = df_dataset[cluster_members].copy()
    cluster_data_train = df_dataset_train[cluster_members].copy()
    cluster_data_test = df_dataset_test[cluster_members].copy()
    
    print(cluster_members)
    
    return cluster_data_train, cluster_data_test

def cldata_tosupervised(cluster_dataset_train, cluster_dataset_test, flag):
    """
    ###BUILD A PCA TOGGLE ON THE CLIENT DATA
    if flag == 1:
        #print(X_train.T.shape)
        pca_cluster_train = PCA(n_components=cluster_dataset.shape[1]) #n should be as many as the features of the dataset to see all components before keeping based on variance ratio
        principalComponents_cl = pca_cluster_train.fit_transform(cluster_dataset)
        print(principalComponents_cl) # Component values
        print(principalComponents_cl.shape)
        print(pca_cluster_train.explained_variance_ratio_)
    ###
    """
    
    #cluster_dataset['total'] = cluster_dataset.sum(axis=1)
    
    cluster_dataset_train['total'] = cluster_dataset_train.sum(axis=1)
    cluster_dataset_test['total'] = cluster_dataset_test.sum(axis=1)
    
    
    #supervised_cldata=cluster_dataset['total'].copy()
    #supervised_cldata=supervised_cldata.to_frame()
    
    supervised_cldata_train=cluster_dataset_train['total'].copy()
    supervised_cldata_train=supervised_cldata_train.to_frame()
    
    supervised_cldata_test=cluster_dataset_test['total'].copy()
    supervised_cldata_test=supervised_cldata_test.to_frame()
    
    
    #print(supervised_cldata[])
    
    #supervised_cldata = lagfeatures(supervised_cldata, featurename='total', nlags=4)
    #supervised_cldata = supervised_cldata.dropna()
    
    supervised_cldata_train = lagfeatures(supervised_cldata_train, featurename='total', nlags=4)
    supervised_cldata_train = supervised_cldata_train.dropna()

    supervised_cldata_test = lagfeatures(supervised_cldata_test, featurename='total', nlags=4)
    supervised_cldata_test = supervised_cldata_test.dropna()
    
    #X_train, X_test, y_train, y_test=prep_in_out(supervised_cldata)
    
    ###FINAL TRAIN AND TEST X,y
    
    X_train=supervised_cldata_train.drop(columns=['total']).values
    y_train=supervised_cldata_train['total'].values
    
    X_test=supervised_cldata_test.drop(columns=['total']).values
    y_test=supervised_cldata_test['total'].values
    
    
    print(supervised_cldata_train, supervised_cldata_test)
    print(X_train, X_test, y_train, y_test)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    y_test, y_hat_xgb, y_hat_lr, y_hat_svr, y_hat_sgd, y_hat_hub, y_hat_lars, y_hat_clf, y_hat_rdg, y_hat_ts, y_hat_bar, y_hat_neigh, y_fusion = estimators(X_train, X_test, y_train, y_test)
    
    return y_test, y_hat_xgb, y_hat_lr, y_hat_svr, y_hat_sgd, y_hat_hub, y_hat_lars, y_hat_clf, y_hat_rdg, y_hat_ts, y_hat_bar, y_hat_neigh,y_fusion
    

#MAIN BODY
#DATASET SHOWS IN KWH
dataset_path = r'D:\Datasets\prtclientconsumption.csv'

df_dataset = pd.read_csv(dataset_path, low_memory=False)

df_dataset['Datetime'] = pd.to_datetime(df_dataset['Datetime'])

#OMIT 2011-2012 DUE TO ZERO VALUES (MAYBE AS AN ALTERNATIVE)
#df_dataset = df_dataset[(df_dataset['Datetime'].dt.year > 2012)]


df_dataset = df_dataset.set_index('Datetime')

#print(df_dataset.tail())
#print(df_dataset.iloc[0].values)
print(df_dataset.shape)


#df_dataset['year'] = df_dataset['Datetime'].dt.year

#CLUSTER BASED ON TRAIN SET [TRAIN - TEST SPLIT OF 80-20]
df1_train=df_dataset.iloc[:int(0.8*df_dataset.shape[0]),:] #else just do df_dataset
df1_test=df_dataset.iloc[int(0.8*df_dataset.shape[0]+1):,:] #else just do df_dataset
#df_dataset_cl=df1_train.resample('1M').sum() #MONTHLY KWh profile FOR CLUSTERING [TEST FOR 2M, 3M=1Q, 2Q, 1Y]
df_dataset_cl=df1_train.resample('1M').sum()
formatted_dataset = to_time_series_dataset(df_dataset_cl.T) #tslearn 3D format for multiple time series clustering
print(df_dataset_cl.tail())
print(formatted_dataset)
print(formatted_dataset.shape)

print(df_dataset_cl.head(10))
print(df_dataset_cl.shape)

#RUN KMeans with DTW to group time series into clusters based on that similarity [CLUSTERING]
c_num=20 # Gives results with 20 clusters
#create new Dataframe to store values from experiments. Save to CSV at the end, after for loop. Each loop writes to Dataframe
df_clustering_metrics_xgb = pd.DataFrame(columns=['MAPE', 'MSE', 'RMSE', 'MAE'])
df_clustering_metrics_lr = pd.DataFrame(columns=['MAPE', 'MSE', 'RMSE', 'MAE'])
df_clustering_metrics_svr = pd.DataFrame(columns=['MAPE', 'MSE', 'RMSE', 'MAE'])
df_clustering_metrics_sgd = pd.DataFrame(columns=['MAPE', 'MSE', 'RMSE', 'MAE'])
df_clustering_metrics_hub = pd.DataFrame(columns=['MAPE', 'MSE', 'RMSE', 'MAE'])
df_clustering_metrics_lars = pd.DataFrame(columns=['MAPE', 'MSE', 'RMSE', 'MAE'])
df_clustering_metrics_clf = pd.DataFrame(columns=['MAPE', 'MSE', 'RMSE', 'MAE'])
df_clustering_metrics_rdg = pd.DataFrame(columns=['MAPE', 'MSE', 'RMSE', 'MAE'])
df_clustering_metrics_ts = pd.DataFrame(columns=['MAPE', 'MSE', 'RMSE', 'MAE'])
df_clustering_metrics_bar = pd.DataFrame(columns=['MAPE', 'MSE', 'RMSE', 'MAE'])
df_clustering_metrics_neigh = pd.DataFrame(columns=['MAPE', 'MSE', 'RMSE', 'MAE'])
df_clustering_metrics_fusion = pd.DataFrame(columns=['MAPE', 'MSE', 'RMSE', 'MAE'])

#FOR 2 TO 10 REQUESTED CLUSTERS FORMING CNUM AND CLIENT GROUPS
for i in range(9):
    
    #start = time.time()
    c_num=i+2 #Number of clusters
    model = TimeSeriesKMeans(n_clusters=c_num, metric="dtw", max_iter=10) #10 iters
    cl_labels=model.fit_predict(formatted_dataset)
    #end = time.time()
    #total_time = end - start
    #print("\n"+ str(total_time))
    
    print(cl_labels)

    #PLOTTING
    #pp.plot(df_dataset[str(12)].values, label='client 1')
    #pp.show()

    client_groups = get_client_groups(cl_labels,c_num)
    print(client_groups)

    #[PREDICTION PER CLUSTER AND ADD THE RESULTS: needs df_dataset, client_groups]
    cluster_dataset_train, cluster_dataset_test=create_cluster_dataset(df1_train, df1_test,client_groups[0])
    print(cluster_dataset_train, cluster_dataset_train)
    #FIRST CLUSTER
    y_test, y_hat_xgb, y_hat_lr, y_hat_svr, y_hat_sgd, y_hat_hub, y_hat_lars, y_hat_clf, y_hat_rdg, y_hat_ts, y_hat_bar, y_hat_neigh, y_fusion=cldata_tosupervised(cluster_dataset_train, cluster_dataset_test, flag=1)

    #REMAINING CLUSTERS IN THAT CNUM. EACH CLUSTER TAKES THE CORRESPONDING GROUP TO COMPUTE AND ADDS IT TO THE AGGREGATE PREDICTION
    for i in range(c_num):
        j=i+1
        if j<c_num:
            cluster_dataset_train, cluster_dataset_test=create_cluster_dataset(df1_train, df1_test,client_groups[j])
            print(cluster_dataset_train, cluster_dataset_test)
            y_test_cl, y_hat_cl_xgb, y_hat_cl_lr, y_hat_cl_svr, y_hat_cl_sgd, y_hat_cl_hub, y_hat_cl_lars, y_hat_cl_clf, y_hat_cl_rdg, y_hat_cl_ts, y_hat_cl_bar, y_hat_cl_neigh, y_cl_fusion=cldata_tosupervised(cluster_dataset_train, cluster_dataset_test, flag=1)
            y_test = np.add(y_test,y_test_cl)
            #FINAL PREDICTION CONSTRUCTION FROM ALL CLUSTERS [FOR 11 ESTIMATORS]
            y_hat_xgb = np.add(y_hat_xgb,y_hat_cl_xgb)
            y_hat_lr = np.add(y_hat_lr,y_hat_cl_lr)
            y_hat_svr = np.add(y_hat_svr,y_hat_cl_svr)
            y_hat_sgd = np.add(y_hat_sgd,y_hat_cl_sgd)
            y_hat_hub = np.add(y_hat_hub,y_hat_cl_hub)
            y_hat_lars = np.add(y_hat_lars,y_hat_cl_lars)
            y_hat_clf = np.add(y_hat_clf,y_hat_cl_clf)
            #y_hat_rnsc = np.add(y_hat_rnsc,y_hat_cl_rnsc)
            y_hat_rdg = np.add(y_hat_rdg,y_hat_cl_rdg)
            y_hat_ts = np.add(y_hat_ts,y_hat_cl_ts)
            y_hat_bar = np.add(y_hat_bar,y_hat_cl_bar)
            y_hat_neigh = np.add(y_hat_neigh,y_hat_cl_neigh)
            y_fusion = np.add(y_fusion,y_cl_fusion)
            #break #Breakpoint for partial summation of results
            
            #FOR CPU USAGE
            print("Sleep before next cluster")
            time.sleep(20)

    



    ###PLOT FINAL RESULT AND EVALUATE [WORKS AS BASELINE K-MEANS DTW]
    #fig, ax=pp.subplots()
    #ax.get_yaxis().set_major_formatter(pp.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    #ax.get_xaxis().set_major_formatter(pp.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))  
    #pp.plot(y_test,label='Test')
    #pp.plot(y_hat_xgb, label='Predicted Total XGB')
    #pp.plot(y_hat_lr, label='Predicted Total LR')
    #pp.xlabel('Timesteps')
    #pp.ylabel('Total Demand (kWh)')
    #pp.legend(loc = 'best')
    #pp.savefig(r'D:\Datasets\dimreduce\charts\all_clusters_test_predict_v2.jpg',dpi=300,bbox_inches="tight")
    #pp.show()
    
    
    #break #Breakpoint for 1 specific number of clusters specified by the initial value of c_num
    
    print("CLUSTERS: ", c_num)
    print("TOTAL DEMAND FORECAST PERFORMANCE FUSION")
    
    #IN THE CASE OF LAGGED FEATURE ESTS
    #output_df_test = pd.DataFrame(y_test)
    #output_df_test = output_df_test.iloc[4:,:]
    #mape, mse, rmse, mae=model_evaluation(output_df_test.to_numpy(),y_fusion, experiment=None)
    
    mape, mse, rmse, mae=model_evaluation(y_test,y_fusion, experiment=None) #FOR ALL CASES THE Y_TEST EVALUATION for fusion model 3 lines
    new_row_fusion = {'MAPE': mape, 'MSE': mse, 'RMSE': rmse, 'MAE': mae}
    df_clustering_metrics_fusion = df_clustering_metrics_fusion.append(new_row_fusion, ignore_index=True)

df_clustering_metrics_fusion.to_csv(r'D:\Datasets\dimreduce\metadata1M\FUSIONVOTEALLWEIGHTED_CL2_10_1M.csv',index=False) #saving the total
    
"""
    #EXPORTING RESULTS
    print("CLUSTERS: ", c_num)
    print("TOTAL DEMAND FORECAST PERFORMANCE XGB")
    mape, mse, rmse, mae=model_evaluation(y_test,y_hat_xgb, experiment=None)
    new_row_xgb = {'MAPE': mape, 'MSE': mse, 'RMSE': rmse, 'MAE': mae}
    df_clustering_metrics_xgb = df_clustering_metrics_xgb.append(new_row_xgb, ignore_index=True)
    
    print("TOTAL DEMAND FORECAST PERFORMANCE LR")
    mape, mse, rmse, mae=model_evaluation(y_test,y_hat_lr, experiment=None)
    new_row_lr = {'MAPE': mape, 'MSE': mse, 'RMSE': rmse, 'MAE': mae}
    df_clustering_metrics_lr = df_clustering_metrics_lr.append(new_row_lr, ignore_index=True)
    
    print("TOTAL DEMAND FORECAST PERFORMANCE SVR")
    mape, mse, rmse, mae=model_evaluation(y_test,y_hat_svr, experiment=None)
    new_row_svr = {'MAPE': mape, 'MSE': mse, 'RMSE': rmse, 'MAE': mae}
    df_clustering_metrics_svr = df_clustering_metrics_svr.append(new_row_svr, ignore_index=True)
    
    print("TOTAL DEMAND FORECAST PERFORMANCE SGD")
    mape, mse, rmse, mae=model_evaluation(y_test,y_hat_sgd, experiment=None)
    new_row_sgd = {'MAPE': mape, 'MSE': mse, 'RMSE': rmse, 'MAE': mae}
    df_clustering_metrics_sgd = df_clustering_metrics_sgd.append(new_row_sgd, ignore_index=True)
    
    print("TOTAL DEMAND FORECAST PERFORMANCE HUB")
    mape, mse, rmse, mae=model_evaluation(y_test,y_hat_hub, experiment=None)
    new_row_hub = {'MAPE': mape, 'MSE': mse, 'RMSE': rmse, 'MAE': mae}
    df_clustering_metrics_hub = df_clustering_metrics_hub.append(new_row_hub, ignore_index=True)
    
    print("TOTAL DEMAND FORECAST PERFORMANCE LARS")
    mape, mse, rmse, mae=model_evaluation(y_test,y_hat_lars, experiment=None)
    new_row_lars = {'MAPE': mape, 'MSE': mse, 'RMSE': rmse, 'MAE': mae}
    df_clustering_metrics_lars = df_clustering_metrics_lars.append(new_row_lars, ignore_index=True)
    
    print("TOTAL DEMAND FORECAST PERFORMANCE LASSO")
    mape, mse, rmse, mae=model_evaluation(y_test,y_hat_clf, experiment=None)
    new_row_clf = {'MAPE': mape, 'MSE': mse, 'RMSE': rmse, 'MAE': mae}
    df_clustering_metrics_clf = df_clustering_metrics_clf.append(new_row_clf, ignore_index=True)
    
    #print("TOTAL DEMAND FORECAST PERFORMANCE RANSAC")
    #model_evaluation(y_test,y_hat_rnsc, experiment=None)
    print("TOTAL DEMAND FORECAST PERFORMANCE RIDGE")
    mape, mse, rmse, mae=model_evaluation(y_test,y_hat_rdg, experiment=None)
    new_row_rdg = {'MAPE': mape, 'MSE': mse, 'RMSE': rmse, 'MAE': mae}
    df_clustering_metrics_rdg = df_clustering_metrics_rdg.append(new_row_rdg, ignore_index=True)
    
    print("TOTAL DEMAND FORECAST PERFORMANCE THEILSEN")
    mape, mse, rmse, mae=model_evaluation(y_test,y_hat_ts, experiment=None)
    new_row_ts = {'MAPE': mape, 'MSE': mse, 'RMSE': rmse, 'MAE': mae}
    df_clustering_metrics_ts = df_clustering_metrics_ts.append(new_row_ts, ignore_index=True)
    
    print("TOTAL DEMAND FORECAST PERFORMANCE BAYESIAN RIDGE")
    mape, mse, rmse, mae=model_evaluation(y_test,y_hat_bar, experiment=None)
    new_row_bar = {'MAPE': mape, 'MSE': mse, 'RMSE': rmse, 'MAE': mae}
    df_clustering_metrics_bar = df_clustering_metrics_bar.append(new_row_bar, ignore_index=True)
    
    print("TOTAL DEMAND FORECAST PERFORMANCE KN REGRESSOR")
    mape, mse, rmse, mae=model_evaluation(y_test,y_hat_neigh, experiment=None)
    new_row_neigh = {'MAPE': mape, 'MSE': mse, 'RMSE': rmse, 'MAE': mae}
    df_clustering_metrics_neigh = df_clustering_metrics_neigh.append(new_row_neigh, ignore_index=True)
    
    
    

df_clustering_metrics_xgb.to_csv(r'D:\Datasets\dimreduce\metadataCV\XGB_CL2_10_1M_TUNECV.csv',index=False)
df_clustering_metrics_lr.to_csv(r'D:\Datasets\dimreduce\metadataCV\lr_CL2_10_1M_TUNECV.csv',index=False)
df_clustering_metrics_svr.to_csv(r'D:\Datasets\dimreduce\metadataCV\svr_CL2_10_1M_TUNECV.csv',index=False)
df_clustering_metrics_sgd.to_csv(r'D:\Datasets\dimreduce\metadataCV\sgd_CL2_10_1M_TUNECV.csv',index=False)
df_clustering_metrics_hub.to_csv(r'D:\Datasets\dimreduce\metadataCV\hub_CL2_10_1M_TUNECV.csv',index=False)
df_clustering_metrics_lars.to_csv(r'D:\Datasets\dimreduce\metadataCV\lars_CL2_10_1M_TUNECV.csv',index=False)
df_clustering_metrics_clf.to_csv(r'D:\Datasets\dimreduce\metadataCV\clf_CL2_10_1M_TUNECV.csv',index=False)
df_clustering_metrics_rdg.to_csv(r'D:\Datasets\dimreduce\metadataCV\rdg_CL2_10_1M_TUNECV.csv',index=False)
df_clustering_metrics_ts.to_csv(r'D:\Datasets\dimreduce\metadataCV\ts_CL2_10_1M_TUNECV.csv',index=False)
df_clustering_metrics_bar.to_csv(r'D:\Datasets\dimreduce\metadataCV\bar_CL2_10_1M_TUNECV.csv',index=False)
df_clustering_metrics_neigh.to_csv(r'D:\Datasets\dimreduce\metadataCV\neigh_CL2_10_1M_TUNECV.csv',index=False)
#END OF EXPORTING RESULTS
"""


#df_dataset = df_dataset.drop(columns=['Datetime'])

#Try to downsample to a smaller series
##CODE BELOW NOT CURRENTLY USED

"""
pp.plot(df_dataset[str(1)].values[60000:80000], label='client 1')
#pp.plot(df_dataset[str(2)].values, label= 'client 2')
#pp.plot(df_dataset[str(3)].values, label= 'client 2')
#pp.plot(df_dataset[str(4)].values, label= 'client 2')
pp.plot(df_dataset[str(5)].values[60000:80000], label= 'client 2')
pp.show()

#score=calc_soft_dtw(df_dataset[str(1)].values[60000:80000],df_dataset[str(1)].values[60000:80000])
#print(score)

"""

"""
x = df_dataset.values
x = StandardScaler().fit_transform(x) # standardizing the features
print(x.shape)
print(np.mean(x),np.std(x))

pca_clients = PCA(n_components=370) #n should be as many as the features of the dataset to see all components before keeping based on variance ratio
principalComponents_clients = pca_clients.fit_transform(x)
print(principalComponents_clients) # Component values
print(principalComponents_clients.shape)

print(pca_clients.explained_variance_ratio_)

principal_clients_Df = pd.DataFrame(data = principalComponents_clients[:,0:2]
             , columns = ['principal component 1', 'principal component 2'])

pp.plot(principal_clients_Df['principal component 1'].values)
pp.show()
"""


