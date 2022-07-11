# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:24:20 2022

@author: dimkonto
"""

import pandas as pd
from sys import argv
import xlsxwriter
import random
from pandas import ExcelWriter
from pandas import ExcelFile

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
from mlxtend.evaluate import bias_variance_decomp

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.linear_model import Ridge

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

"""
#PERFORMANCE OF EACH INDIVIDUAL ESTIMATOR OVER DIFFERENT SUBSAMPLING PERIODS
estimators = ['XGB', 'lr', 'svr', 'sgd', 'hub', 'lars', 'clf','rdg', 'ts', 'bar', 'neigh']
subsampling = ['1M', '2M', '1Q', '2Q','1Y']
for j in range(11):
    fig, ax=pp.subplots()
    #ax.get_yaxis().set_major_formatter(pp.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    #ax.get_xaxis().set_major_formatter(pp.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))  
    for i in range(5):
        dataset_path = r'D:\Datasets\dimreduce\metadata'+subsampling[i]+'\\'+estimators[j]+'_CL2_10_'+subsampling[i]+'.csv'
        df_dataset = pd.read_csv(dataset_path, low_memory=False)
        pp.plot(df_dataset['MAPE'],label=subsampling[i])
        pp.xticks(range(len(df_dataset['MAPE'])-1), range(2,10,1))
    pp.xlabel('Clusters')
    pp.ylabel(estimators[j]+' MAPE')
    pp.legend(loc = 'best')
    pp.savefig(r'D:\Datasets\dimreduce\charts\clustering_performance_'+estimators[j]+'.jpg',dpi=300,bbox_inches="tight")
    pp.show()
"""
    
#SINGLE METRIC BASED GRAPHS FOR ALL ESTIMATORS IN 1M SUBSAMPLING
estimators = ['XGB', 'lr', 'svr', 'sgd', 'hub', 'lars', 'clf','rdg', 'ts', 'bar', 'neigh', 'FUSION', 'FUSIONPEAK','FUSIONNONPEAK','FUSIONVOTEBASE','FUSIONVOTEPEAK','FUSIONVOTEALL','FUSIONVOTEBASEWEIGHTED','FUSIONVOTEPEAKWEIGHTED','FUSIONVOTEALLWEIGHTED']
estimator_names = ['XGBoost', 'LR', 'Linear SVR', 'SGD', 'Huber', 'LARS', 'Lasso','Ridge', 'Theil-Sen', 'Bayesian Ridge', 'KNN','SRA','SRP','SRNP','VRUNP','VRUP','VRUA','VROWNP','VROWP','VROWA']
colors = ['#006400','#ED0DD9','#FAC205','#F97306','#8C000F','#C875C4','#069AF3','#C1F80A','#EF4026','#929591','#40E0D0','#0000FF']
metrics = ['MAPE', 'MSE', 'RMSE', 'MAE']

"""
for error_metric in metrics:
    for j in range(len(estimators)):
        dataset_path = r'D:\Datasets\dimreduce\metadata'+'1M'+'\\'+estimators[j]+'_CL2_10_'+'1M'+'.csv'
        df_dataset = pd.read_csv(dataset_path, low_memory=False)
        pp.plot(df_dataset[error_metric], label=estimator_names[j], color=colors[j])
        pp.xticks(range(len(df_dataset[error_metric])-1), range(2,10,1))
    pp.xlabel('Clusters')
    pp.ylabel(error_metric)
    pp.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    pp.savefig(r'D:\Datasets\dimreduce\charts\error_scores_'+error_metric+'.jpg',dpi=300,bbox_inches="tight")
    pp.show()
"""
    
#PLOT BAR CHART OF ALL BASE ESTIMATORS FOR CL2 (1 BAR CHART FOR EACH METRIC):
    
for error_metric in metrics:
    print("PRINTING ",error_metric,"FOR ALL ESTIMATORS")
    silhouette_cluster_choice = [] #2
    elbow_cluster_choice = [] #7
    for j in range(len(estimators)):
        #dataset_path = r'D:\Datasets\dimreduce\metadata'+'1M'+'\\'+estimators[j]+'_CL2_10_'+'1M'+'.csv'
        
        #FOR TUNED MODELS
        dataset_path = r'D:\Datasets\dimreduce\metadataCV'+'\\'+estimators[j]+'_CL2_10_'+'1M_TUNECV'+'.csv'
        
        df_dataset = pd.read_csv(dataset_path, low_memory=False)
        print(df_dataset[error_metric][0])
        silhouette_cluster_choice.append(df_dataset[error_metric][0]) #index denoting the optimal silhouette cluster
        elbow_cluster_choice.append(df_dataset[error_metric][6]) #index denoting the optimal elbow cluster
        
    
    #PLOTS FOR OPTIMAL SILHOUETTE CLUSTER
    fig, ax = pp.subplots()
    
    bars = ax.barh(estimator_names, silhouette_cluster_choice, color = 'orange')

    ax.bar_label(bars, label_type = 'center')
    #pp.title(error_metric + " of Base Learners  - Optimal Silhoutte Clusters")
    #pp.savefig(r'D:\Datasets\dimreduce\charts\bar_plots_silhouette_base_'+error_metric+'.jpg',dpi=300,bbox_inches="tight")
    
    #FOR TUNED MODELS
    pp.title(error_metric + " of Estimators  - Optimal Silhoutte Clusters")
    pp.savefig(r'D:\Datasets\dimreduce\charts\bar_plots_silhouette_est_'+error_metric+'.jpg',dpi=300,bbox_inches="tight")
    
    pp.show()
    
    #PLOTS FOR OPTIMAL ELBOW CLUSTER
    fig, ax = pp.subplots()
    
    bars = ax.barh(estimator_names, elbow_cluster_choice, color = 'orange')

    ax.bar_label(bars, label_type = 'center')
    #pp.title(error_metric + " of Base Learners  - Optimal Elbow Clusters")
    #pp.savefig(r'D:\Datasets\dimreduce\charts\bar_plots_elbow_base_'+error_metric+'.jpg',dpi=300,bbox_inches="tight")
    
    #FOR TUNED MODELS
    pp.title(error_metric + " of Estimators  - Optimal Elbow Clusters")
    pp.savefig(r'D:\Datasets\dimreduce\charts\bar_plots_elbow_est_'+error_metric+'.jpg',dpi=300,bbox_inches="tight")
    
    pp.show()
        
        