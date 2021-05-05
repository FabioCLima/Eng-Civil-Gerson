import pandas as pd
import numpy as np
import os
import sklearn

import seaborn as sns   
import matplotlib.pyplot as plt 
import matplotlib as mpl  
import warnings
import streamlit as st
import plotly.figure_factory as ff
import plotly.graph_objects as go
warnings.filterwarnings('ignore')

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
pd.options.display.max_columns = 200

from IPython.core.pylabtools import figsize
figsize(12, 8)

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error,  explained_variance_score, max_error, median_absolute_error
from sklearn.metrics import make_scorer
from sklearn.impute import SimpleImputer

BASE_DIR = os.path.join( os.path.abspath('..') )
MODELS_DIR = os.path.join( BASE_DIR, 'models' )
IMGS_DIR = os.path.join( BASE_DIR, 'imgs' )
DATA_DIR = os.path.join(BASE_DIR, 'data')

@st.cache(allow_output_mutation = True)
def train_model(df):     
    
    numeric_features = df.drop('s (mm)', axis = 1).select_dtypes(include=['int64', 'float64']).columns
    
    X, y = df[numeric_features], df['s (mm)']
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 123)
    
    numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])
    
    preprocessor = ColumnTransformer(
        transformers=[('num', numeric_transformer, numeric_features)])
    
    gbr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('gbr', GradientBoostingRegressor(
        learning_rate = 0.2,
        max_depth = 2,
        subsample =0.8,
        n_estimators = 100,
        random_state = 123))])
    
    gbr.fit(X_train, y_train)
    modelo = gbr
    y_pred = gbr.predict(X_test)
    training_score = gbr.score(X_train, y_train)
    testing_score = r2_score(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(MSE)
    
    y_pred = modelo.predict(X_test)
    residuals = y_test - y_pred
    df_pred_actual = pd.DataFrame({
   
    's(mm)_estimado'             : np.round(y_pred, 3),    # recalque_estimado
    's(mm)_real'                 : np.round(y_test, 3),    # recalque_real
    'erro_residual(mm)'          : np.round(residuals, 3)  # diferença
    })
    df_results = df_pred_actual.reset_index(drop=True)
    
    #! Salvando o modelo 
    #
    k = X.shape[1]
    n = len(y)
    #  é a única métrica aqui que considera o problema de overfitting 
    r2_adjusted = 1 - ( (1 - testing_score) * (n - 1)/ (n - k - 1) )
    # mean absolute percentual error
    mape = np.mean( np.abs( (y_test - y_pred) / y_test ) ) * 100

    model_data = pd.Series({
    'features'      : X.columns.tolist(),
    'model'         : modelo,
    'score'         : testing_score,
    'r2_adjusted'   : r2_adjusted,
    'RMSE'          : rmse,
    'MAPE'          : mape,
    })

    model_data.to_pickle( os.path.join( MODELS_DIR, 'gbr_model.pkl' ) )
    return  modelo, training_score, testing_score, rmse, df_results, y_test, y_pred


def plot_resultados_modelo():     
    pass
