# modeling_recalque.py 
#%%
import pandas as pd
import numpy as np
import os
import sklearn

import seaborn as sns   
import matplotlib.pyplot as plt 
import matplotlib as mpl  
import warnings
warnings.filterwarnings('ignore')

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
pd.options.display.max_columns = 200
pd.options.display.float_format = '{:.2f}'.format
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

#! Constantes de endereço necessárias para melhor organização de diretórios e pastas de trabalho.
#
SRC_DIR = os.path.join( os.path.abspath('..'), 'src')
BASE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
IMGS_DIR = os.path.join(BASE_DIR, 'imgs')
MODELS_DIR = os.path.join( BASE_DIR, 'models' )
input_file = 'recalque_to_modeling.csv'
input_path = os.path.join(DATA_DIR, input_file)

#! Lendo o arquivo sem "outliers"
data = pd.read_csv(input_path, index_col=0)
data.head()
#%%
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMGS_DIR, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
#! Modelagem

def split_type_features(df, target):
    """
        Esta função recebe como argumentos df (dataframe) e target      (variável dependente) e retorna uma tupla com a lista de features     numéricas e categóricas.
    """

    numeric_features = df.drop(target, axis = 1).select_dtypes(include=['int64', 'float64']).columns

    return numeric_features

numeric_features = split_type_features(data, target= 's (mm)')

#! Separando o conjunto de dados em train/test
#! 


X, y = data[numeric_features], data['s (mm)']
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size = 0.2, 
    random_state = 123)

#!Pipeline de processamento

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

#!Preprocessamento

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)])

gbr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('gbr', GradientBoostingRegressor(
        max_depth = 2,
        subsample =0.8,
        n_estimators = 100,
        random_state = 123))])
gbr.fit(X_train, y_train)

y_pred = gbr.predict(X_test)
training_score = gbr.score(X_train, y_train)
testing_score = r2_score(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(MSE)
print(f"Train R2 = {training_score:.3f}, Test R2 = {testing_score:.3f}")
print(f"MSE: {MSE:.2f}, RMSE = {rmse:.2f}")

def RMSE(y_test, y_pred):
    """
    Calculates Root Mean Squared Error between the actual and the predicted labels.
    
    """
    RMSE = np.sqrt(mean_squared_error(y_test,y_pred))
    return RMSE


# Função para calcular as métricas do modelo
#
def show_scores(model, name):

    k = X.shape[1]
    n = len(y)
    y_pred = model.predict(X_test)
    

    # explained variance score
    evars = explained_variance_score(y_test, y_pred)
    # maximum residual error
    max_erro = max_error(y_test, y_pred)
    
    mse = mean_squared_error(y_test, y_pred)
   
    r2_training = model.score(X_train, y_train)
    r2_true = r2_score(y_test, y_pred)
    # r2 score ajustado para o número de features
    #  é a única métrica aqui que considera o problema de overfitting 
    r2_adjusted = 1 - ( (1 - r2_true) * (n - 1)/ (n - k - 1) )
    rmse = RMSE(y_test, y_pred)
    # mean absolute percentual error
    mape = np.mean( np.abs( (y_test - y_pred) / y_test ) ) * 100
     
    scores = {
        "model name" : name,
        "R2_score(training data)" : np.round(r2_training, 3),
        "R2_score(test data)" : np.round(r2_true, 3),
        "R2_adjusted" : np.round(r2_adjusted, 3),
        "Explained Variance Score" : np.round(evars, 3),
        "Maximum Residual Error" : np.round(max_erro, 3),
        "Mean Square Error" : np.round(mse, 3),
        "Root Mean Square Error" : np.round(rmse, 3),
        "Mean Absolute Percentual Error(%)" : np.round(mape, 3)
    }
    results = pd.Series(scores)
    metricas = results.to_frame()
    return metricas

metricas_baseline = show_scores(gbr, 'GradientBoostingRegressor')
metricas_baseline

# %%
# Create scorer
mse_scorer = make_scorer(mean_squared_error)

# Implement LOOCV
scores = cross_val_score(gbr, X=X, y=y, cv=LeaveOneOut(), scoring=mse_scorer)

# Print the mean and standard deviation
print("The mean of the errors is: %s." % np.mean(scores))
print("The standard deviation of the errors is: %s." % np.std(scores))
# %%
#! Tuning
#!
#! Find the best hyperparameter combination to optimize the R2 metric
score = make_scorer(r2_score)
min_val_error = float("inf")
error_increasing = 0
#! Tuning for n_estimators
gbr = GradientBoostingRegressor(max_depth=3, warm_start=True)

for n_estimators in range(1, 1000):
    gbr.n_estimators = n_estimators
    gbr.fit(X_train, y_train)
    
    y_pred = gbr.predict(X_test)
    val_error = mean_squared_error(y_test, y_pred)
    
    print('No. of estimators: ', gbr.n_estimators_)
    print('Validation error: ', val_error)
    
    if val_error < min_val_error:
        min_val_error = val_error
        error_increasing = 0
    else:
        error_increasing += 1
        if error_increasing == 10:
            break

n_estimators
# %%
gbr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('gbr', GradientBoostingRegressor(
        random_state = 123))])

parameters = {'gbr__n_estimators'    : [n_estimators],
              'gbr__learning_rate'   : [0.2, 0.1],
              'gbr__max_depth'       : [1, 2],
              'gbr__subsample'       : [0.8, 1.0]}

gridsearch = GridSearchCV(
    estimator=gbr, 
    param_grid=parameters, 
    cv=10,
    scoring = score,
    n_jobs = -1,
    verbose = 1
    )

gridsearch.fit(X_train, y_train)
print("Best parameter combination:", gridsearch.best_params_, "\n")

# Get the best model
model=gridsearch.best_estimator_
print(model, "\n")

# Evaluate the model using the test data
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("MSE:", mse)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
r2 = r2_score(y_test, predictions)
print("R2:", r2)

# Plot predicted vs actual
plt.scatter(y_test, predictions)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Previsões do Recalque')
# overlay the regression line
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='magenta')
save_fig('Previsões Recalque')
plt.show()


# %%
metricas = show_scores(model, 'gradienteboostingtunned')
metricas
# %%
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('gbr', GradientBoostingRegressor(
        learning_rate = 0.2,     
        n_estimators = 200,  
        max_depth = 2,   
        subsample = 0.8,
        random_state = 123))])
model.fit(X_train, y_train)
metricas = show_scores(model, 'test')
metricas
# %%
# Create scorer
mse_scorer = make_scorer(mean_squared_error)

# Implement LOOCV
scores = cross_val_score(model, X=X, y=y, cv=LeaveOneOut(), scoring=mse_scorer)

# Print the mean and standard deviation
print("The mean of the errors is: %s." % np.mean(scores))
print("The standard deviation of the errors is: %s." % np.std(scores))
# %%
#! Avaliando com a validação cruzada 

scores = cross_val_score(model, 
                         X_train, y_train,
                         scoring='neg_mean_squared_error',
                         cv = LeaveOneOut())
final_rmse_scores = np.sqrt(-scores)

def display_scores(scores):      
    print(f"Média dos erros(RMSE): {scores.mean():.2f}")
    print(f"Desvio padrão dos erros: {scores.std():.2f}")
    
display_scores(final_rmse_scores)
# %%
