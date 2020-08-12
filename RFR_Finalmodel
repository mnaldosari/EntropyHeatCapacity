import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import csv
pd.options.mode.chained_assignment = None

# Load the dataset
df = pd.read_csv('C:/Users/Mohammed/Documents/Data/final_Dataset.csv', index_col=0)
target = df['HeatCapacity']
features = df[df.columns[4:]]

# Define search space
n_estimators=[100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,2000]

# Setup the grid to be searched over
param_grid = dict(n_estimators=n_estimators)

# Define grid search
grid_search = GridSearchCV(RandomForestRegressor(n_jobs=-1,random_state=9), param_grid, cv=LeaveOneOut(),
                           n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')

# Split data in to features and target
x_train = features.values
y_train = target.values

# Normalization
scaler = MinMaxScaler().fit(x_train)
x_train = scaler.transform(x_train)

# Find the best parameters
grid_search.fit(x_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)

# Retrain model with best parameters found from grid search
best_params = grid_search.best_params_
model= RandomForestRegressor(n_jobs=-1,random_state=9,n_estimators=best_params['n_estimators'])
model.fit(x_train, y_train)

# save the final model as SPSS file 
filename = './final_RFR_model_Cp.sav'
pickle.dump(model, open(filename, 'wb'))
