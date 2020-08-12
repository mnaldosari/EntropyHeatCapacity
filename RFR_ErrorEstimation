import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
import csv
from sklearn.metrics import mean_absolute_error

#Load the Dataset 
df = pd.read_csv('C:/Users/Mohammed/Documents/Data/final_Dataset.csv', index_col=0)
target = df['HeatCapacity']
features = df[df.columns[4:]]

# Define the search space of hyperparameters
n_estimators=[100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,2000]
                 
param_grid = dict(n_estimators=n_estimators)

# Define outer folds
kFolds = LeaveOneOut().split(X=features.values, y=target.values)
                 
# Define inner folds
GS = GridSearchCV(RandomForestRegressor(n_jobs=-1,random_state=9), param_grid,
                  cv=LeaveOneOut(), n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')

# Open results file and write out headers
out_file = open("C:/Users/Mohammed/Documents/Results/RFR_LOOCV.csv", 'w')
wr = csv.writer(out_file, dialect='excel')
headers = ["n_estimators", "error_ma", "Experimental Values", "Predicted Values"]
wr.writerow(headers)
out_file.flush()

for index_train, index_test in kFolds:
    # Get train and test splits
    x_train, x_test = features.iloc[index_train].values, features.iloc[index_test].values
    y_train, y_test = target.iloc[index_train].values, target.iloc[index_test].values

    # Normalization
    scaler = MinMaxScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # Fitting
    GS.fit(x_train, y_train)
    
     # Get the best parameters
    best_params = GS.best_params_

    # Predictions of the test set and calculations of the errors 
    predictions = GS.predict(x_test)
    error_ma = mean_absolute_error(y_test, predictions)
    

    # Write results
    row = [best_params['n_estimators'], error_ma, y_test,predictions]
    wr.writerow(row)
    out_file.flush()

out_file.close()
