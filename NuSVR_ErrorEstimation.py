import numpy as np
import pandas as pd
import csv
from sklearn.metrics import  mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.svm import NuSVR
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None

# Load the dataset
df = pd.read_csv('G:/Project/2020/Data/final_Dataset.csv', index_col=0)
target = df['HeatCapacity']
features = df[df.columns[4:]]

# Define the search space of hyperparameters

C = [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
nu = [0.1,0.2,0.3,0.4,0.5,0.6]

# Setup the grid to be searched over
param_grid = dict(C=C, nu=nu)


# Define outer folds
kFolds = LeaveOneOut().split(X=features.values, y=target.values)


# Define inner folds
grid_search = GridSearchCV(NuSVR(kernel='rbf'), param_grid, cv=LeaveOneOut(),
                           n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')

# Open results file and write out headers
out_file = open("./v-SVR.csv", 'w')
wr = csv.writer(out_file, dialect='excel')
headers = ['C', 'v',"MAE",'Experimental Values', "Predicted Values"]
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
    grid_search.fit(x_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_

    # Predictions of the test set and calculations of the errors 
    predictions = grid_search.predict(x_test)
    error_ma = mean_absolute_error(y_test, predictions)
   
    

    # Write results
    row = [best_params['C'], best_params['nu'],error_ma, y_test ,predictions]
    wr.writerow(row)
    out_file.flush()

out_file.close()
