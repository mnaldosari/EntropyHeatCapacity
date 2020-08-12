import numpy as np
import pandas as pd
import csv
from sklearn.metrics import  mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None

# Load the dataset
df = pd.read_csv('C:/Users/Mohammed/Documents/Data/final_Dataset.csv', index_col=0)
target = df['HeatCapacity']
features = df[df.columns[4:]]

# Define the search space of hyperparameters
Cs = [1000,2000,3000,4000,5000,6000,7000,8000,9000]
epsilons = [0.1, 0.2, 0.3,0.4,0.5]

# Setup the grid to be searched over
param_grid = dict(C=Cs, epsilon=epsilons)

# Define outer folds
kFolds = LeaveOneOut().split(X=features.values, y=target.values)


# Define inner folds
grid_search = GridSearchCV(SVR(kernel='rbf', gamma='auto'), param_grid, cv=LeaveOneOut(),
                           n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')

# Open results file and write out headers
out_file = open("C:/Users/Mohammed/Documents/Results/SVR_LOOCV.csv", 'w')
wr = csv.writer(out_file, dialect='excel')
headers = ['C', 'epsilon',"MAE",'Experimental Values', "Predicted Values"]
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
    row = [best_params['C'], best_params['epsilon'],error_ma, y_test ,predictions]
    wr.writerow(row)
    out_file.flush()

out_file.close()
