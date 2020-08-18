import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import NuSVR
pd.options.mode.chained_assignment = None

# Load the dataset
df = pd.read_csv('G:/Project/2020/Data/final_Dataset.csv', index_col=0)
target = df['HeatCapacity']
features = df[df.columns[4:]]

# Define search space
C = [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
nu = [0.1,0.2,0.3,0.4,0.5,0.6]

# Setup the grid to be searched over
param_grid = dict(C=C, nu=nu)

# Define grid search
grid_search = GridSearchCV(NuSVR(kernel='rbf'), param_grid, cv=LeaveOneOut(),
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
model = NuSVR(kernel='rbf', gamma='auto', C=best_params['C'], nu=best_params['nu'])
model.fit(x_train, y_train)

# save the final model as SPSS file 
filename = 'G:/Project/2020/final_NuSVR_model_Cp.sav'
pickle.dump(model, open(filename, 'wb'))
