import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler



def get_sensitivity_scores(model, features, top_n):
    """
    Finds the sensitivity of each feature in features for model. Returns the top_n
    feature names, features_top, alongside the sensitivity values, scores_top.
    """
    # Get just the values of features
    x_train = features.values
    # Apply min max normalization
    scaler = MinMaxScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    # Find mean and standard deviation of each feature
    x_train_avg = np.mean(x_train, axis=0).reshape(1, -1)
    x_train_std = np.std(x_train, axis=0).reshape(1, -1)
    prediction_mean = model.predict(x_train_avg)

    scores_max = []
    # Iterate over each feature
    for i in range(x_train_avg.shape[1]):
        # Copy x_train_avg
        x_train_i = x_train_avg.copy()
        # Add the standard deviation of i to that column
        x_train_i[:, i] = x_train_i[:, i] + x_train_std[:, i]
        result_i = model.predict(x_train_i)
        # Take the difference and divide by standard deviation
        diff = (result_i - prediction_mean) / x_train_std[:, i]
        scores_max.append(diff.flatten()[0])
    scores_max = np.absolute(scores_max)
    indices_top = np.argsort(scores_max)[-top_n:]
    features_top = features.iloc[:, indices_top].columns
    scores_top = scores_max[indices_top]
    return features_top, scores_top


# Create result sheet
df_sensitivity = pd.DataFrame()
n_top = 10

# Load dataset
df = pd.read_csv('G:/Project/310/final_Dataset.csv', index_col=0)
features = df[df.columns[4:]]

# Load SVR model
filename = 'G:/Project/2020/models/final_SVR_model_Cp.sav'
model = pickle.load(open(filename, 'rb'))

features_top, scores_top = get_sensitivity_scores(model, features, n_top)
df_sensitivity['SVR features'] = features_top
df_sensitivity['SVR sensitivity values (abs)'] = scores_top

#Load RFR model 
filename = 'G:/Project/2020/models/final_RFR_model_Cp.sav'
model = pickle.load(open(filename,'rb'))

features_top, scores_top = get_sensitivity_scores(model, features, n_top)
df_sensitivity['RFR features'] = features_top
df_sensitivity['RFR sensitivity values (abs)'] = scores_top

#Load NuSVR model 
filename = 'G:/Project/2020/models/final_NuSVR_model_Cp.sav'
model = pickle.load(open(filename,'rb'))

#find top features 
features_top, scores_top = get_sensitivity_scores(model, features, n_top)
df_sensitivity['NuSVR features'] = features_top
df_sensitivity['NuSVR sensitivity values (abs)'] = scores_top

#Save the results 
df_sensitivity.to_csv('G:/Project/310/sensitivity_analysis_Cp_3models.csv')
