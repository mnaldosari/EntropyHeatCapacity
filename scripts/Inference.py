import pandas as pd
import pickle
import argparse
from keras.models import load_model
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None


# Command line argument to determine what model to use for inference
parser = argparse.ArgumentParser(description='SVR, RFR, or NuSVR')
parser.add_argument('model_type', type=str, nargs=1)
model_type =parser.parse_args().model_type[0]

# Load the entire set
df = pd.read_csv('G:/Project/2020/Data/final_Dataset.csv', index_col=0)
target = df['HeatCapacity']
features = df[df.columns[4:]]
test_data = pd.read_csv('G:/Project/2020/Data/Small_set_raw.csv')


# Reduce to the same columns used in features
df_test = test_data[list(features.columns)]
df_test.dropna(inplace=True)

scaler = MinMaxScaler()
scaler.fit(features.values)
x_test = scaler.transform(df_test.values)

# Load the model from disk
model = None

if model_type == 'SVR':
    filename = 'G:/Project/2020/models/final_SVR_model_Cp.sav'
    model = pickle.load(open(filename, 'rb'))
    
if model_type == 'RFR':
    filename = 'G:/Project/2020/models/final_RFR_model_Cp.sav'
    model = pickle.load(open(filename, 'rb'))
    
if model_type == 'NuSVR':
    filename = 'G:/Project/2020/models/final_NuSVR_model_Cp.sav'
    model = pickle.load(open(filename, 'rb'))   

    
# Infer
predictions = model.predict(x_test).flatten()
print(model_type)
for p in predictions:
    print(p)
