import pandas as pd


# Drop columns with correlation above defined threshold
def drop_correlated_cols(df, cols_features, threshold):
    """
    Return a new dataframe, df, with columns correlated above threshold removed.
    The column removed is the right most one.
    """
    cols_to_remove = set()
    correlations_abs = df[cols_features].corr().abs()
    for i in range(len(correlations_abs.columns)):
        for j in range(i):
            if (correlations_abs.iloc[i, j] >= threshold) and (correlations_abs.columns[j] not in cols_to_remove):
                colname = correlations_abs.columns[i]
                if colname in df.columns:
                    cols_to_remove.add(colname)
                    df.drop(colname, axis=1, inplace=True)
    return df


# Load the raw dataset
df = pd.read_excel('C:/Users/Mohammed/Documents/Data/Dataset_Raw.xlsx', na_values=['na', 'nan'], index_col=0)
print(df.shape)

# Drop columns with empty cells
df.dropna(axis='columns', inplace=True)
print(df.shape)

# Drop columns that have only a unique value
count_unique = df.apply(pd.Series.nunique)
cols_to_drop = count_unique[count_unique == 1].index
df.drop(columns=cols_to_drop, inplace=True)
print(df.shape)

# Drop highly correlated features
cols_features = df.columns[2:]
df = drop_correlated_cols(df, cols_features, 0.8)
print(df.shape)



# Save dataset as csv file
df.to_csv('C:/Users/Mohammed/Documents/Data/Final_Dataset.csv')
