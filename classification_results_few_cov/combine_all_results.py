import numpy as np
import pandas as pd


dfs = []
for model_type in ['xgb', 'rf', 'gbt', 'logreg']:
    for feat_type in ['eeg+cov', 'eeg', 'cov']:
        path = f'perf_cutoff_sens80_{model_type}_{feat_type}.csv'
        df = pd.read_csv(path)
        df.insert(0, 'FeatureType', [feat_type]*len(df))
        df.insert(0, 'ModelName', [model_type]*len(df))
        dfs.append(df)
dfs = pd.concat(dfs, axis=0, ignore_index=True)
dfs = dfs.rename(columns={'Unnamed: 0':'Metric'})
print(dfs)
dfs.to_excel('perf_all.xlsx', index=False)

