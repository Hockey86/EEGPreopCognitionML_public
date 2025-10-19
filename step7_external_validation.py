from itertools import product
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from step5_ordinal_regression import get_ordinal_regression_perf


def main():
    random_seed = 2023
    cutoff = 15
    suffix = '_few_cov'
    yname = 'Pre.op.T.MOCA'

    # prepare external validation dataset
    data_path = 'dataset_external_validation.xlsx'
    if not os.path.exists(data_path):
        df1 = pd.read_csv('../Validation_Dataset_98_EEGFeatures_v3_LP35_170Hz.csv')
        df2 = pd.read_excel('../TABLA_FONDEF_UCHILE_PUC_230802_231101.xlsx', sheet_name='011123', skiprows=1)
        df2 = df2[df2.Excluded!='si'].reset_index(drop=True)
        df = df1.merge(df2, on='ID', how='inner', validate='1:1')
        df['SexF'] = (df.Sex=='F').astype(int)
        df.loc[df.education<8, 'education'] = 1
        df.loc[(df.education>=8)&(df.education<=11), 'education'] = 2
        df.loc[(df.education>=12)&(df.education<=13), 'education'] = 3
        df.loc[(df.education>=14)&(df.education<=16), 'education'] = 4
        df.loc[(df.education>=17)&(df.education<=18), 'education'] = 5
        df.loc[(df.education>=19)&(df.education<=20), 'education'] = 6
        df.loc[df.education>=21, 'education'] = 7
        df = df[pd.notna(df[yname])].reset_index(drop=True)

        # impute missing BMI
        df.loc[df.bmi=='NAN', 'bmi'] = np.nan
        nanids = np.where(pd.isna(df.bmi))[0]
        df2 = df[pd.notna(df.bmi)].reset_index(drop=True)
        vals = []
        for idx in nanids:
            ids = (df2.Age<=df.Age.iloc[idx]+5)&(df2.Age>=df.Age.iloc[idx]-5)
            ids &= (df2.SexF==df.SexF.iloc[idx])
            ids &= (df2.education<=df.education.iloc[idx]+2)&(df2.education>=df.education.iloc[idx]-2)
            vals.append( np.mean(df2[ids].bmi.values) )
        df.loc[nanids, 'bmi'] = vals

        df.to_excel(data_path, index=False)
    else:
        df = pd.read_excel(data_path)

    # combine <=cutoff
    df.loc[df[yname]<=cutoff, yname] = cutoff
    print(f'N(va) = {len(df)}')

    # get cross validation fold
    feature_setups = ['eeg+cov', 'cov', 'eeg']
    #model_names = ['inc_clf_xgb', 'inc_clf_rf', 'inc_clf_gbt', 'inc_clf_logreg', 'ltr_pair']
    model_names = ['inc_clf_logreg']
    model_dir = f'ordinal_regression_results_cutoff{cutoff}{suffix}'
    output_dir = 'external_validation'
    os.makedirs(output_dir, exist_ok=True)

    iters = list(product(model_names, feature_setups))
    df_perf = []
    for ii, (model_name, fs) in enumerate(tqdm(iters)):
        print(model_name, fs)
        # load model
        model_path = os.path.join(model_dir, f'result_{model_name}_{fs}.pickle')
        with open(model_path, 'rb') as ff:
            res = pickle.load(ff)
        model = res['model']
        Xnames = res['Xnames']
        # get data
        X = df[Xnames].values.astype(float)
        y = df[yname].values.astype(float)
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()
        # predict
        yp = model.predict(X)
        #ypp = model.predict_proba(X)
        # get performance
        perf = get_ordinal_regression_perf(y, yp, random_state=random_seed)
        print(perf)
        perf = perf.reset_index().rename(columns={'index':'Metric'})
        perf.insert(0, 'FeatureType', [fs]*len(perf))
        perf.insert(0, 'ModelName', [model_name]*len(perf))
        df_perf.append(perf)
        # save
        df_pred = pd.concat([
            df[['ID', yname]],
            pd.DataFrame(data={'Pred':yp}),
            #pd.DataFrame(data=ypp, columns=[f'P(x)' for x in model.named_steps['model'].classes_]),
            df[Xnames], ], axis=1)
        df_pred.to_csv(os.path.join(output_dir,f'external_pred_{model_name}_{fs}.csv'), index=False)

    df_perf = pd.concat(df_perf, axis=0, ignore_index=True)
    df_perf.to_csv(os.path.join(output_dir, f'external_perfs.csv'), index=False)


if __name__=='__main__':
    main()

