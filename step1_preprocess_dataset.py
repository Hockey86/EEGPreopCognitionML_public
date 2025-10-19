import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


def main():
    df1 = pd.read_csv('../380_Combined_EEG_Clinical_rev2_equal_fs.csv')
    df1 = df1[df1.id.str.startswith('MINDDS')].reset_index(drop=True)
    df1['SexF'] = (df1.Sex=='F').astype(int)
    """
    df1['clDelta_Power_db'] = 10*np.log10(df1.clDelta_Power)
    df1['clTheta_Power_db'] = 10*np.log10(df1.clTheta_Power)
    df1['clAlpha_Power_db'] = 10*np.log10(df1.clAlpha_Power)
    df1['clBeta_Power_db'] = 10*np.log10(df1.clBeta_Power)
    df1['Total_Power_db'] = 10*np.log10(df1.Total_Power)
    """

    df2 = pd.read_excel('../MINDDS_Dem_Cognitive_EEG_data_June_23_2023.xlsx')
    df2['race_white'] = (df2.race==1).astype(int)
    df2['race_black'] = (df2.race==3).astype(int)
    df2['race_other'] = ((df2.race!=1)&(df2.race!=3)).astype(int)
    df2['hispanic'] = (df2.hispanic=='Yes').astype(int)
    df2.loc[df2.tobacco=='Never smoker', 'tobacco'] = 0
    df2.loc[df2.tobacco=='Former smoker', 'tobacco'] = 1
    df2.loc[df2.tobacco=='Current some day smoker', 'tobacco'] = 2
    df2.loc[df2.tobacco=='Current every day smoker', 'tobacco'] = 2
    df2['tobacco'] = df2.tobacco.astype(int)
    df2.loc[df2.htn=='Yes', 'htn'] = 1
    df2.loc[df2.htn=='No', 'htn'] = 0
    df2['htn'] = df2.htn.astype(int)
    df2.loc[df2.diabetes=='Yes', 'diabetes'] = 1
    df2.loc[df2.diabetes=='No', 'diabetes'] = 0
    df2['diabetes'] = df2.diabetes.astype(int)
    df2.loc[df2.sleep_apnea=='Yes', 'sleep_apnea'] = 1
    df2.loc[df2.sleep_apnea=='No', 'sleep_apnea'] = 0
    df2['sleep_apnea'] = df2.sleep_apnea.astype(int)
    df2.loc[(df2.stroke=='Yes')|(df2.cerebrovascular_dx=='Yes'), 'stroke'] = 1
    df2.loc[(df2.stroke=='No')&(df2.cerebrovascular_dx=='No'), 'stroke'] = 0
    df2['stroke'] = df2.stroke.astype(int)
    df2.loc[df2.prior_mi=='Yes', 'mi'] = 1
    df2.loc[df2.prior_mi=='No', 'mi'] = 0
    df2['mi'] = df2.mi.astype(int)
    df2.loc[df2.peripheral_art_dx=='Yes', 'peripheral_art_dx'] = 1
    df2.loc[df2.peripheral_art_dx=='No', 'peripheral_art_dx'] = 0
    df2['peripheral_art_dx'] = df2.peripheral_art_dx.astype(int)
    df2.loc[df2.liver_dx=='Yes', 'liver_dx'] = 1
    df2.loc[df2.liver_dx=='No', 'liver_dx'] = 0
    df2['liver_dx'] = df2.liver_dx.astype(int)
    df2.loc[df2.prev_cardiac_intervent=='Yes', 'prev_cardiac_intervent'] = 1
    df2.loc[df2.prev_cardiac_intervent=='No', 'prev_cardiac_intervent'] = 0
    df2['prev_cardiac_intervent'] = df2.prev_cardiac_intervent.astype(int)

    sid_name = 'id'
    outcome_name = 'Pre.op.T.MOCA'

    cov_names = [
        'Age', 'SexF', 'bmi', 'education', 'race_black', 'race_other', 'hispanic',#'marital', 'race_white', 'promis_global0',
        'tobacco', 'illicit_drugs',
        'htn', 'diabetes', 'sleep_apnea', 'stroke',
        'mi', 'peripheral_art_dx', 'afib', 'prev_cardiac_intervent',
        'chronic_lung_dx', 'renal_failure', 'liver_dx',
        # not found from Lancet 2020 Dementia:
        # hearing impairment, depression, alcohol, TBI, air pollution
    ]

    eeg_names = [
        #'clDelta_Power', 'clTheta_Power', 'clAlpha_Power', 'clBeta_Power', 'Total_Power',
        'clDelta_Power_db', 'clTheta_Power_db', 'clAlpha_Power_db', 'clBeta_Power_db', 'Total_Power_db',
        'osAlpha_freq', 'osAlpha_Power', 'osAlpha_BW', 'osAlpha_Prev',
        'Offset', 'Slope', 'error', 'nPeaksTotalFound', #'r2', 'BadFits_Percent',
        'LZc', 'PermEntropy', 'DispEntropy', 'h_complexity', 'num_zerocross', 'Higuchi_FD', 'DFA',#, 'h_mobility' # strong corr with h_complexity
        'Coh_Inter', 'Coh_Intra', 'wPLI_Inter', 'wPLI_Intra', 'MI_Inter', 'MI_Intra',
    ]

    df = df1[['id','Age','SexF','Pre.op.T.MOCA']+eeg_names].merge(df2[['id']+cov_names[2:]], on='id', how='inner', validate='1:1')
 
    #df = df[(df.race==1)&(df.hispanic=='No')].reset_index(drop=True)
    df = df[pd.notna(df[sid_name])].reset_index(drop=True)
    #df = df.dropna(subset=[sid_name, outcome_name]+cov_names+eeg_names).reset_index(drop=True)

    # impute missing value
    """
    idx = np.where(pd.isna(df).any(axis=1))[0]
    assert len(idx)==1
    idx = idx[0]
    bin_cols = [col for col in cov_names+eeg_names if set(df[col].unique())==set([0,1])]
    nonbin_cols = [col for col in cov_names+eeg_names if col not in bin_cols]
    bin_equal_count = [ (df[col]==df[col].iloc[idx]) for col in bin_cols]
    good_ids = np.array(bin_equal_count).sum(axis=0)>=len(bin_cols)-2
    df2 = df.iloc[good_ids].reset_index(drop=True)
    df2 = df2[(df2.Age<=df.Age.iloc[idx]+5)&(df2.Age>=df.Age.iloc[idx]-5)].reset_index(drop=True)
    X = df2[nonbin_cols].values
    Xmean = np.nanmean(X, axis=0)
    Xstd = np.nanstd(X, axis=0)
    X = (X-Xmean)/Xstd
    X2 = KNNImputer(n_neighbors=10).fit_transform(X)*Xstd+Xmean
    df.loc[idx, pd.isna(df.iloc[idx])] = X2[np.isnan(X)]
    """
    X = df.iloc[:,1:].values
    Xmean = np.nanmean(X, axis=0)
    Xstd = np.nanstd(X, axis=0)
    X = (X-Xmean)/Xstd
    X2 = KNNImputer(n_neighbors=10).fit_transform(X)*Xstd+Xmean
    df.loc[:,df.columns[1:]] = X2

    print(df)
    df2 = df.describe().T
    print(df2)
    df2.to_excel('dataset_summary.xlsx')
    with pd.ExcelWriter("dataset.xlsx") as writer:
        df[[sid_name, outcome_name]].to_excel(writer, sheet_name="Outcome", index=False)
        df[cov_names].to_excel(writer, sheet_name="Covariates", index=False)
        df[eeg_names].to_excel(writer, sheet_name="EEGFeatures", index=False)


if __name__=='__main__':
    main()

