import pickle
import numpy as np
import pandas as pd
from scipy.stats import *
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from pingouin import partial_corr


def get_mannwhitneyu_result(y, yp, covariate=None):
    y0 = yp[y==0]; y1 = yp[y==1]
    try:
        test = mannwhitneyu(y0,y1)
        pval = test.pvalue
    except Exception as ee:
        pval = np.nan
    return { 'TestType':'MannWhitneyU',
        'N(-)':len(y0), 'N(+)':len(y1), 'N':len(y0)+len(y1),
        'Median(-)':np.median(y0), 'Median(+)':np.median(y1),
        'Statistic':np.nan, 'PValue':pval}


def get_chi2_result(y, yp, covariate=None):
    y = LabelEncoder().fit_transform(y)
    yp = LabelEncoder().fit_transform(yp)
    y0 = yp[y==0]; y1 = yp[y==1]
    try:
        assert set(y)=={0,1}, set(y)
        cf = confusion_matrix(y,yp)
        test = chi2_contingency(cf)
        pval = test.pvalue
    except Exception as ee:
        pval = np.nan
    return {'TestType':'Chi2',
        'N(-)':len(y0), 'N(+)':len(y1), 'N':len(y0)+len(y1),
        'Median(-)':np.nan, 'Median(+)':np.nan,
        'Statistic':np.nan, 'PValue':pval}


def get_spearmanr_result(y, yp, covariate=None):
    try:
        if covariate is None:
            res = spearmanr(y, yp)
            corr = res.statistic
            pval = res.pvalue
        else:
            df_ = pd.DataFrame(data={'a':y, 'b':yp, 'c':covariate})
            res = partial_corr(data=df_, x='a', y='b', covar='c', method='spearman')
            corr = res['r'].iloc[0]
            pval = res['p-val'].iloc[0]
    except Exception as ee:
        corr = np.nan
        pval = np.nan
    return {'TestType':'SpearmanCorr',
            'N(-)':np.nan, 'N(+)':np.nan, 'N':len(y),
            'Median(-)':np.nan, 'Median(+)':np.nan,
            'Statistic':corr, 'PValue':pval}


def main():
    or_bin_cutoff = 18
    data_type = 'eeg+cov'
    suffix = '_adjust_dex'

    with open(f'ordinal_regression_results_cutoff15_few_cov/result_ltr_pair_{data_type}.pickle','rb') as ff:
        res = pickle.load(ff)
    ypr_lr_pair = res['yp_cv'].astype(int)
    #ypb = np.digitize(ypr_lr_pair, (16,22))
    yprb_lr_pair = (ypr_lr_pair>=or_bin_cutoff).astype(int)
    with open(f'ordinal_regression_results_cutoff15_few_cov/result_inc_clf_logreg_{data_type}.pickle','rb') as ff:
        res = pickle.load(ff)
    ypr_lr = res['yp_cv'].astype(int)
    yprb_lr = (ypr_lr>=or_bin_cutoff).astype(int)
    with open(f'ordinal_regression_results_cutoff15_few_cov/result_inc_clf_rf_{data_type}.pickle','rb') as ff:
        res = pickle.load(ff)
    #aa=np.array([res['model'].steps[-1][-1].estimators[x].base_estimator.feature_importances_ for x in range(7)])
    #kk=pd.DataFrame(data={'Xname':res['Xnames'],'Importance':aa.mean(axis=0)}).sort_values('Importance',ascending=False)
    ypr_rf = res['yp_cv'].astype(int)
    yprb_rf = (ypr_rf>=or_bin_cutoff).astype(int)
    with open(f'ordinal_regression_results_cutoff15_few_cov/result_inc_clf_xgb_{data_type}.pickle','rb') as ff:
        res = pickle.load(ff)
    ypr_xgb = res['yp_cv'].astype(int)
    yprb_xgb = (ypr_xgb>=or_bin_cutoff).astype(int)
    with open(f'ordinal_regression_results_cutoff15_few_cov/result_inc_clf_gbt_{data_type}.pickle','rb') as ff:
        res = pickle.load(ff)
    ypr_gbt = res['yp_cv'].astype(int)
    yprb_gbt = (ypr_gbt>=or_bin_cutoff).astype(int)

    with open(f'classification_results_few_cov/result_cutoff_sens80_logreg_{data_type}.pickle','rb') as ff:
        res = pickle.load(ff)
    ypc_lr = (res['yp_cv']>res['thres']['Youden']).astype(int)
    with open(f'classification_results_few_cov/result_cutoff_sens80_rf_{data_type}.pickle','rb') as ff:
        res = pickle.load(ff)
    ypc_rf = (res['yp_cv']>res['thres']['Youden']).astype(int)
    with open(f'classification_results_few_cov/result_cutoff_sens80_xgb_{data_type}.pickle','rb') as ff:
        res = pickle.load(ff)
    ypc_xgb = (res['yp_cv']>res['thres']['Youden']).astype(int)
    with open(f'classification_results_few_cov/result_cutoff_sens80_gbt_{data_type}.pickle','rb') as ff:
        res = pickle.load(ff)
    ypc_gbt = (res['yp_cv']>res['thres']['Youden']).astype(int)

    df = pd.read_excel('dataset.xlsx')
    df2 = pd.read_csv('../380_Combined_EEG_Clinical_rev2_equal_fs.csv')
    df = df.merge(df2.drop(columns='Pre.op.T.MOCA'), on='id', how='left', validate='1:1')
    if suffix=='_adjust_dex':
        df.loc[df.treatment=='Dexmed', 'treatment'] = 1
        df.loc[df.treatment=='Placebo', 'treatment'] = 0
        df['treatment'] = df.treatment.astype(float)
    y = df['Pre.op.T.MOCA'].values
    y[y<=15] = 15
    yb = (y>=or_bin_cutoff).astype(int)

    """
    df = pd.read_csv(f'external_validation/external_pred_ltr_pair_{data_type}.csv')
    ypr_lr_pair = df.Pred.values.astype(int)
    yprb_lr_pair = (ypr_lr_pair>=or_bin_cutoff).astype(int)
    df = pd.read_csv(f'external_validation/external_pred_inc_clf_logreg_{data_type}.csv')
    ypr_lr = df.Pred.values.astype(int)
    yprb_lr = (ypr_lr>=or_bin_cutoff).astype(int)
    df = pd.read_csv(f'external_validation/external_pred_inc_clf_rf_{data_type}.csv')
    ypr_rf = df.Pred.values.astype(int)
    yprb_rf = (ypr_rf>=or_bin_cutoff).astype(int)
    df = pd.read_csv(f'external_validation/external_pred_inc_clf_xgb_{data_type}.csv')
    ypr_xgb = df.Pred.values.astype(int)
    yprb_xgb = (ypr_xgb>=or_bin_cutoff).astype(int)
    df = pd.read_csv(f'external_validation/external_pred_inc_clf_gbt_{data_type}.csv')
    ypr_gbt = df.Pred.values.astype(int)
    yprb_gbt = (ypr_gbt>=or_bin_cutoff).astype(int)
    df = pd.read_excel('dataset_external_validation.xlsx')
    y = df['Pre.op.T.MOCA'].values.astype(int)
    y[y<=15] = 15
    yb = (y>=or_bin_cutoff).astype(int)
    suffix += '-external'
    """

    y_types = {
        'actual':'cont',
        'actual_bin':'bin',
        'predicted_ordinal_lr_pair':'cont',
        'predicted_ordinal_lr':'cont',
        'predicted_ordinal_rf':'cont',
        'predicted_ordinal_gbt':'cont',
        'predicted_ordinal_xgb':'cont',
        'predicted_ordinal_bin_lr_pair':'bin',
        'predicted_ordinal_bin_lr':'bin',
        'predicted_ordinal_bin_rf':'bin',
        'predicted_ordinal_bin_gbt':'bin',
        'predicted_ordinal_bin_xgb':'bin',
        #'predicted_classif_lr':'bin',
        #'predicted_classif_rf':'bin',
        #'predicted_classif_gbt':'bin',
        #'predicted_classif_xgb':'bin',
        }
    cols = {
        'died30':'bin',
        'died90':'bin',
        'died180':'bin',
        #'icu_readmission':'bin',#
        'readmission':'bin',
        'delirium1to3':'bin',
        'delirium_sev1to3':'cont',
        'hosp_los':'cont',
        #'icu_los':'cont',#
        #'tmoca30':'cont',#
        #'tmoca90':'cont',#
        #'tmoca180':'cont',#
        }
    test_funcs = {
        ('cont', 'cont'): get_spearmanr_result,
        ('cont', 'bin'): get_mannwhitneyu_result,
        ('bin', 'cont'): get_mannwhitneyu_result,
        ('bin', 'bin'): get_chi2_result,
        }
    df_res = []
    y_type_sig_count = {x:0 for x in y_types}

    for col, col_type in cols.items():
        for yt, y_type in y_types.items():
            #print(col, yt)
            if yt=='actual':
                y_ = y
            elif yt=='actual_bin':
                y_ = yb
            elif yt=='predicted_ordinal_lr_pair':
                y_ = ypr_lr_pair
            elif yt=='predicted_ordinal_lr':
                y_ = ypr_lr
            elif yt=='predicted_ordinal_rf':
                y_ = ypr_rf
            elif yt=='predicted_ordinal_gbt':
                y_ = ypr_gbt
            elif yt=='predicted_ordinal_xgb':
                y_ = ypr_xgb
            elif yt=='predicted_ordinal_bin_lr_pair':
                y_ = yprb_lr_pair
            elif yt=='predicted_ordinal_bin_lr':
                y_ = yprb_lr
            elif yt=='predicted_ordinal_bin_rf':
                y_ = yprb_rf
            elif yt=='predicted_ordinal_bin_gbt':
                y_ = yprb_gbt
            elif yt=='predicted_ordinal_bin_xgb':
                y_ = yprb_xgb
            elif yt=='predicted_classif_lr':
                y_ = ypc_lr
            elif yt=='predicted_classif_rf':
                y_ = ypc_rf
            elif yt=='predicted_classif_xgb':
                y_ = ypc_xgb
            elif yt=='predicted_classif_gbt':
                y_ = ypc_gbt
            df[col] = df[col].astype(float)

            if 'die' in col:
                ids = (~pd.isna(df[col]))&(~pd.isna(y_))
            # remove died people to avoid bias
            elif col=='tmoca30':
                ids = (~pd.isna(df[col]))&(~pd.isna(y_))&(df.died30.values==0)
            elif col=='tmoca90':
                ids = (~pd.isna(df[col]))&(~pd.isna(y_))&(df.died90.values==0)
            else:  # elif col=='tmoca180':
                ids = (~pd.isna(df[col]))&(~pd.isna(y_))&(df.died180.values==0)
            y_ = y_[ids]
            df_ = df[ids].reset_index(drop=True)
            func = test_funcs[(y_type, col_type)]
            covariate = df_['treatment'] if suffix=='_adjust_dex' else None
            if y_type=='bin' and col_type=='cont':
                res = func(y_, df_[col], covariate=covariate)
            else:
                res = func(df_[col], y_, covariate=covariate)
            res = {'Outcome':col, 'TMoCAType':yt} | res | {'Significance':'*' if res['PValue']<0.05 else ''}
            df_res.append(res)
            if res['Significance']=='*':
                y_type_sig_count[yt] += 1 

    df_res = pd.DataFrame(data=df_res)
    #cols = ['Outcome', 'TMoCAType', 'TestType', 'N(-)', 'N(+)', 'N', 'Median(-)', 'Median(+)', 'Statistic', 'PValue']
    #df_res = df_res[cols]
    print(df_res)
    df_count = pd.Series(data=y_type_sig_count)
    df_count = df_count.sort_values(ascending=False)
    print(df_count)

    with pd.ExcelWriter(f"correlation_with_outcomes{suffix}.xlsx") as writer:
        df_res.to_excel(writer, sheet_name="Correlation", index=False)
        df_count.to_excel(writer, sheet_name="Count")


if __name__=='__main__':
    main()

