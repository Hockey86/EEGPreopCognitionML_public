import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
import pingouin as pg


def main():
    with pd.ExcelFile('dataset.xlsx') as ff:
        df1 = pd.read_excel(ff, sheet_name="Outcome")
        df2 = pd.read_excel(ff, sheet_name="Covariates")
        df3 = pd.read_excel(ff, sheet_name="EEGFeatures")
    yname = df1.columns[1]
    cov_cols = list(df2.columns)
    x_cols = list(df3.columns)
    df = pd.concat([df1, df2, df3], axis=1)

    #suffix = ''
    suffix = '_few_cov'
    if suffix=='_few_cov':
        cov_cols = ['Age', 'SexF', 'bmi', 'education']#, 'race_black', 'race_other', 'hispanic']

    all_res = []
    for xname in tqdm(x_cols):
        res = pg.partial_corr(data=df, x=xname, y=yname, covar=cov_cols, method='spearman')
        res['Xname'] = xname
        res['Yname'] = yname
        res2 = spearmanr(df[xname], df[yname])
        res['UnadjSpearmanR'] = res2.statistic
        res['UnadjSpearmanRPval'] = res2.pvalue
        all_res.append(res[['Xname', 'Yname', 'r', 'CI95%', 'p-val', 'n', 'UnadjSpearmanR', 'UnadjSpearmanRPval']])
    all_res = pd.concat(all_res, ignore_index=True, axis=0)
    all_res = all_res.sort_values('p-val', ignore_index=True)
    all_res['Sig'] = (all_res['p-val']<0.05).astype(int)
    all_res['SigBonf'] = (all_res['p-val']<0.05/len(all_res)).astype(int)
    all_res['SigBH'] = multipletests(all_res['p-val'], alpha=0.05, method='holm')[0].astype(int)
    print(all_res)
    all_res.to_excel(f'result_univariate{suffix}.xlsx', index=False)


if __name__=='__main__':
    main()

