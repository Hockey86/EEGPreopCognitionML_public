import pandas as pd
from sklearn.model_selection import StratifiedKFold


def main():
    with pd.ExcelFile('dataset.xlsx') as ff:
        df1 = pd.read_excel(ff, sheet_name="Outcome")
        df2 = pd.read_excel(ff, sheet_name="Covariates")
        df3 = pd.read_excel(ff, sheet_name="EEGFeatures")
    sid_name, yname = df1.columns
    cov_cols = list(df2.columns)
    x_cols = list(df3.columns)
    df = pd.concat([df1, df2, df3], axis=1)
    assert len(df[sid_name].unique())==len(df), 'detected duplicates'

    # generate stratefied K-fold
    random_seed = 2023
    Ncv = 10
    cv_path = f'CV_{Ncv}fold_N{len(df)}_seed{random_seed}.csv'
    df.loc[df[yname]<=15, 'Ycoarse']=0
    df.loc[df[yname]>15, 'Ycoarse']=df.loc[df[yname]>15, yname]
    cvf = StratifiedKFold(n_splits=Ncv, shuffle=True, random_state=random_seed)
    for i, (_, idte) in enumerate(cvf.split(df[x_cols], df.Ycoarse)):
        df.loc[idte, 'CV'] = i
    df[[sid_name, 'CV']].to_csv(cv_path, index=False)


if __name__=='__main__':
    main()

