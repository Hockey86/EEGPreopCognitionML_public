import sys
from itertools import product
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.stats.mstats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve, matthews_corrcoef
from sklearn.metrics import make_scorer
from skopt import BayesSearchCV
from skopt.space import Integer
from mymodels import balanced_score, cindex_score, spearmanr_score


def train_model(model_name, df, Xnames, Yname, random_state=None, n_jobs=1, verbose=True):
    """
    """
    if random_state is None:
        random_state = np.random.randn(0,10000)

    Ncv = len(df.CV.unique())
    yp_cv = np.zeros(len(df))+np.nan
    hparams_cv = []
    models_cv = []
    for cvi in tqdm(range(Ncv+1), disable=not verbose):  #TODO assumes df.CV=0,1,2,...Ncv-1
        if cvi==Ncv:  # final fit
            dftr = df
        else:
            dftr = df[df.CV!=cvi].reset_index(drop=True)

        # define model
        if model_name=='ltr_pair':
            from mymodels import MonotonicL2LogisticRegression, LTRPairwise
            model = MonotonicL2LogisticRegression(random_state=random_state+(cvi+1)*2, max_iter=10000, pvalue_cutoff=1)
            model = LTRPairwise(model, class_weight='balanced', verbose=False)
            hparams = {
                'model__estimator__pvalue_cutoff': (0.99,1, 'log-uniform'),#0.1,0.5
                'model__estimator__C': (1e+3, 1e+3+0.01, 'log-uniform'),#1e-3, 1e+3
                'model__min_level_diff': Integer(1,4),}
        elif model_name=='inc_clf_logreg':
            from mymodels import MonotonicL2LogisticRegression, IncrementalClassifier
            model = MonotonicL2LogisticRegression(class_weight='balanced',
                    random_state=random_state+(cvi+1)*2, max_iter=10000, pvalue_cutoff=1)
            model = IncrementalClassifier(model)
            hparams = {
                'model__estimator__pvalue_cutoff': (0.99, 1, 'log-uniform'),
                'model__estimator__C': (1e+3, 1e+3+0.01, 'log-uniform'),}
        elif model_name=='inc_clf_gbt':
            from sklearn.ensemble import HistGradientBoostingClassifier
            from mymodels import MonotonicWrapper, IncrementalClassifier
            model = MonotonicWrapper(HistGradientBoostingClassifier(
                    max_leaf_nodes=31, min_samples_leaf=20,
                    l2_regularization=0.0, verbose=0,
                    random_state=random_state+(cvi+1)*2),
                    class_weight='balanced')
            model = IncrementalClassifier(model)
            hparams = {
                'model__estimator__pvalue_cutoff': (0.1, 0.5, 'log-uniform'),
                'model__estimator__estimator__learning_rate': (1e-3, 1e0, 'log-uniform'),
                'model__estimator__estimator__l2_regularization': (1e-3,1e3, 'log-uniform'),
                'model__estimator__estimator__max_iter': (10,50),
                'model__estimator__estimator__max_depth': (2,4),}
        elif model_name=='inc_clf_xgb':
            from xgboost import XGBClassifier
            from mymodels import MonotonicWrapper, IncrementalClassifier
            model = MonotonicWrapper(XGBClassifier(
                    random_state=random_state+(cvi+1)*2),
                    class_weight='balanced')
            model = IncrementalClassifier(model)
            hparams = {
                'model__estimator__pvalue_cutoff': (0.1, 0.5, 'log-uniform'),
                'model__estimator__estimator__eta': (0.,1.),
                'model__estimator__estimator__gamma': (1e-3,1e3, 'log-uniform'),
                'model__estimator__estimator__max_depth': (2,4),
                'model__estimator__estimator__min_child_weight': (1e-2, 1e2, 'log-uniform'),
                'model__estimator__estimator__subsample': (0.5,1.),}
        elif model_name=='inc_clf_rf':
            from sklearn.ensemble import RandomForestClassifier
            from mymodels import IncrementalClassifier
            model = RandomForestClassifier(
                    random_state=random_state+(cvi+1)*2,
                    class_weight='balanced')
            model = IncrementalClassifier(model)
            hparams = {
                'model__estimator__n_estimators': (10,200),
                'model__estimator__max_depth': (1,5),
                'model__estimator__min_samples_leaf': (1,50),
                'model__estimator__ccp_alpha': (1e-3, 1e3, 'log-uniform'), }
        else:
            raise NotImplementedError(f'Unknown model name {model_name}')
        scorer_ = make_scorer(balanced_score, greater_is_better=False, method='rmse')

        # select features
        # add standardization
        model = Pipeline([
            #('feature_selection', GenericUnivariateSelect(score_func=spearmanr_score, mode='fpr', param=0.1)),
            ('standardization', StandardScaler()),
            ('model', model), ])
        
        if cvi==Ncv:  # final fit
            for pn in hparams_cv[0]:
                vals = [hp[pn] for hp in hparams_cv]
                is_int = all([type(x)==int for x in vals])
                val = np.median(vals)
                if is_int:
                    val = int(val)
                model.set_params(**{pn:val})
        else:
            # hparam search
            model = BayesSearchCV(model, hparams,
                n_iter=5, scoring=scorer_, n_jobs=n_jobs, n_points=8,
                cv=Ncv, random_state=random_state+(cvi+1)*2+1,
                verbose=0)

        # fit
        model.fit(dftr[Xnames].values, dftr[Yname].values)
        if cvi<Ncv:
            hparams_cv.append(model.best_params_)
            model = model.best_estimator_

        if cvi==Ncv:  # final fit
            #ypp_final = model.predict_proba(dftr[Xnames].values)
            yp_final = model.predict(dftr[Xnames].values)
        else:
            models_cv.append(model)
            dfte = df[df.CV==cvi].reset_index(drop=True)
            #ypp_cv[df.CV==cvi] = model.predict_proba(dfte[Xnames].values)
            yp_cv[df.CV==cvi] = model.predict(dfte[Xnames].values)

    return models_cv, model, yp_cv, yp_final


def get_ordinal_regression_perf(y, yp, nbt=1000, verbose=True, random_state=None):
    """
    performance with CI
    """
    if random_state is None:
        random_state = np.random.randn(0,10000)
    np.random.seed(random_state)

    y = np.array(y).astype(int)
    corrs = []
    cindexs = []
    rmses = []
    maes = []
    bal_rmses = []
    bal_maes = []
    acc0s = []
    acc1s = []
    acc2s = []
    acc3s = []
    #acc4s = []
    #acc5s = []
    for bti in tqdm(range(nbt+1), disable=not verbose):
        try:
            if bti==0:
                ybt = y
                ypbt = yp
            else:
                btids = np.random.choice(len(y),len(y),replace=True)
                ybt = y[btids]
                ypbt = yp[btids]

            corrs.append( spearmanr(ybt, ypbt)[0] )
            cindexs.append( cindex_score(ybt, ypbt) )
            rmses.append( np.sqrt(np.mean((ybt-ypbt)**2)) )
            maes.append( np.mean(np.abs(ybt-ypbt)) )
            bal_rmses.append( balanced_score(ybt, ypbt, method='rmse') )
            bal_maes.append( balanced_score(ybt, ypbt, method='mae') )
            acc0s.append( np.mean(np.abs(ybt-ypbt)<=0) )
            acc1s.append( np.mean(np.abs(ybt-ypbt)<=1) )
            acc2s.append( np.mean(np.abs(ybt-ypbt)<=2) )
            acc3s.append( np.mean(np.abs(ybt-ypbt)<=3) )
            #acc4s.append( np.mean(np.abs(ybt-ypbt)<=4) )
            #acc5s.append( np.mean(np.abs(ybt-ypbt)<=5) )

        except Exception as ee:
            continue
    
    index = ['SpearmanR', 'CIndex', 'RMSE', 'MAE', 'WRMSE', 'WMAE', 'Acc0', 'Acc1', 'Acc2', 'Acc3']#, 'Acc4', 'Acc5']
    perf_data = [
        [corrs[0], np.percentile(corrs[1:], 2.5), np.percentile(corrs[1:], 97.5)],
        [cindexs[0], np.percentile(cindexs[1:], 2.5), np.percentile(cindexs[1:], 97.5)],
        [rmses[0], np.percentile(rmses[1:], 2.5), np.percentile(rmses[1:], 97.5)],
        [maes[0], np.percentile(maes[1:], 2.5), np.percentile(maes[1:], 97.5)],
        [bal_rmses[0], np.percentile(bal_rmses[1:], 2.5), np.percentile(bal_rmses[1:], 97.5)],
        [bal_maes[0], np.percentile(bal_maes[1:], 2.5), np.percentile(bal_maes[1:], 97.5)],
        [acc0s[0], np.percentile(acc0s[1:], 2.5), np.percentile(acc0s[1:], 97.5)],
        [acc1s[0], np.percentile(acc1s[1:], 2.5), np.percentile(acc1s[1:], 97.5)],
        [acc2s[0], np.percentile(acc2s[1:], 2.5), np.percentile(acc2s[1:], 97.5)],
        [acc3s[0], np.percentile(acc3s[1:], 2.5), np.percentile(acc3s[1:], 97.5)],
        #[acc4s[0], np.percentile(acc4s[1:], 2.5), np.percentile(acc4s[1:], 97.5)],
        #[acc5s[0], np.percentile(acc5s[1:], 2.5), np.percentile(acc5s[1:], 97.5)],
        ]
    perf = pd.DataFrame(data=np.array(perf_data), columns=['Val', 'LB', 'UB'], index=index)

    return perf


def main(cutoff=15):
    with pd.ExcelFile('dataset.xlsx') as ff:
        df1 = pd.read_excel(ff, sheet_name="Outcome")
        df2 = pd.read_excel(ff, sheet_name="Covariates")
        df3 = pd.read_excel(ff, sheet_name="EEGFeatures")
    sid_name, yname_ = df1.columns
    cov_cols = list(df2.columns)
    x_cols = list(df3.columns)
    df = pd.concat([df1, df2, df3], axis=1)
    assert len(df[sid_name].unique())==len(df), 'detected duplicates'

    #suffix = ''
    suffix = '_few_cov'
    if suffix=='_few_cov':
        cov_cols = ['Age', 'SexF', 'bmi', 'education']#, 'race_black', 'race_other', 'hispanic']

    # combine <=cutoff
    df.loc[df[yname_]<=cutoff, yname_] = cutoff

    # get cross validation fold
    random_state = 2023
    Ncv = 10
    cv_path = f'CV_{Ncv}fold_N{len(df)}_seed{random_state}.csv'
    df2 = pd.read_csv(cv_path)
    df = df.merge(df2, on=sid_name, how='inner', validate='1:1')

    feature_setups = {
        'eeg+cov':x_cols+cov_cols,
        #'cov':cov_cols,
        #'eeg':x_cols,
        #'alpha':['osAlpha_Power'],#clAlpha_Power_db'],
        }
    model_names = ['inc_clf_logreg']#, 'ltr_pair']#, 'inc_clf_xgb', 'inc_clf_rf', 'inc_clf_gbt']
    result_folder = f'ordinal_regression_results_cutoff{cutoff}{suffix}'
    os.makedirs(result_folder, exist_ok=True)
    n_jobs = 8
    yname = yname_

    iters = list(product(model_names, feature_setups.keys()))
    for ii, (model_name, fs) in enumerate(tqdm(iters)):
        """
        # permutation test
        save_path = os.path.join(result_folder, f'result_{model_name}_{fs}.pickle')
        with open(save_path, 'rb') as ff:
            aa = pickle.load(ff)
        vals2 =[balanced_score(df.loc[np.random.choice(len(df),len(df),replace=False),yname].values,aa['yp_cv']) for _ in tqdm(range(10000))]
        vals2=np.array(vals2)
        np.percentile(vals2,(2.5,25,50,75,97.5))
        """
        breakpoint()
        models_cv, model, yp_cv, yp_final = train_model(
            model_name, df, feature_setups[fs], yname,
            random_state=random_state+ii, n_jobs=n_jobs)
        perf_cv = get_ordinal_regression_perf(df[yname], yp_cv, random_state=random_state)
        print(model_name, fs)
        print(perf_cv)

        """
        # show coef
        dd={'Name':aa['Xnames']}
        for xi,x in enumerate(['16','17','18','19','20','21','22']):dd[x] = aa['model'].steps[-1][-1].estimators[xi].base_estimator.coef_.flatten()
        pd.DataFrame(data=dd).to_excel('coefs_inc_clf_lr_eeg+cov.xlsx', index=False)
        """

        # save
        save_path = os.path.join(result_folder, f'result_{model_name}_{fs}.pickle')
        with open(save_path, 'wb') as ff:
            pickle.dump({'models_cv':models_cv, 'model':model,
                'yp_cv':yp_cv, 'yp_final':yp_final, 'Xnames':feature_setups[fs],
                }, ff)

        save_path = os.path.join(result_folder, f'perf_{model_name}_{fs}.csv')
        perf_cv.to_csv(save_path)
        
        if model_name=='ltr_pair':
            save_path = os.path.join(result_folder, f'coef_{model_name}_{fs}.csv')
            df_coef = pd.DataFrame(data={'Name':feature_setups[fs], 'Coef':np.zeros(len(feature_setups[fs]))})
            df_coef.loc[:, 'Coef'] = model.steps[-1][-1].coef_.flatten()
            df_coef = df_coef.sort_values('Coef', ascending=False, ignore_index=True)
            df_coef.to_csv(save_path, index=False)


if __name__=='__main__':
    #cutoff = int(sys.argv[1])
    main()#cutoff)

