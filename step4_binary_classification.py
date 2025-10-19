from itertools import product
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve, matthews_corrcoef
from skopt import BayesSearchCV
#from mymodels import mannwhitneyu_score


def train_classifier(model_name, df, Xnames, Yname, random_state=None, n_jobs=1, verbose=True):
    """
    """
    if random_state is None:
        random_state = np.random.randn(0,10000)

    Ncv = len(df.CV.unique())
    ypp_cv = np.zeros(len(df))+np.nan
    hparams_cv = []
    models_cv = []
    for cvi in tqdm(range(Ncv+1), disable=not verbose):  #TODO assumes df.CV=0,1,2,...Ncv-1
        if cvi==Ncv:  # final fit
            dftr = df
        else:
            dftr = df[df.CV!=cvi].reset_index(drop=True)

        # define model
        if model_name=='logreg':
            from mymodels import MonotonicL2LogisticRegression
            model = MonotonicL2LogisticRegression(class_weight='balanced',
                random_state=random_state+(cvi+1)*2, max_iter=10000, pvalue_cutoff=1)
            hparams = {
                #'feature_selection__param': (0.1, 0.5, 'log-uniform'),
                'model__pvalue_cutoff':(0.1,0.5,'log-uniform'),
                'model__C': (1e-3, 1e+3, 'log-uniform'),}
        elif model_name=='gbt':
            from sklearn.ensemble import HistGradientBoostingClassifier
            from mymodels import MonotonicWrapper
            model = MonotonicWrapper(HistGradientBoostingClassifier(
                    max_leaf_nodes=31, min_samples_leaf=20,
                    l2_regularization=0.0, verbose=0,
                    random_state=random_state+(cvi+1)*2),
                    class_weight='balanced')
            hparams = {
                'model__pvalue_cutoff': (0.1, 0.5, 'log-uniform'),
                'model__estimator__learning_rate': (1e-3, 1e0, 'log-uniform'),
                'model__estimator__l2_regularization': (1e-3,1e3, 'log-uniform'),
                'model__estimator__max_iter': (10,50),
                'model__estimator__max_depth': (2,4),}
        elif model_name=='xgb':
            from xgboost import XGBClassifier
            from mymodels import MonotonicWrapper
            model = MonotonicWrapper(XGBClassifier(
                    random_state=random_state+(cvi+1)*2),
                    class_weight='balanced')
            hparams = {
                'model__pvalue_cutoff': (0.1, 0.5, 'log-uniform'),
                'model__estimator__eta': (0.,1.),
                'model__estimator__gamma': (1e-3,1e3, 'log-uniform'),
                'model__estimator__max_depth': (2,4),
                'model__estimator__min_child_weight': (1e-2, 1e2, 'log-uniform'),
                'model__estimator__subsample': (0.5,1.),}
        elif model_name=='rf':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                    random_state=random_state+(cvi+1)*2,
                    class_weight='balanced')
            hparams = {
                'model__n_estimators': (10,200),
                'model__max_depth': (1,5),
                'model__min_samples_leaf': (1,50),
                'model__ccp_alpha': (1e-3, 1e3, 'log-uniform'), }
        else:
            raise NotImplementedError(f'Unknown model name {model_name}')
        scorer_ = 'roc_auc'

        # select features
        # add standardization
        model = Pipeline([
            #('feature_selection', GenericUnivariateSelect(score_func=mannwhitneyu_score, mode='fpr', param=0.1)),
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
                n_iter=50, scoring=scorer_, n_jobs=n_jobs, n_points=8,
                cv=Ncv, random_state=random_state+(cvi+1)*2+1,
                verbose=0)

        # fit
        model.fit(dftr[Xnames].values, dftr[Yname].values)
        if cvi<Ncv:
            hparams_cv.append(model.best_params_)
            model = model.best_estimator_

        # calibration
        model = CalibratedClassifierCV(model, cv='prefit')
        model.fit(dftr[Xnames].values, dftr[Yname].values)

        if cvi==Ncv:  # final fit
            ypp_final = model.predict_proba(dftr[Xnames].values)[:,1]
        else:
            models_cv.append(model)
            dfte = df[df.CV==cvi].reset_index(drop=True)
            ypp_cv[df.CV==cvi] = model.predict_proba(dfte[Xnames].values)[:,1]

    return models_cv, model, ypp_cv, ypp_final


def get_classification_perf(y, yp, nbt=1000, verbose=True, random_state=None):
    """
    performance with CI
    """
    if random_state is None:
        random_state = np.random.randn(0,10000)
    np.random.seed(random_state)

    y = np.array(y).astype(int)
    assert set(y)==set([0,1]), 'y must be binary, 0 or 1'
    aurocs = []
    auprcs = []
    fpr_curves = []
    tpr_curves = []
    tt_roc_curves = []
    pre_curves = []
    rec_curves = []
    tt_prc_curves = []
    op_point_methods = ['min-distance-to-perfect', 'Youden', 'sens80', 'sens90', 'spec80', 'spec90']
    accs = {x:[] for x in op_point_methods}
    f1s = {x:[] for x in op_point_methods}
    mccs = {x:[] for x in op_point_methods}
    sens = {x:[] for x in op_point_methods}
    spes = {x:[] for x in op_point_methods}
    ppvs = {x:[] for x in op_point_methods}
    npvs = {x:[] for x in op_point_methods}
    cms = {}
    for bti in tqdm(range(nbt+1), disable=not verbose):
        try:
            if bti==0:
                ybt = y
                ypbt = yp
            else:
                btids = np.random.choice(len(y),len(y),replace=True)
                ybt = y[btids]
                ypbt = yp[btids]

            aurocs.append( roc_auc_score(ybt, ypbt) )
            auprcs.append( average_precision_score(ybt, ypbt) )

            fpr, tpr, tt1 = roc_curve(ybt, ypbt)
            fpr_curves.append(fpr); tpr_curves.append(tpr); tt_roc_curves.append(tt1)

            pre, rec, tt2 = precision_recall_curve(ybt, ypbt)
            pre_curves.append(pre); rec_curves.append(rec); tt_prc_curves.append(tt2)

            if bti==0:
                thres = {
                    'min-distance-to-perfect': tt1[np.argmin(fpr**2+(1-tpr)**2)],
                    'Youden': tt1[np.argmax(tpr-fpr)],
                    'sens80': tt1[min([i for i, x in enumerate(tpr) if x>0.8], key=lambda i: tpr[i]-0.8)],
                    'sens90': tt1[min([i for i, x in enumerate(tpr) if x>0.9], key=lambda i: tpr[i]-0.9)],
                    'spec80': tt1[min([i for i, x in enumerate(1-fpr) if x>0.8], key=lambda i: 1-fpr[i]-0.8)],
                    'spec90': tt1[min([i for i, x in enumerate(1-fpr) if x>0.9], key=lambda i: 1-fpr[i]-0.9)],
                    }
            for op_point in op_point_methods:
                ypb = (ypbt>=thres[op_point]).astype(int)
                tp = np.sum((ybt==1)&(ypb==1))
                fn = np.sum((ybt==1)&(ypb==0))
                fp = np.sum((ybt==0)&(ypb==1))
                tn = np.sum((ybt==0)&(ypb==0))
                p = tp+fn; n = fp+tn
                pp = tp+fp; pn = fn+tn
                N = p+n
                if bti==0:
                    cms[op_point] = np.array([[tn,fp],[fn,tp]])#confusion_matrix(ybt, ypb)
                accs[op_point].append( (tp+tn)/N )
                f1s[op_point].append( 2*tp/(2*tp+fp+fn) )#f1_score(ybt, ypb)
                mccs[op_point].append( matthews_corrcoef(ybt, ypb) )
                sens[op_point].append( tp/p )#tpr[op_point]
                spes[op_point].append( tn/n )#1-fpr[op_point]
                ppvs[op_point].append( tp/pp )#precision_score(ybt,ypb)
                npvs[op_point].append( tn/pn )
        except Exception as ee:
            continue
    
    index = ['AUROC', 'AUPRC']
    perf_data = [
        [aurocs[0], np.percentile(aurocs[1:], 2.5), np.percentile(aurocs[1:], 97.5)],
        [auprcs[0], np.percentile(auprcs[1:], 2.5), np.percentile(auprcs[1:], 97.5)], ]
    for op_point in op_point_methods:
        for m, mn in zip(['accs', 'f1s', 'mccs', 'sens', 'spes', 'ppvs', 'npvs'], ['Accuracy', 'F1', 'MCC', 'Sensitivity', 'Specificity', 'PPV', 'NPV']):
            x = eval(f'{m}[op_point]')
            perf_data.append( [x[0], np.percentile(x[1:], 2.5), np.percentile(x[1:], 97.5)] )
            index.append(f'{op_point}:{mn}')
    perf = pd.DataFrame(data=np.array(perf_data), columns=['Val', 'LB', 'UB'], index=index)

    fpr_bt = np.sort(np.unique(np.concatenate(fpr_curves[1:]).round(3)))
    tpr_curves_bt = []
    for i in range(1,len(fpr_curves)):
        f = interp1d(fpr_curves[i], tpr_curves[i], kind='linear', bounds_error=False)
        tpr_curves_bt.append(f(fpr_bt))
    tpr_bt_ci = np.nanpercentile(np.array(tpr_curves_bt), (2.5, 97.5), axis=0)

    pre_bt = np.sort(np.unique(np.concatenate(pre_curves[1:]).round(3)))
    rec_curves_bt = []
    for i in range(1,len(pre_curves)):
        f = interp1d(pre_curves[i], rec_curves[i], kind='linear', bounds_error=False)
        rec_curves_bt.append(f(pre_bt))
    rec_bt_ci = np.nanpercentile(np.array(rec_curves_bt), (2.5, 97.5), axis=0)

    return (perf, cms,
            fpr_curves[0], tpr_curves[0], tt_roc_curves[0], fpr_bt, tpr_bt_ci,
            pre_curves[0], rec_curves[0], tt_prc_curves[0], pre_bt, rec_bt_ci, thres)


def main():
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

    # get cross validation fold
    random_state = 2023
    Ncv = 10
    cv_path = f'CV_{Ncv}fold_N{len(df)}_seed{random_state}.csv'
    df2 = pd.read_csv(cv_path)
    df = df.merge(df2, on=sid_name, how='inner', validate='1:1')

    # Table 3 in Katz et al. T‐MoCA. ADDADM. 2021
    # <=cutoff as positive
    y_cutoffs = {
        #'Youden':17,  # Youden's index
        'sens80':18,  # sensitivity>=80%
        #'spec80':15,  # specificity>=80%
        }
    feature_setups = {
        'eeg+cov':x_cols+cov_cols,
        'eeg':x_cols,
        'cov':cov_cols,
        }
    model_names = ['rf', 'xgb', 'gbt', 'logreg']
    result_folder = f'classification_results{suffix}'
    os.makedirs(result_folder, exist_ok=True)
    n_jobs = 8
    yname = 'Y'

    for cutoff_name, cutoff in y_cutoffs.items():
        print(cutoff_name)
        df[yname] = (df[yname_]<=cutoff).astype(int)

        iters = list(product(model_names, feature_setups.keys()))
        for ii, (model_name, fs) in enumerate(tqdm(iters)):
            models_cv, model, yp_cv, yp_final = train_classifier(
                model_name, df, feature_setups[fs], yname,
                random_state=random_state+ii, n_jobs=n_jobs)
            perf_cv, cm_cv, \
            fpr_cv, tpr_cv, tt_roc_cv, fpr_bt_cv, tpr_ci_cv, \
            pre_cv, rec_cv, tt_prc_cv, pre_bt_cv, rec_ci_cv, thres = get_classification_perf(df[yname], yp_cv, random_state=random_state)
            print(model_name, fs)
            print(perf_cv)

            # save
            save_path = os.path.join(result_folder, f'result_cutoff_{cutoff_name}_{model_name}_{fs}.pickle')
            with open(save_path, 'wb') as ff:
                pickle.dump({'models_cv':models_cv, 'model':model, 'Xnames':feature_setups[fs], 'thres':thres,
                    'yp_cv':yp_cv, 'yp_final':yp_final, 'cm_cv':cm_cv,
                    'fpr_cv':fpr_cv, 'tpr_cv':tpr_cv, 'tt_roc_cv':tt_roc_cv, 'fpr_bt_cv':fpr_bt_cv, 'tpr_ci_cv':tpr_ci_cv,
                    'pre_cv':pre_cv, 'rec_cv':rec_cv, 'tt_prc_cv':tt_prc_cv, 'pre_bt_cv':pre_bt_cv, 'rec_ci_cv':rec_ci_cv,
                    }, ff)

            save_path = os.path.join(result_folder, f'perf_cutoff_{cutoff_name}_{model_name}_{fs}.csv')
            perf_cv.to_csv(save_path)
            
            if model_name=='logreg':
                save_path = os.path.join(result_folder, f'coef_{model_name}_{fs}.csv')
                df_coef = pd.DataFrame(data={'Name':feature_setups[fs], 'Coef':np.zeros(len(feature_setups[fs]))})
                df_coef.loc[:, 'Coef'] = model.base_estimator.steps[-1][-1].coef_.flatten()
                df_coef = df_coef.sort_values('Coef', ascending=False, ignore_index=True)
                df_coef.to_csv(save_path, index=False)


if __name__=='__main__':
    main()

