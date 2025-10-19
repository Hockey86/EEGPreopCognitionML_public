from itertools import combinations
import numpy as np
from scipy.stats.mstats import mannwhitneyu, spearmanr
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import compute_class_weight


def mannwhitneyu_score(x, y):
    # x.shape = (N, L)
    # y.shape = (N,)
    return mannwhitneyu(x[y==0], x[y==1])


def spearmanr_score(x, y):
    # x.shape = (N, L)
    # y.shape = (N,)
    stats = []
    pvals = []
    for i in range(x.shape[1]):
        s, p = spearmanr(x[:,i], y)
        stats.append(s)
        pvals.append(p)
    return (np.array(stats), np.array(pvals))


def balanced_score(y, yp, method='rmse'):
    # Baccianella et al. Evaluation measures for ordinal regression. 2009.
    unique_y = np.sort(np.unique(y))
    sw = np.zeros(len(y))
    for uy in unique_y:
        sw[y==uy] = 1/np.sum(y==uy)
    if method=='rmse':
        return np.sqrt(np.sum((y-yp)**2*sw)/np.sum(sw))
    elif method=='mae':
        return np.sum(np.abs(y-yp)*sw)/np.sum(sw)
    else:
        raise NotImplementedError(method)


def cindex_score(y, yp):
    c1 = 0; c2 = 0
    for i,j in combinations(range(len(y)),2):
        c2 += 1
        if y[i]<y[j] and yp[i]<yp[j]:
            c1 += 1
        elif y[i]>y[j] and yp[i]>yp[j]:
            c1 += 1
        elif y[i]==y[j] and yp[i]==yp[j]:
            c1 += 1
    return c1/c2


class MonotonicL2LogisticRegression(LogisticRegression):
    """
    Impose monotonicity constraint based on univariate direction to avoid collinearity
    """
    def __init__(self, tol=0.0001, C=1.0, class_weight=None, random_state=None, max_iter=10000, n_jobs=None, pvalue_cutoff=0):
        super().__init__(penalty='l2', tol=tol, C=C, class_weight=class_weight, random_state=random_state, solver='lbfgs', max_iter=max_iter, n_jobs=n_jobs, l1_ratio=None)
        self.pvalue_cutoff = pvalue_cutoff

    def fit(self, X, y, sample_weight=None):
        # scikit-learn==1.0.2
        from sklearn.linear_model._logistic import _logistic_loss_and_grad
        self.classes_ = np.array([0,1])
        if sample_weight is None:
            sample_weight = np.ones(len(X))
        if self.class_weight=="balanced":
            cw = compute_class_weight(self.class_weight, classes=[0,1], y=y)
            sample_weight *= cw[y]

        y2 = np.array(y).astype(float)
        y2[y2==0] = -1
        #X2 = np.c_[X, np.ones(len(X))]
        w0 = np.zeros(X.shape[1]+1)

        bounds = []
        for xi in range(X.shape[1]):
            x0 = X[:,xi][y==0]
            x1 = X[:,xi][y==1]
            pval = mannwhitneyu(x0, x1).pvalue
            if pval<self.pvalue_cutoff:
                if np.median(x0)<np.median(x1):
                    bounds.append((0,None))
                else:
                    bounds.append((None,0))
            else:
                bounds.append((None,None))
        bounds.append((None,None)) # intercept

        self.opt_res = minimize(
            _logistic_loss_and_grad,
            w0, method="L-BFGS-B", jac=True,
            args=(X, y2, 1.0 / self.C, sample_weight),
            bounds=bounds,
            options={"gtol": self.tol, "maxiter": self.max_iter},
        )

        self.coef_ = self.opt_res.x[:-1].reshape(1,-1)
        self.intercept_ = self.opt_res.x[-1]
        self.fitted_ = True

        return self


# from https://github.com/bdsp-core/VE-CAM-S/blob/main/step2_fit_model_delirium.py
class LTRPairwise(BaseEstimator, ClassifierMixin):
    """Learning to rank, pairwise approach
    For each pair A and B, learn a score so that A>B or A<B based on the ordering.

    Parameters
    ----------
    estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface.
        It must be a classifier with a ``decision_function`` function.
    verbose : bool, optional, defaults to False
        Whether prints more information.
    """
    def __init__(self, estimator, class_weight=None, min_level_diff=1, verbose=False):
        super().__init__()
        self.estimator = estimator
        self.class_weight = class_weight
        self.min_level_diff = min_level_diff
        self.verbose = verbose
        
    #def __setattr__(self, name, value):
    #    setattr(self.estimator, name, value)
    #    super().__setattr__(name, value)
        
    def _generate_pairs(self, X, y, sample_weight):
        X2 = []
        y2 = []
        sw2 = []
        for i, j in combinations(range(len(X)), 2):
            # if there is a tie, ignore it
            if np.abs(y[i]-y[j])<self.min_level_diff:
                continue
            X2.append( X[i]-X[j] )
            y2.append( 1 if y[i]>y[j] else 0 )
            if sample_weight is not None:
                sw2.append( max(sample_weight[i], sample_weight[j]) )
        
        if sample_weight is None:
            sw2 = None
        else:
            sw2 = np.array(sw2)

        return np.array(X2), np.array(y2), sw2

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.sort(np.unique(y))
        if sample_weight is None:
            if self.class_weight is not None:
                #sample_weight = get_sample_weights(y, class_weight=self.class_weight, prior_count=2)
                sample_weight = np.zeros(len(X))
                for uy in self.classes_:
                    sample_weight[y==uy] = 1./np.sum(y==uy)
                sample_weight /= sample_weight.mean()
            else:
                sample_weight = np.ones(len(X))
        #sample_weight /= (np.mean(sample_weight)*len(X))
        
        # generate pairs
        X2, y2, sw2 = self._generate_pairs(X, y, sample_weight)
        sw2 = sw2/sw2.mean()
        if self.verbose:
            print('Generated %d pairs from %d samples'%(len(X2), len(X)))

        # fit the model
        self.estimator.fit(X2, y2, sample_weight=sw2)
        # get AIC and BIC
        yp = self.estimator.predict_proba(X2)[:,1]
        ll = -log_loss(y2, yp, normalize=False, sample_weight=sw2)
        k = (np.abs(np.r_[self.estimator.coef_.flatten(), self.estimator.intercept_.flatten()])>1e-4).sum()
        self.aic = 2*k-2*ll
        self.bic = k*np.log(len(X2))-2*ll

        # get the mean of z for each level of y
        z = self.predict_z(X)
        for tol in range(0,len(self.classes_)//2):
            z_means = np.array([z[(y>=cl-tol)&(y<=cl+tol)].mean() for cl in self.classes_])
            if tol==0:
                self.z_means = z_means
            if np.all(np.diff(z_means)>0):
                self.z_means = z_means
                break

        self.coef_ = self.estimator.coef_
        self.intercept_ = np.zeros(1)#self.estimator.intercept_
        return self

    def predict_z(self, X):
        z = self.estimator.decision_function(X)
        return z

    def decision_function(self, X):
        z = self.predict_z(X)
        return z

    def predict_proba(self, X, z=None):
        if z is None:
            z = self.predict_z(X)
        dists = -(z.reshape(-1,1) - self.z_means)**2
        dists[np.isnan(dists)] = -np.inf
        yp = softmax(dists, axis=1)
        return yp

    def predict(self, X):
        #yp1d = self.predict_z(X)
        yp = self.predict_proba(X)
        yp1d = self.classes_[np.argmax(yp, axis=1)]
        return yp1d


class IncrementalClassifier(BaseEstimator, ClassifierMixin):
    """
    Encodes labels into [0,0,0], [1,0,0], [1,1,0], [1,1,1], ...
    and treat as multiple classification problems.
    Ref: Cheng et al. A neural network approach to ordinal regression. 2008.
    """
    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator

    def fit(self, X, y, sample_weight=None):
        self.le = LabelEncoder().fit(y)
        self.classes_ = self.le.classes_
        y2 = self.le.transform(y)
        y3 = np.zeros((len(X),len(self.classes_)-1), dtype=int)
        for i in range(len(X)):
            y3[i, :y2[i]] = 1

        self.estimators = []
        self.thres = []
        for j in range(len(self.classes_)-1):
            model = clone(self.estimator)
            model.fit(X, y3[:,j], sample_weight=sample_weight)

            # calibration
            model = CalibratedClassifierCV(model, cv='prefit')
            model.fit(X, y3[:,j], sample_weight=sample_weight)

            fpr, tpr, tt = roc_curve(y3[:,j], model.predict_proba(X)[:,1])
            self.thres.append(tt[np.argmax(tpr-fpr)])  # Youden

            self.estimators.append(model)

        return self

    def predict(self, X):
        yp = []
        for j in range(len(self.classes_)-1):
            ypp = self.estimators[j].predict_proba(X)[:,1]
            yp_ = (ypp>=self.thres[j]).astype(int)
            yp.append(yp_)
        yp = np.array(yp).sum(axis=0)
        yp = self.le.inverse_transform(yp)
        return yp
        

class MonotonicWrapper(BaseEstimator, ClassifierMixin):
    """
    Impose monotonicity constraint based on univariate direction to avoid collinearity
    """
    def __init__(self, estimator, pvalue_cutoff=1, class_weight=None):
        self.estimator = estimator
        self.pvalue_cutoff = pvalue_cutoff
        self.class_weight = class_weight

    def predict(self, X):
        return self.estimator.predict(X)
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.sort(np.unique(y))
        bounds = []
        for xi in range(X.shape[1]):
            x0 = X[:,xi][y==0]
            x1 = X[:,xi][y==1]
            if len(x0)<=5 or len(x1)<=5:
                bounds.append(0)
                continue
            pval = mannwhitneyu(x0, x1).pvalue
            if pval<self.pvalue_cutoff:
                if np.median(x0)<np.median(x1):
                    bounds.append(1)
                else:
                    bounds.append(-1)
            else:
                bounds.append(0)
        self.estimator.monotonic_cst = np.array(bounds)
        self.estimator.monotone_constraints = tuple(bounds)

        if sample_weight is None:
            sample_weight = np.ones(len(X))
        if self.class_weight == 'balanced':
            cw = compute_class_weight(self.class_weight, classes=[0,1], y=y)
            cw = cw[y.astype(int)]
            sample_weight *= cw

        self.estimator.fit(X, y, sample_weight=sample_weight)
        return self

