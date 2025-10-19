import os, glob, pickle, sys
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 17})
import seaborn as sns
sns.set_style('ticks')
sys.path.insert(0, '..')


def plot_cf(y, yp, title=None, perc=False, save_path=None):
    cf = confusion_matrix(y-15,yp-15)
    if perc:
        cf = cf*100./cf.sum(axis=1, keepdims=True)
    cf2 = pd.DataFrame(data=cf,
       index=['≤15','16','17','18','19','20','21','22'],
       columns=['≤15','16','17','18','19','20','21','22'])

    plt.close()
    fig = plt.figure(figsize=(6.3,6.0))
    ax = fig.add_subplot(111)
    sns.heatmap(cf2, annot=True, linewidth=.5,cmap='Blues', cbar=False)#cbar_kws={'ticks': [0,5,10,15,20,22]})
    ax.set(xlabel='Predicted TMoCA',ylabel='Actual TMoCA')
    #ax.set_title(title)
    ax.xaxis.tick_top()
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


def main():
    model = 'inc_clf_rf'#logreg'
    input_data = 'eeg+cov'
    title = None#'Logistic Regression Model'
    suffix = ''#_perc'

    with open(f'../ordinal_regression_results_cutoff15_few_cov/result_{model}_{input_data}.pickle','rb') as ff:
        res = pickle.load(ff)
    df = pd.read_excel('../dataset.xlsx')
    y = df['Pre.op.T.MOCA'].values
    y[y<=15]=15
    yp = res['yp_cv'].astype(int)
    plot_cf(y, yp, title=title, perc=suffix=='_perc', save_path=f'ordinal_result_confusion_matrix-{model}_{input_data}{suffix}.png')
    """
    output_dir = '../external_validation'
    file_paths = glob.glob('../external_validation/external_pred_*.csv')
    for fp in tqdm(file_paths):
        df = pd.read_csv(fp)
        y = df['Pre.op.T.MOCA'].values.astype(int)
        y[y<=15] = 15
        yp = df['Pred'].values.astype(int)
        save_path = os.path.join(output_dir, os.path.basename(fp).replace('external_pred','confusion_matrix').replace('.csv','.png'))
        plot_cf(y, yp, save_path=save_path)
    """


if __name__=='__main__':
    main()

