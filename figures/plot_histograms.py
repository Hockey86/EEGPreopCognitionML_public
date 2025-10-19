import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('pdf', fonttype=42)
plt.rcParams.update({'font.size': 14})
import seaborn as sns
sns.set_style('ticks')


def main(display_type):
    with pd.ExcelFile('../dataset.xlsx') as ff:
        df1 = pd.read_excel(ff, sheet_name="Outcome")
        df2 = pd.read_excel(ff, sheet_name="Covariates")
        df3 = pd.read_excel(ff, sheet_name="EEGFeatures")
    sid_name, outcome_name = df1.columns
    cov_names = list(df2.columns)
    eeg_names = list(df3.columns)
    df = pd.concat([df1, df2, df3], axis=1)

    bins = np.arange(0,22+1+1)-0.5

    plt.close()
    fig = plt.figure(figsize=(7,4.5))

    ax = fig.add_subplot(111)
    ns, _, _ = ax.hist(df[outcome_name], bins=bins, rwidth=0.9, color='k', alpha=0.5)
    for x in np.arange(0,22+1):
        if df[outcome_name].min()<=x<=df[outcome_name].max():
            ax.text(x, ns[x]+0.5, str(int(ns[x])), ha='center', va='bottom', color='k')
    ax.set_xticks(np.arange(df[outcome_name].min(), df[outcome_name].max()+1))
    ax.set_xlabel('T-MoCA')
    ax.set_xlim(df[outcome_name].min()-1, df[outcome_name].max()+1)
    ax.set_ylabel('Count')
    sns.despine()

    plt.tight_layout()
    if display_type=='png':
        plt.savefig(f'hist_{outcome_name}.png', bbox_inches='tight')#, pad_inches=0.03, dpi=300)
    elif display_type in ['pdf', 'svg']:
        plt.savefig(f'hist_{outcome_name}.{display_type}', bbox_inches='tight')#, pad_inches=0.03)
    else:
        plt.show()


if __name__=='__main__':
    display_type = sys.argv[1]
    main(display_type)

