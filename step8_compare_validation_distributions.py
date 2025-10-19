import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, brunnermunzel
from statsmodels.stats.proportion import test_proportions_2indep
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')


df_tr = pd.read_csv('../380_Combined_EEG_Clinical_rev2_equal_fs.csv')
df_va = pd.read_excel('dataset_external_validation.xlsx')
#df_va1 = pd.read_csv('../Validation_Dataset_98_EEGFeatures_v3_LP35_170Hz.csv')
#df_va2 = pd.read_excel('../TABLA_FONDEF_UCHILE_PUC_230802_231101.xlsx', sheet_name='011123', skiprows=1)
#df_va2 = df_va2[df_va2.Excluded!='si'].reset_index(drop=True)
#df_va = df_va1.merge(df_va2, on='ID', how='inner', validate='1:1')
print(f'N(tr) = {len(df_tr)}, N(va) = {len(df_va)}')

df_tr.loc[df_tr.Sex=='M', 'Sex'] = 1
df_tr.loc[df_tr.Sex=='F', 'Sex'] = 0
df_va.loc[df_va.Sex=='M', 'Sex'] = 1
df_va.loc[df_va.Sex=='F', 'Sex'] = 0

save_dir = 'hist_tr_va'
os.makedirs(save_dir, exist_ok=True)

df = pd.read_excel('dataset.xlsx', sheet_name='EEGFeatures')
feat_names_cont = list(df.columns)+[
    'Age', 'education',
    'Pre.op.T.MOCA',
    'delirium_sev1to3', 'hosp_los',]
feat_names_bin = ['Sex',
    'delirium1to3', 'readmission',
    'died30', 'died90', 'died180']
feat_names = feat_names_cont + feat_names_bin
is_cont = [True]*len(feat_names_cont) + [False]*len(feat_names_bin)

df_res = pd.DataFrame(data={'FeatureName':feat_names})
for fi, fn in enumerate(tqdm(feat_names)):
    print(fn)
    save_path = os.path.join(save_dir, f'hist_tr_va_{fn}.png')
    #if os.path.exists(save_path): continue
    if not (fn in df_tr.columns and fn in df_va.columns): continue

    xtr = df_tr[fn].values.astype(float)
    xtr = xtr[~np.isnan(xtr)]
    xva = df_va[fn].values.astype(float)
    xva = xva[~np.isnan(xva)]

    if is_cont[fi]:
        res = mannwhitneyu(xtr,xva)
        is_sig = res.pvalue<0.05
        df_res.loc[fi, 'MannWhitneyU_PVal'] = res.pvalue
        if is_sig:
            df_res.loc[fi,'MannWhitneyU_Direction'] = 'tr<va' if np.median(xtr)<np.median(xva) else 'tr>va'
        else:
            df_res.loc[fi,'MannWhitneyU_Direction'] = np.nan

        res = brunnermunzel(xtr,xva)
        is_sig = res.pvalue<0.05
        df_res.loc[fi, 'BrunnerMunzel_PVal'] = res.pvalue
        if is_sig:
            df_res.loc[fi,'BrunnerMunzel_Direction'] = 'tr<va' if np.median(xtr)<np.median(xva) else 'tr>va'
        else:
            df_res.loc[fi,'BrunnerMunzel_Direction'] = np.nan

    else:
        res = test_proportions_2indep(xtr.sum(), len(xtr), xva.sum(), len(xva))
        is_sig = res.pvalue<0.05
        df_res.loc[fi, 'BinaryProportion_PVal'] = res.pvalue
        if is_sig:
            df_res.loc[fi,'BinaryProportion_Direction'] = 'tr<va' if xtr.mean()<xva.mean() else 'tr>va'
        else:
            df_res.loc[fi,'BinaryProportion_Direction'] = np.nan

    if is_sig:
        print(f'Different: {fn}')

    bins = np.linspace(np.r_[xtr,xva].min(), np.r_[xtr,xva].max(), 30)

    plt.close()
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    ax.hist(xtr, bins=bins, color='r', alpha=0.4, label='Model development set')
    ax.hist(xva, bins=bins, color='b', alpha=0.4, label='External validation set')
    if is_sig:
        ax.text(0.1,0.9,'Significantly different',ha='left',va='top',transform=ax.transAxes)
    ax.legend(frameon=False)
    ax.set_ylabel('Count')
    ax.set_xlabel(fn)
    sns.despine()
    plt.tight_layout()
    #plt.show()
    plt.savefig(save_path)

print(df_res)
df_res.to_excel(os.path.join(save_dir, 'dist_comparison_result.xlsx'), index=False)

