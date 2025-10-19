import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 8})
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')


def main(display_type):
    with pd.ExcelFile('../dataset.xlsx') as ff:
        df1 = pd.read_excel(ff, sheet_name="Outcome")
        df2 = pd.read_excel(ff, sheet_name="Covariates")
        df3 = pd.read_excel(ff, sheet_name="EEGFeatures")
    df = pd.concat([df1, df2, df3], axis=1)

    Xnames = ['Pre.op.T.MOCA',
        'Age', 'SexF', 'bmi', 'education', 'race_black', 'race_other', 'hispanic',#'marital', 'race_white', 'promis_global0',
        'tobacco', 'illicit_drugs',
        'htn', 'diabetes', 'sleep_apnea', 'stroke',
        'mi', 'peripheral_art_dx', 'afib', 'prev_cardiac_intervent',
        'chronic_lung_dx', 'renal_failure', 'liver_dx',
        'clDelta_Power_db', 'clTheta_Power_db', 'clAlpha_Power_db', 'clBeta_Power_db', 'Total_Power_db',
        'osAlpha_freq', 'osAlpha_Power', 'osAlpha_BW', 'osAlpha_Prev',
        'Offset', 'Slope', 'error', 'nPeaksTotalFound',# 'BadFits_Percent', 'r2'
        'LZc', 'PermEntropy', 'DispEntropy', 'h_complexity', 'num_zerocross', 'Higuchi_FD', 'DFA',#, 'h_mobility' # strong corr with h_complexity
        'Coh_Inter', 'Coh_Intra', 'wPLI_Inter', 'wPLI_Intra', 'MI_Inter', 'MI_Intra',
    ]
    Xnames_display = ['T-MoCA',
        'Age', 'SexF', 'BMI', 'Educ', 'RaceBlack', 'RaceOther', 'Hispanic',
        'Smoking', 'DrugAbuse', # 'PROMIS'
        'HTN', 'T2D', 'SA', 'CVA',
        'MI', 'PerArtDx', 'AF', 'CardiacInt',
        'ChrLungDx', 'RF', 'LiverDx',
        r'$\delta$ pow', r'$\theta$ pow', r'$\alpha$ pow', r'$\beta$ pow', 'TotalPow',
        r'$\alpha$ freq', r'$\alpha$ pow FOOF', r'$\alpha$ BW', r'$\alpha$ prev',
        'FOOFOffset', 'FOOFSlope', 'FOORErr', 'nPeak',# 'BadFOOFFit%', FOOFR2',
        'LZc', 'PermEntr', 'DispEntr', 'H-Comp', '0cross', 'HiguchiFD', 'DFA',#'H-Mobi', 
        'CohInter', 'CohIntra', 'PLIInter', 'PLIIntra', 'MIInter', 'MIIntra',
    ]
    X = df[Xnames].values

    # plot
    plt.close()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14*1.3, 7*1.3))
    
    # hierarchical clustering
    corr = spearmanr(X).correlation# based on Spearman's correlation
    corr_linkage = hierarchy.ward(np.abs(corr))# based on absolute value
    dendro = hierarchy.dendrogram( corr_linkage, ax=ax1, labels=Xnames_display)
    dendro_idx = np.arange(0, len(dendro['ivl']))
    ax1.set_ylabel('distance')
    ax1.set_yticks([])
    ax1.set_xticks(dendro_idx*10)
    ax1.set_xticklabels(dendro['ivl'], rotation=-55, ha='left')#, fontsize=16)
    sns.despine()
    
    # pairwise matrix plot
    sorted_corr = corr[dendro['leaves'], :][:, dendro['leaves']] # sort according to clustering
    im = ax2.imshow(sorted_corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro['ivl'], rotation=-55, ha='left')
    ax2.set_yticklabels(dendro['ivl'])
    cbar = plt.colorbar(mappable=im)
    cbar.ax.set_ylabel('Spearman\'s correlation')#, rotation=270)
    
    plt.tight_layout()
    #plt.subplots_adjust(wspace=0.45)
    if display_type=='show':
        plt.show()
    else:
        plt.savefig(f'colinear.{display_type}', bbox_inches='tight', pad_inches=0.05)
        

if __name__=='__main__':
    display_type = sys.argv[1]
    main(display_type)

