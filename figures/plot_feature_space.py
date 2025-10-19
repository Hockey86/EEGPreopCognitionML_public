import sys
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from umap import UMAP
from pacmap import PaCMAP
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 12})
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')


def main(display_type):
    with pd.ExcelFile('../dataset.xlsx') as ff:
        df1 = pd.read_excel(ff, sheet_name="Outcome")
        df2 = pd.read_excel(ff, sheet_name="Covariates")
        df3 = pd.read_excel(ff, sheet_name="EEGFeatures")
    df = pd.concat([df1, df2, df3], axis=1)

    y = df.iloc[:,1].values
    X = df.iloc[:,2:].values
    X = (X-X.mean(axis=0))/X.std(axis=0)
    print(X.shape)

    ymin = y.min()
    ymax = y.max()
    
    random_state = 2023
    methods = ['umap-sup']#'tsne', 'umap', 'pacmap']
    for method in methods:
        print(method)
        if method=='tsne':
            model = TSNE(random_state=random_state)
            Xvis = model.fit_transform(X)
        elif method=='umap':
            model = UMAP(random_state=random_state)
            Xvis = model.fit_transform(X)
        elif method=='umap-sup':
            model = UMAP(random_state=random_state)
            Xvis = model.fit_transform(X, y=y)
        elif method=='pacmap':
            model = PaCMAP(random_state=random_state)
            Xvis = model.fit_transform(X)

        plt.close()
        fig = plt.figure(figsize=(6*1.4,5*1.4))

        ax = fig.add_subplot(111)
        ax.scatter(Xvis[:,0], Xvis[:,1], s=20, alpha=0.7, cmap='turbo', c=(y-ymin)/(ymax-ymin))
        ax.axis('off')
        
        plt.tight_layout()
        if display_type=='show':
            plt.show()
        else:
            plt.savefig(f'feature_space_vis_{method}.{display_type}', bbox_inches='tight', pad_inches=0.05)
        

if __name__=='__main__':
    display_type = sys.argv[1]
    main(display_type)

