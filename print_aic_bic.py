import os
import pickle


data_types = ['eeg+cov', 'cov', 'eeg']
model = 'ltr_pair'
result_folder = 'ordinal_regression_results_cutoff15_few_cov'

for dt in data_types:
    print(dt)
    with open(os.path.join(result_folder, f'result_{model}_{dt}.pickle'),'rb') as ff:
        aa=pickle.load(ff)
    print('AIC = {}'.format(aa['model'].steps[-1][-1].aic))
    print('BIC = {}'.format(aa['model'].steps[-1][-1].bic))
    print()
