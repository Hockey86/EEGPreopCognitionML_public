import os
import pickle
import numpy as np
import pandas as pd


def main():
    result_dir = 'ordinal_regression_results_cutoff15_few_cov'
    model_name = 'inc_clf_logreg'
    #model_name = 'inc_clf_rf'
    data_type = 'eeg'
    inc_clf_model_id = 2  # 0 is <=15 vs >15, 1 is <=16 vs >16, 2 is <=17 vs >17

    result_path = os.path.join(result_dir, f'result_{model_name}_{data_type}.pickle')
    with open(result_path, 'rb') as f:
        res = pickle.load(f)
    Xnames = res['Xnames']
    model = res['model']
    if model_name=='inc_clf_logreg':
        fi = model.named_steps['model'].estimators[inc_clf_model_id].base_estimator.coef_.flatten()
    elif model_name=='inc_clf_rf':
        fi = model.named_steps['model'].estimators[inc_clf_model_id].base_estimator.feature_importances_
    df = pd.DataFrame(data={'Name':Xnames, 'Importance':fi})
    df = df.sort_values('Importance', ignore_index=True, key=lambda x:np.abs(x), ascending=False)
    print(df)
    df.to_csv(os.path.join(result_dir, f'feature_importance_{model_name}_{data_type}.csv'), index=False)



if __name__=='__main__':
    main()

