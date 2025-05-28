import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from stabl.adaptive import ALogitLasso
import argparse



from localStabl import Stabl,export_stabl_to_csv, plot_stabl_path, plot_fdr_graph

path = "models"

X = pd.read_csv("./Sample_Data/data/COVID-19/Training/Proteomics.csv").drop(columns=['sampleID'])
original_feature_names = X.columns.tolist() 
y = pd.read_csv("./Sample_Data/data/COVID-19/Training/Mild&ModVsSevere.csv")['Mild&ModVsSevere']



preproc = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('var_thresh', VarianceThreshold(0.0000000001)),
    ('scaler', StandardScaler())
])
X_proc = preproc.fit_transform(X)

alasso = ALogitLasso(penalty="l1", solver="liblinear", max_iter=int(1e6), class_weight='balanced', random_state=42)



artificial_type = "knockoff"  #or "knockoff" random_permutation

def run(i): 
    random = i
    lasso = Lasso(
        max_iter=int(1e4), 
        random_state=42 #this is not useful it changes nothing
    )
    

    stabl = Stabl(
        base_estimator=alasso,
        n_bootstraps=3000,  
        artificial_type=artificial_type,
        artificial_proportion=1.0,
        replace=False,
        fdr_threshold_range=np.arange(0.01, 1, 0.01), 
        sample_fraction=0.5,
        random_state=random,
        lambda_grid={"C": np.logspace(-2, 2, 10)},  # Grid log pour alpha #previous -4 -> 1 #when I did logspace from -4 to +1 it took 6 min/seed so i started again from the beginning and now it's 1min30/seed.
        verbose=1, 
        repetition=50
    )

    
    stabl.fit(X_proc, y)

   
    os.makedirs(f"./{path}/{random}", exist_ok=True)
    mask = preproc.named_steps['var_thresh'].get_support() # since there's a variance thershold we need to filter the columns that have been deleted by the threshold
    filtered_feature_names = np.array(original_feature_names)[mask] #

    # extracting the useful files. I copy pasted the main function from stabl.py file
    selected_features = stabl.get_feature_names_out(input_features=filtered_feature_names)
    pd.DataFrame({
        "Features": selected_features,
        "Max_Stability_Score": stabl.stabl_scores_.max(axis=1)[stabl.get_support()]
    }).to_csv(f"./{path}/{random}/selected_features_with_scores.csv") #these lines are the most relevant if we want to compare the frequency of selection.

  
    export_stabl_to_csv(stabl, path=f"./{path}/{random}")

   
    plot_stabl_path(stabl, export_file=True, path=f"./{path}/{random}/stability_path.pdf")
    plot_fdr_graph(stabl, export_file=True, path=f"./{path}/{random}/fdr_curve.pdf")

   
    

    print("Done")
    print(f"Number of features: {len(selected_features)}")
    print(f"results in: {os.path.abspath(f'./{path}/{random}')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("idx", type=int, default=0,nargs="?")
    args = parser.parse_args()
    run(args.idx)
