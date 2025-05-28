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
from stabl.stabl import Stabl, export_stabl_to_csv, plot_stabl_path, plot_fdr_graph

# Charger les données
X = pd.read_csv("./Sample_Data/data/COVID-19/Training/Proteomics.csv").drop(columns=['sampleID'])
original_feature_names = X.columns.tolist()
y = pd.read_csv("./Sample_Data/data/COVID-19/Training/Mild&ModVsSevere.csv")['Mild&ModVsSevere']

alasso = ALogitLasso(penalty="l1", solver="liblinear", max_iter=int(1e6), class_weight='balanced', random_state=42)

# 1. Pré-traitement
preproc = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('var_thresh', VarianceThreshold(0.0000000001)),
    ('scaler', StandardScaler())
])
X_proc = preproc.fit_transform(X)


# 2. Configuration Lasso + Stabl
artificial_type = "random_permutation"  # Alternative: "knockoff" "random_permutation"

for i in range(28, 50):
    random = i
    lasso = Lasso(
        max_iter=int(1e4),  # Augmentation nécessaire pour convergence
        random_state=42
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
        lambda_grid={"C": np.logspace(-2, 2, 10)},  # Grid log pour alpha #previous -4 -> 1
        verbose=1,
        repetition=50

    )

    
    stabl.fit(X_proc, y)

   
    os.makedirs(f"models/{random}", exist_ok=True)
    mask = preproc.named_steps['var_thresh'].get_support() # since there's a variance thershold we need to filter the columns that have been deleted by the threshold
    filtered_feature_names = np.array(original_feature_names)[mask] #


    selected_features = stabl.get_feature_names_out(input_features=filtered_feature_names)
    pd.DataFrame({
        "Features": selected_features,
        "Max_Stability_Score": stabl.stabl_scores_.max(axis=1)[stabl.get_support()]
    }).to_csv(f"./models/{random}/selected_features_with_scores.csv")

  
    export_stabl_to_csv(stabl, path=f"./models/{random}")

   
    plot_stabl_path(stabl, export_file=True, path=f"./models/{random}/stability_path.pdf")
    plot_fdr_graph(stabl, export_file=True, path=f"./models/{random}/fdr_curve.pdf")

   
    joblib.dump(preproc, f"./models/{random}/preprocessor.joblib")

    print("Analyse terminée avec succès!")
    print(f"Features sélectionnées: {len(selected_features)}")
    print(f"Dossier des résultats: {os.path.abspath(f'./models/{random}')}")
