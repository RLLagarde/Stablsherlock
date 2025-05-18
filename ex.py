# import numpy as np
# import pandas as pd
# import joblib
# import os
# from pathlib import Path
# from sklearn.linear_model import Lasso
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.feature_selection import VarianceThreshold

# from stabl.stabl import Stabl, export_stabl_to_csv, plot_stabl_path, plot_fdr_graph

# # Charger les données
# X = pd.read_csv("./Sample_Data/data/Biobank_SSI/CyTOF.csv").drop(columns=['sampleID'])
# original_feature_names = X.columns.tolist()
# y = pd.read_csv("./Sample_Data/data/Biobank_SSI/outcome.csv")['model1b']

# # 1. Pré-traitement
# preproc = Pipeline([
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())
# ])
# X_proc = preproc.fit_transform(X)


# # 2. Configuration Lasso + Stabl
# artificial_type = "random_permutation"  # Alternative: "knockoff" "random_permutation"

# for i in range(949, 1000):
#     random = i
#     lasso = Lasso(
#         max_iter=int(1e4),  # Augmentation nécessaire pour convergence
#         random_state=42
#     )

#     stabl = Stabl(
#         base_estimator=lasso,
#         n_bootstraps=2000,  
#         artificial_type=artificial_type,
#         artificial_proportion=1.0,
#         replace=False,
#         fdr_threshold_range=np.arange(0.1, 1, 0.01), 
#         sample_fraction=0.5,
#         random_state=random,
#         lambda_grid={"alpha": np.logspace(-2, 2, 10)},  # Grid log pour alpha #previous -4 -> 1
#         verbose=1  
#     )

    
#     stabl.fit(X_proc, y)

   
#     os.makedirs(f"./lasso_5{random}", exist_ok=True)


#     selected_features = stabl.get_feature_names_out(input_features=original_feature_names)
#     pd.DataFrame({
#         "Features": selected_features,
#         "Max_Stability_Score": stabl.stabl_scores_.max(axis=1)[stabl.get_support()]
#     }).to_csv(f"./lasso_5{random}/selected_features_with_scores.csv")

  
#     export_stabl_to_csv(stabl, path=f"./lasso_5{random}")

   
#     plot_stabl_path(stabl, export_file=True, path=f"./lasso_5{random}/stability_path.pdf")
#     plot_fdr_graph(stabl, export_file=True, path=f"./lasso_5{random}/fdr_curve.pdf")

   
#     joblib.dump(preproc, f"./lasso_5{random}/preprocessor.joblib")

#     print("Analyse terminée avec succès!")
#     print(f"Features sélectionnées: {len(selected_features)}")
#     print(f"Dossier des résultats: {os.path.abspath(f'./lasso_5{random}')}")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare deux pipelines STABL :

baseline      : models/{seed}
randomization : rerandomizations10/models/{seed}

• Compte dans combien de seeds chaque feature est sélectionnée
• Agrège Max STABL scores (mean & std bootstrap counts)
• Mappe x.nnn -> vrai nom via DATA_FILE
• Exporte trois CSV et un bar-plot groupé.
"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import warnings, os, re

# -------------------- CONFIG --------------------
N_SEEDS        = 1000
N_BOOTSTRAPS   = 2000

DATA_FILE      = "./Sample_Data/data/Biobank_SSI/CyTOF.csv"     # ← ajuste si besoin
FILE_SCORE     = "Max STABL scores.csv"
FILE_SELECTED  = "selected_features_with_scores.csv"

BASELINE_DIR   = "baseline_cytof/{}"
RANDOM_DIR     = "rerandomization5/lasso_5{}"

# ----------------- MAP x.nnn -> vrai nom -----------------
raw = pd.read_csv(DATA_FILE)
if raw.columns[0].lower().startswith("sample"):
    raw = raw.drop(columns=raw.columns[0])          # drop sampleID

real_names = raw.columns.tolist()                   # index 0 ↔ vrai nom 0

def translate(key: str) -> str:
    """Convertit x.nnn -> vrai nom ; laisse les autres intactes."""
    m = re.fullmatch(r"x\.(\d+)", str(key))
    if m:
        idx = int(m[1]) - 1
        if idx < len(real_names):
            return real_names[idx]
    return str(key)

# ----------------- OUTILS GÉNÉRIQUES -----------------
def folder(version: str, seed: int) -> Path:
    return Path(BASELINE_DIR.format(seed) if version == "baseline"
                else RANDOM_DIR.format(seed))

def count_per_seed(version: str) -> Counter:
    """+1  pour chaque seed où la feature apparaît dans selected_features."""
    cnt = Counter()
    for s in range(N_SEEDS):
        f = folder(version, s) / FILE_SELECTED
        if not f.exists():
            warnings.warn(f"[{version}] seed{s} : {f} manquant")
            continue
        feats = pd.read_csv(f)["Features"].map(translate)
        cnt.update(feats.unique())           # +1 par feature, pas par ligne
    return cnt

def max_prob_matrix(version: str) -> pd.DataFrame:
    """Matrix features × seeds des Max Proba (remplie de 0 si fichier absent)."""
    cols = []
    for s in range(N_SEEDS):
        f = folder(version, s) / FILE_SCORE
        if f.exists():
            df = pd.read_csv(f, index_col=0)
            df.index = df.index.map(translate)
            df = df.groupby(df.index).max()        # fusion x.nnn + nom
            cols.append(df["Max Proba"].rename(f"seed_{s}"))
        else:
            warnings.warn(f"[{version}] seed{s} : {f} manquant")
    if not cols:
        raise RuntimeError(f"Aucun {FILE_SCORE} pour {version}")
    return pd.concat(cols, axis=1).fillna(0.0)

# ----------------- PIPELINE -----------------
print("→ Comptage des features sélectionnées …")
cnt_base = count_per_seed("baseline")
cnt_rand = count_per_seed("random")

# union de toutes les features rencontrées
all_feats = sorted(set(cnt_base) | set(cnt_rand), key=str)

df_counts = pd.DataFrame({
    "feature":           all_feats,
    "n_seeds_baseline":  [cnt_base.get(f, 0) for f in all_feats],
    "n_seeds_random":    [cnt_rand.get(f, 0) for f in all_feats],
})
df_counts.to_csv("features_count_baseline_vs_random.csv", index=False)
print("✓ CSV features_count_baseline_vs_random.csv")

print("→ Agrégation des Max STABL scores …")
stats_base = max_prob_matrix("baseline") * N_BOOTSTRAPS
stats_rand = max_prob_matrix("random")   * N_BOOTSTRAPS

def summary(mat: pd.DataFrame, outfile: str):
    (pd.DataFrame({"feature": mat.index,
                   "mean_count": mat.mean(axis=1),
                   "std_count":  mat.std(axis=1, ddof=0)})
       .sort_values("mean_count", ascending=False)
       .to_csv(outfile, index=False))
    print("✓", outfile)

summary(stats_base, "selection_stats_baseline.csv")
summary(stats_rand, "selection_stats_random.csv")

# ----------------- PLOT BAR GROUPÉ -----------------
os.makedirs("plots", exist_ok=True)
x = np.arange(len(all_feats)); width = .4
plt.figure(figsize=(max(8, len(all_feats)*0.25), 6))
plt.bar(x - width/2, df_counts["n_seeds_random"],   width, color="green", label="randomization")
plt.bar(x + width/2, df_counts["n_seeds_baseline"], width, color="blue",  label="baseline")
plt.xticks(x, all_feats, rotation=90, fontsize=6)
plt.ylabel("Number of seeds (out of 1000)")
plt.title("Feature robustness across seeds\n(green = randomization, blue = baseline)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/bar_seed_counts_grouped.png")
# ---------- PLOT MEAN ± STD (en % de bootstraps) ----------
# 1. Moyenne & écart-type à partir des matrices brutes
mb_mean = stats_base.mean(axis=1)
mb_std  = stats_base.std(axis=1, ddof=0)
mr_mean = stats_rand.mean(axis=1)
mr_std  = stats_rand.std(axis=1, ddof=0)

# 2. Features VRAIMENT sélectionnées : intersection des deux compteurs
sel_feats = df_counts.loc[
    (df_counts["n_seeds_baseline"] > 0) & (df_counts["n_seeds_random"] > 0),
    "feature"
]

# 3. Ordre cohérent + conversion en %
mb_pct = (mb_mean[sel_feats] / (N_BOOTSTRAPS / 100)).fillna(0)
mr_pct = (mr_mean[sel_feats] / (N_BOOTSTRAPS / 100)).fillna(0)
mb_err = (mb_std [sel_feats] / (N_BOOTSTRAPS / 100)).fillna(0)
mr_err = (mr_std [sel_feats] / (N_BOOTSTRAPS / 100)).fillna(0)

x, width = np.arange(len(sel_feats)), .4
plt.figure(figsize=(max(8, len(sel_feats)*0.25), 6))
plt.bar(x - width/2, mr_pct, width, yerr=mr_err,
        color="green", label="randomization", capsize=3)
plt.bar(x + width/2, mb_pct, width, yerr=mb_err,
        color="blue",  label="baseline",      capsize=3)
plt.xticks(x, sel_feats, rotation=90, fontsize=6)
plt.ylabel("Mean bootstrap selections (%)")
plt.title("Mean ± std of bootstrap selections (features selected in BOTH pipelines)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/bar_mean_std_grouped.png")
plt.close()
print("✓ plot → plots/bar_mean_std_grouped.png")
