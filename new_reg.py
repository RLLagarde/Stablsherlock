

####### V3
######## V3


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Late-fusion OOF empilée (stacking) – stims 16-17-18
#   1. prédictions OOF par stim via 5×100 splits (500 folds)
#   2. méta-modèle logistique (GridSearch : L1 / L2 / ElasticNet + grille C)
#   3. exports : poids OOF (moyenne & médiane), intercept, prédictions OOF
# """

# from pathlib import Path
# import numpy as np, pandas as pd
# from sklearn.model_selection import (RepeatedStratifiedKFold, KFold,
#                                      RepeatedStratifiedKFold, GridSearchCV)
# from sklearn.pipeline        import Pipeline
# from sklearn.impute          import SimpleImputer
# from sklearn.preprocessing   import StandardScaler
# from sklearn.linear_model    import Lasso, LogisticRegression
# from sklearn.metrics         import roc_auc_score
# from scipy.stats             import pearsonr

# # ------------------------------------------------------------------
# ROOT      = Path("./files")
# STIMS     = ["16", "17", "18"]
# CSV_DATA  = "data4.csv"
# CSV_FEATS = "selectedFeats.csv"
# Y_FILE    = Path("./files/outcome4.csv")
# SEED      = 42
# # ------------------------------------------------------------------

# y_full = pd.read_csv(Y_FILE)["group"].astype(float).reset_index(drop=True)
# X_full = pd.read_csv(ROOT / CSV_DATA, index_col=0).reset_index(drop=True)

# PREP = Pipeline([("imp", SimpleImputer(strategy="median")),
#                  ("std", StandardScaler())])

# # ---------- 1ʳᵉ couche : OOF par stim (5×100) ----------
# cache   = {s: [] for s in STIMS}                   
# rskf    = RepeatedStratifiedKFold(n_splits=5, n_repeats=200,
#                                   random_state=SEED)

# for fold, (tr_idx, te_idx) in enumerate(rskf.split(X_full, y_full), 1):
#     y_tr = y_full.iloc[tr_idx]

#     for stim in STIMS:
#         feat_file = ROOT / stim / CSV_FEATS
#         if not feat_file.exists():
#             cache[stim].append((te_idx, np.full(len(te_idx), 0.5)))
#             continue

#         feats = pd.read_csv(feat_file, index_col=0).astype(bool)
#         feats = feats.columns[feats.any()].tolist()
#         if not feats:
#             cache[stim].append((te_idx, np.full(len(te_idx), 0.5)))
#             continue

#         X_tr, X_te = X_full.loc[tr_idx, feats], X_full.loc[te_idx, feats]
#         model = Pipeline([("prep", PREP),
#                           ("reg",  Lasso(alpha=0.01, max_iter=100_000))])
#         model.fit(X_tr, y_tr)
#         cache[stim].append((te_idx, model.predict(X_te)))

#     if fold % 50 == 0:
#         print(f"Fold {fold}/{rskf.get_n_splits()} terminé.")

# # Agrégation médiane des 500 prédictions
# oof_proba = {}
# for stim in STIMS:
#     n_folds = len(cache[stim])
#     stack   = np.full((n_folds, len(y_full)), np.nan)
#     for k, (idx, pred) in enumerate(cache[stim]):
#         stack[k, idx] = pred
#     oof_proba[stim] = np.nanmedian(stack, axis=0)

# P = np.column_stack([oof_proba[s] for s in STIMS])
# P[np.isnan(P)] = 0.5
# y = y_full.values




# from scipy.stats import mannwhitneyu
# from sklearn.metrics import make_scorer

# def mw_pvalue(estimator, X, y):
#     """Return the two-tailed Mann–Whitney p-value for probas."""
#     proba = estimator.predict_proba(X)[:, 1]
#     p_neg = proba[y == 0]
#     p_pos = proba[y == 1]
#     _, p = mannwhitneyu(p_neg, p_pos, alternative="two-sided")
#     return p                              # we will *minimise* this value

# # scorer that tells GridSearchCV ‘smaller is better’
# mw_scorer = make_scorer(mw_pvalue, greater_is_better=False, needs_proba=False)


# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# meta_pipe = Pipeline([
#     ("scaler", StandardScaler()),
#     ("clf",    LogisticRegression(solver="saga",
#                                   class_weight="balanced",
#                                   max_iter=20000,
#                                   random_state=SEED))
# ])
# param_grid = {
#     "clf__penalty":  ["l1", "l2", "elasticnet"],
#     "clf__C":        np.logspace(-4, 3, 40),
#     "clf__l1_ratio": [0.2, 0.5, 0.8]          
# }
# inner_cv = RepeatedStratifiedKFold(
#     n_splits=5, n_repeats=5, random_state=SEED)
# kf_ext = KFold(n_splits=5, shuffle=True, random_state=SEED)
# meta_oof       = np.zeros_like(y, dtype=float)
# weights_folds  = []
# intercepts_folds = []
# for tr, te in kf_ext.split(P):
#     gs = GridSearchCV(meta_pipe, param_grid, scoring="roc_auc",
#                       cv=inner_cv, n_jobs=-1, verbose=0)
#     gs.fit(P[tr], y[tr])
#     best = gs.best_estimator_
#     meta_oof[te] = best.predict_proba(P[te])[:, 1]

#     w  = best.named_steps["clf"].coef_[0]
#     b  = best.named_steps["clf"].intercept_[0]
#     weights_folds.append(w)
#     intercepts_folds.append(b)


# # -----------------------------------------------------------
# # 3) ÉVALUATION OOF + p-value non paramétrique (Mann–Whitney)
# # -----------------------------------------------------------
# from scipy.stats import mannwhitneyu, pearsonr

# auc_oof = roc_auc_score(y, meta_oof)

# # Mann–Whitney U (p-value non paramétrique)
# pred_neg = meta_oof[y == 0]
# pred_pos = meta_oof[y == 1]
# U, p_mw  = mannwhitneyu(pred_neg, pred_pos, alternative="two-sided")

# # (facultatif) corrélation de Pearson pour information
# r, p_pear = pearsonr(y, meta_oof)

# print("\n=== Méta-modèle OOF ===")
# print(f"AUC OOF             : {auc_oof:.3f}")
# print(f"Mann–Whitney p-val  : {p_mw:.3e}")
# print(f"Pearson r (option)  : {r:.3f} | p={p_pear:.3e}")

# # -----------------------------------------------------------
# # 4) AGRÉGATION DES POIDS (déjà calculée plus haut)
# # -----------------------------------------------------------
# weights_folds   = np.vstack(weights_folds)
# intercepts_folds= np.array(intercepts_folds)

# mean_w   = weights_folds.mean(axis=0)
# median_w = np.median(weights_folds, axis=0)
# mean_b   = intercepts_folds.mean()
# median_b = np.median(intercepts_folds)

# print("\nPoids (moyenne) :", dict(zip(STIMS, np.round(mean_w, 3))))
# print("Poids (médiane) :", dict(zip(STIMS, np.round(median_w, 3))))

# # ---------- Exports ----------
# pd.Series(mean_w, index=STIMS, name="weight_mean")      .to_csv("lf_weights_mean.csv")
# pd.Series(median_w, index=STIMS, name="weight_median")  .to_csv("lf_weights_median.csv")
# pd.Series([mean_b], index=["intercept_mean"]).to_csv("lf_intercept_mean.csv")
# pd.Series([median_b], index=["intercept_median"]).to_csv("lf_intercept_median.csv")
# pd.DataFrame({"y_true": y, "y_pred": meta_oof}).to_csv("lf_predictions.csv", index=False)

# print("\nExports : lf_weights_mean.csv | lf_weights_median.csv | "
#       "lf_intercept_mean.csv | lf_intercept_median.csv | lf_predictions.csv")


# !/usr/bin/env python3
# -*- coding: utf-8 -*-


# from pathlib import Path
# from itertools import product
# import numpy as np, pandas as pd
# from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.pipeline        import Pipeline
# from sklearn.impute          import SimpleImputer
# from sklearn.preprocessing   import StandardScaler
# from sklearn.linear_model    import LogisticRegression
# from sklearn.metrics         import roc_auc_score
# from scipy.stats             import mannwhitneyu, pearsonr


# ROOT      = Path("./files")
# STIMS     = ["16", "17", "18"]
# CSV_DATA  = "data4.csv"
# CSV_FEATS = "selectedFeats.csv"
# Y_FILE    = Path("./files/outcome4.csv")
# SEED      = 42

# y_full = pd.read_csv(Y_FILE)["group"].astype(int).reset_index(drop=True)
# X_full = pd.read_csv(ROOT / CSV_DATA, index_col=0).reset_index(drop=True)

# prep = Pipeline([
#     ("imp", SimpleImputer(strategy="median")),
#     ("std", StandardScaler())
# ])

# cache = {s: [] for s in STIMS}
# rskf  = RepeatedStratifiedKFold(n_splits=5, n_repeats=100, random_state=SEED)

# base_clf = LogisticRegression(
#     penalty="l1",        
#     solver="saga",
#     class_weight="balanced",
#     max_iter=20000,
#     C=1,
#     random_state=SEED
# )

# for f, (tr, te) in enumerate(rskf.split(X_full, y_full), 1):
#     y_tr = y_full.iloc[tr]

#     for stim in STIMS:
#         feat_file = ROOT / stim / CSV_FEATS
#         if not feat_file.exists():
#             cache[stim].append((te, np.full(len(te), 0.5)))
#             continue

#         feats = pd.read_csv(feat_file, index_col=0).astype(bool)
#         feats = feats.columns[feats.any()].tolist()
#         if not feats:
#             cache[stim].append((te, np.full(len(te), 0.5)))
#             continue

#         X_tr, X_te = X_full.loc[tr, feats], X_full.loc[te, feats]
#         model = Pipeline([("prep", prep), ("clf", base_clf)])
#         model.fit(X_tr, y_tr)
#         proba = model.predict_proba(X_te)[:, 1]  
#         cache[stim].append((te, proba))

#     if f % 50 == 0:
#         print(f"Fold {f}/{rskf.get_n_splits()} terminé.")

# clf = model.named_steps['clf']
# print("Penalty:", clf.penalty)        # devrait afficher "l1"
# print("C:", clf.C)  

# #mediane preds
# oof_proba = {}
# for stim in STIMS:
#     stack = np.full((len(cache[stim]), len(y_full)), np.nan)
#     for k, (idx, preds) in enumerate(cache[stim]):
#         stack[k, idx] = preds
#     oof_proba[stim] = np.nanmedian(stack, axis=0)      

# P = np.column_stack([oof_proba[s] for s in STIMS])    
# P[np.isnan(P)] = 0.5
# y = y_full.values

# # random looking
# step   = 0.02
# grid   = np.arange(0.0, 1.0 + 1e-9, step)
# best_auc = -1.0
# best_w   = (1, 0, 0)

# for w16, w17 in product(grid, repeat=2):
#     w18 = 1.0 - w16 - w17
#     if w18 < 0:
#         continue
#     y_pred = w16*P[:, 0] + w17*P[:, 1] + w18*P[:, 2]
#     auc    = roc_auc_score(y, y_pred)
#     if auc > best_auc:
#         best_auc, best_w = auc, (w16, w17, w18)

# y_hat = best_w[0]*P[:, 0] + best_w[1]*P[:, 1] + best_w[2]*P[:, 2]

# # metrics
# u_stat, p_mw = mannwhitneyu(y_hat[y == 0], y_hat[y == 1], alternative="two-sided")
# r, p_r = pearsonr(y, y_hat)

# print("\n=== Late-fusion simplexe (classification) ===")
# print("Poids optimaux :", dict(zip(STIMS, np.round(best_w, 3))))
# print(f"AUC OOF        : {best_auc:.3f}")
# print(f"p-value MW     : {p_mw:.3e}")
# print(f"Pearson r      : {r:.3f} | p={p_r:.3e}")

# # results
# pd.Series(best_w, index=STIMS, name="weight_simplex").to_csv("lf_weights_simplex.csv")
# pd.DataFrame({"y_true": y, "y_pred": y_hat}).to_csv("lf_predictions.csv", index=False)

# print("\nExports : lf_weights_simplex.csv | lf_predictions.csv")


# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# Repérage des variables sélectionnées (coef ≠ 0) par une régression
# logistique L1 *entraînée sur TOUT le jeu de données* pour chaque stim.

# • dataset commun       : ./files/data4.csv
# • masque de variables  : ./files/<stim>/selectedFeats.csv
# • étiquettes (0 / 1)   : ./files/outcome4.csv  (colonne “group”)
# • stims analysés       : 16, 17, 18
# • modèle               : LogisticRegression L1  (solver = saga)

# Exports
# -------
# un fichier CSV par stim :  features_<stim>.csv  contenant la liste
# des variables dont le coefficient final est non-nul.
# """

# from pathlib import Path
# import numpy as np
# import pandas as pd
# from sklearn.pipeline      import Pipeline
# from sklearn.impute        import SimpleImputer
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model  import LogisticRegression

# # ─────────── Fichiers / hyperparamètres ───────────
# ROOT      = Path("./files")
# STIMS     = ["16", "17", "18"]

# X_FILE    = ROOT / "data4.csv"
# Y_FILE    = ROOT / "outcome4.csv"
# MASK_TPL  = "{stim}/selectedFeats.csv"      # masque binaire par stim

# SEED      = 42
# TOL       = 1e-9                            # seuil “≈ 0” pour les coefs

# # ─────────── Lecture des données ───────────
# y = pd.read_csv(Y_FILE)["group"].astype(int).reset_index(drop=True)
# X = pd.read_csv(X_FILE, index_col=0).reset_index(drop=True)

# # ─────────── Pré-processing & modèle ───────────
# pipeline = Pipeline([
#     ("imp", SimpleImputer(strategy="median")),
#     ("std", StandardScaler()),
#     ("clf", LogisticRegression(
#         penalty="l1",
#         solver="saga",
#         class_weight="balanced",
#         max_iter=20_000,
#         C=1,
#         random_state=SEED))
# ])

# # ─────────── Boucle sur chaque stim ───────────
# for stim in STIMS:
#     # 1) variables retenues pour ce stim
#     mask_path = ROOT / MASK_TPL.format(stim=stim)
#     mask_df   = pd.read_csv(mask_path, index_col=0).astype(bool)
#     cols      = mask_df.columns[mask_df.any()]

#     # 2) apprentissage sur TOUT le jeu
#     pipeline.fit(X[cols], y)

#     # 3) récupération des coefficients
#     coef = pipeline.named_steps["clf"].coef_[0]
#     sel  = np.abs(coef) > TOL               # booléen : coef non-nul
#     kept_features = cols[sel]

#     # 4) export
#     out_file = f"features_{stim}.csv"
#     kept_features.to_series(index=kept_features).to_csv(out_file, header=False)
#     print(f"{stim} → {len(kept_features)} variables retenues  ➜  {out_file}")




#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Late-fusion propre (nested) avec masques de features par fold si disponibles.

Niveau 1 (par stimulus) :
  - Pipeline: Imputer(median) -> StandardScaler -> LogisticRegression L1 (saga)
  - OOF via RepeatedStratifiedKFold
  - Utilise un masque de features "par fold" depuis selectedFeats.csv si présent :
      - lignes attendues : Fold_1, Fold_2, ...
      - sinon fallback : colonnes avec any(True)

Niveau 2 (fusion) :
  - Poids sur le simplexe (>=0, somme=1) choisis sur TRAIN outer (max AUC)
  - Évaluation sur TEST outer (StratifiedKFold, shuffle=True)
  - Exports : poids moyens/médians, prédictions OOF niveau-2
"""

from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu, pearsonr

# ─────────────── I/O & constantes ───────────────
ROOT      = Path("./files")
STIMS     = ["16", "17", "18"]        
CSV_DATA  = "data4.csv"               
CSV_FEATS = "selectedFeats.csv"       
Y_FILE    = ROOT / "outcome4.csv"     
SEED      = 42

# ─────────────── Chargement données ─────────────
y_full = pd.read_csv(Y_FILE)["group"].astype(int).reset_index(drop=True)
X_full = pd.read_csv(ROOT / CSV_DATA, index_col=0).reset_index(drop=True)

# ─────────────── Pipeline niveau 1 ──────────────
prep = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("std", StandardScaler())
])

base_clf = LogisticRegression(
    penalty="l1",
    solver="saga",
    class_weight="balanced",
    max_iter=20_000,
    C=1.0,
    random_state=SEED
)

# ─────────────── OOF par stimulus ───────────────
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=100, random_state=SEED)
cache = {s: [] for s in STIMS}  

def cols_for_fold(feat_file: Path, fold_idx_1based: int) -> list[str]:
    """
    Retourne la liste des colonnes sélectionnées pour CE fold si possible.
    - Si selectedFeats.csv a une ligne 'Fold_{i}' => on l'utilise.
    - Sinon, on prend les colonnes avec any(True) (fallback global).
    """
    df = pd.read_csv(feat_file, index_col=0).astype(bool)

    # mask per fold
    row_label = f"Fold_{fold_idx_1based}"
    if row_label in df.index:
        row = df.loc[row_label]
        return row.index[row.values].tolist()

    #2 mod K if number of fold too big
    fold_rows = [idx for idx in df.index if str(idx).startswith("Fold_")]
    if len(fold_rows) > 0:
        # essai wrap modulo
        try:
            
            k = len(fold_rows)
            wrap_label = f"Fold_{((fold_idx_1based - 1) % k) + 1}"
            if wrap_label in df.index:
                row = df.loc[wrap_label]
                return row.index[row.values].tolist()
        except Exception:
            pass  

    
    return df.columns[df.any()].tolist()

for f, (tr, te) in enumerate(rskf.split(X_full, y_full), 1):
    y_tr = y_full.iloc[tr]

    for stim in STIMS:
        feat_file = ROOT / stim / CSV_FEATS
        if not feat_file.exists():
            #no mask: 0.5
            cache[stim].append((te, np.full(len(te), 0.5)))
            continue

        cols = cols_for_fold(feat_file, f)  # essaie Fold_f puis wrap, sinon global
        if not cols:
            cache[stim].append((te, np.full(len(te), 0.5)))
            continue

        X_tr, X_te = X_full.loc[tr, cols], X_full.loc[te, cols]
        model = Pipeline([("prep", prep), ("clf", base_clf)])
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        cache[stim].append((te, proba))

    if f % 50 == 0:
        print(f"[N1] Folds complétés : {f}/{rskf.get_n_splits()}")

# Agrégation des OOF par médiane (stabilité)
oof_proba = {}
for stim in STIMS:
    stack = np.full((len(cache[stim]), len(y_full)), np.nan)
    for k, (idx, preds) in enumerate(cache[stim]):
        stack[k, idx] = preds
    oof_proba[stim] = np.nanmedian(stack, axis=0)

# Matrice P (N x 3) des scores niveau-1
P = np.column_stack([oof_proba[s] for s in STIMS])
P[np.isnan(P)] = 0.5
y = y_full.values

# ─────────────── LF  ───────────────
outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


step = 0.02
grid = np.arange(0.0, 1.0 + 1e-9, step)

oof_level2 = np.zeros_like(y, dtype=float)
weights_outer = []

for tr, te in outer.split(P, y):
    
    best_auc, best_w = -1.0, (1.0, 0.0, 0.0)
    for w16 in grid:
        for w17 in grid:
            w18 = 1.0 - w16 - w17
            if w18 < 0:
                continue
            y_pred_tr = w16 * P[tr, 0] + w17 * P[tr, 1] + w18 * P[tr, 2]
            auc_tr = roc_auc_score(y[tr], y_pred_tr)
            if auc_tr > best_auc:
                best_auc, best_w = auc_tr, (w16, w17, w18)

    weights_outer.append(best_w)

    
    oof_level2[te] = best_w[0] * P[te, 0] + best_w[1] * P[te, 1] + best_w[2] * P[te, 2]

# ─────────────── Metrics
auc_clean = roc_auc_score(y, oof_level2)
u_stat, p_mw = mannwhitneyu(oof_level2[y == 0], oof_level2[y == 1], alternative="two-sided")
r, p_r = pearsonr(y, oof_level2)

print("\n=== Late-fusion simplexe (nested, propre) ===")
print("Poids par fold (outer) :")
for i, w in enumerate(weights_outer, 1):
    print(f"  Fold {i}: w16={w[0]:.2f}, w17={w[1]:.2f}, w18={w[2]:.2f}")

print(f"\nAUC OOF niveau-2 : {auc_clean:.3f}")
print(f"p-value Mann–Whitney : {p_mw:.3e}")
print(f"Pearson r (point-biserial) : {r:.3f} | p={p_r:.3e}")


W = np.array(weights_outer)
w_mean = W.mean(axis=0)
w_med  = np.median(W, axis=0)

pd.Series(w_mean, index=STIMS, name="weight_mean").to_csv("lf_weights_mean.csv")
pd.Series(w_med,  index=STIMS, name="weight_median").to_csv("lf_weights_median.csv")

pd.DataFrame({"y_true": y, "y_pred": oof_level2}).to_csv("lf_predictions.csv", index=False)

print("\nExports : lf_weights_mean.csv | lf_weights_median.csv | lf_predictions.csv")