
from pathlib import Path
from itertools import product
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline        import Pipeline
from sklearn.impute          import SimpleImputer
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import roc_auc_score
from scipy.stats             import mannwhitneyu, pearsonr


ROOT      = Path("./files")
STIMS     = ["16", "17", "18"]
CSV_DATA  = "data4.csv"
CSV_FEATS = "selectedFeats.csv"
Y_FILE    = Path("./files/outcome4.csv")
SEED      = 42

y_full = pd.read_csv(Y_FILE)["group"].astype(int).reset_index(drop=True)
X_full = pd.read_csv(ROOT / CSV_DATA, index_col=0).reset_index(drop=True)

prep = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("std", StandardScaler())
])

cache = {s: [] for s in STIMS}
rskf  = RepeatedStratifiedKFold(n_splits=5, n_repeats=100, random_state=SEED)

base_clf = LogisticRegression(
    penalty="l1",        
    solver="saga",
    class_weight="balanced",
    max_iter=20000,
    C=1,
    random_state=SEED
)

for f, (tr, te) in enumerate(rskf.split(X_full, y_full), 1):
    y_tr = y_full.iloc[tr]

    for stim in STIMS:
        feat_file = ROOT / stim / CSV_FEATS
        if not feat_file.exists():
            cache[stim].append((te, np.full(len(te), 0.5)))
            continue

        feats = pd.read_csv(feat_file, index_col=0).astype(bool)
        feats = feats.columns[feats.any()].tolist()
        if not feats:
            cache[stim].append((te, np.full(len(te), 0.5)))
            continue

        X_tr, X_te = X_full.loc[tr, feats], X_full.loc[te, feats]
        model = Pipeline([("prep", prep), ("clf", base_clf)])
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]  
        cache[stim].append((te, proba))

    if f % 50 == 0:
        print(f"Fold {f}/{rskf.get_n_splits()} terminé.")

clf = model.named_steps['clf']
print("Penalty:", clf.penalty)        # devrait afficher "l1"
print("C:", clf.C)  

#mediane preds
oof_proba = {}
for stim in STIMS:
    stack = np.full((len(cache[stim]), len(y_full)), np.nan)
    for k, (idx, preds) in enumerate(cache[stim]):
        stack[k, idx] = preds
    oof_proba[stim] = np.nanmedian(stack, axis=0)      

P = np.column_stack([oof_proba[s] for s in STIMS])    
P[np.isnan(P)] = 0.5
y = y_full.values

# random looking
step   = 0.02
grid   = np.arange(0.0, 1.0 + 1e-9, step)
best_auc = -1.0
best_w   = (1, 0, 0)

for w16, w17 in product(grid, repeat=2):
    w18 = 1.0 - w16 - w17
    if w18 < 0:
        continue
    y_pred = w16*P[:, 0] + w17*P[:, 1] + w18*P[:, 2]
    auc    = roc_auc_score(y, y_pred)
    if auc > best_auc:
        best_auc, best_w = auc, (w16, w17, w18)

y_hat = best_w[0]*P[:, 0] + best_w[1]*P[:, 1] + best_w[2]*P[:, 2]

# metrics
u_stat, p_mw = mannwhitneyu(y_hat[y == 0], y_hat[y == 1], alternative="two-sided")
r, p_r = pearsonr(y, y_hat)

print("\n=== Late-fusion simplexe (classification) ===")
print("Poids optimaux :", dict(zip(STIMS, np.round(best_w, 3))))
print(f"AUC OOF        : {best_auc:.3f}")
print(f"p-value MW     : {p_mw:.3e}")
print(f"Pearson r      : {r:.3f} | p={p_r:.3e}")

# results
pd.Series(best_w, index=STIMS, name="weight_simplex").to_csv("lf_weights_simplex.csv")
pd.DataFrame({"y_true": y, "y_pred": y_hat}).to_csv("lf_predictions.csv", index=False)

print("\nExports : lf_weights_simplex.csv | lf_predictions.csv"
