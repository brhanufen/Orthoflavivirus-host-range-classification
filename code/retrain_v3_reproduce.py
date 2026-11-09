#!/usr/bin/env python3
"""
Standalone reproduction of the expanded-dataset (v3) retraining and LOSO-CV.
Input: X_features_expanded_v3.csv (1,528 genomes, 97 features + metadata)
Reproduces: held-out test metrics (Table/Fig S2) and leave-one-species-out CV.
Positive class = ISFV (invertebrate_only). Protocol matches the deposited pipeline.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             balanced_accuracy_score, matthews_corrcoef, confusion_matrix)

df = pd.read_csv("X_features_expanded_v3.csv")

META = ['accession','species_key','host_range_label','y','split','vector_type','provenance']
feat_cols = [c for c in df.columns if c not in META]
assert len(feat_cols) == 97, f"expected 97 features, got {len(feat_cols)}"

# label: ISFV is the positive class
df['y_isfv'] = (df['host_range_label'] == 'invertebrate_only').astype(int)

# drop any all-NaN feature rows (one original WNV genome, M12294.2)
df = df.dropna(subset=feat_cols).reset_index(drop=True)

# ---- leakage guard: no species may appear in more than one split ----
sp_splits = df.groupby('species_key')['split'].nunique()
leaked = sp_splits[sp_splits > 1]
assert len(leaked) == 0, f"SPECIES LEAKAGE: {leaked.index.tolist()}"

tr = df[df.split == 'train']; te = df[df.split == 'test']
# guard: no shared species across the specific train/test partition either
assert set(tr.species_key) & set(te.species_key) == set(), "train/test species overlap"

# ---- held-out test: full 97-feature models, scaler fit on TRAIN only ----
scaler = StandardScaler().fit(tr[feat_cols])
Xtr, Xte = scaler.transform(tr[feat_cols]), scaler.transform(te[feat_cols])
ytr, yte = tr['y_isfv'].values, te['y_isfv'].values

lr = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000,
                        class_weight='balanced', random_state=42).fit(Xtr, ytr)
rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5,
                            class_weight='balanced', random_state=42).fit(Xtr, ytr)

def report(clf, name):
    p = clf.predict_proba(Xte)[:, 1]; pred = clf.predict(Xte)
    cm = confusion_matrix(yte, pred)
    return dict(model=name,
                PR_AUC=average_precision_score(yte, p),
                ROC_AUC=roc_auc_score(yte, p),
                BalAcc=balanced_accuracy_score(yte, pred),
                MCC=matthews_corrcoef(yte, pred),
                spec=cm[0,0]/cm[0].sum(), sens=cm[1,1]/cm[1].sum())

print("=== HELD-OUT TEST (n=%d, %d ISFV / %d species) ==="
      % (len(te), yte.sum(), te.species_key[te.y_isfv==1].nunique()))
for r in (report(lr,'LR'), report(rf,'RF')):
    print("  %s: PR-AUC=%.4f ROC=%.4f BalAcc=%.4f MCC=%.4f sens=%.3f spec=%.3f"
          % (r['model'], r['PR_AUC'], r['ROC_AUC'], r['BalAcc'], r['MCC'], r['sens'], r['spec']))

# ---- LOSO-CV: dinucleotide features only (16 freq + 18 O/E = 34), per-fold scaler ----
dinuc = [c for c in feat_cols if (c.endswith('_freq') and len(c.split('_')[0])==2) or c.endswith('_OE')]
assert len(dinuc) == 34, f"expected 34 dinuc features, got {len(dinuc)}"
isfv_species = sorted(df[df.y_isfv==1].species_key.unique())
rows = []
for sp in isfv_species:
    hold = df.species_key == sp
    train_mask = ~hold
    sc = StandardScaler().fit(df.loc[train_mask, dinuc])
    clf = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000,
                             class_weight='balanced', random_state=42).fit(
                             sc.transform(df.loc[train_mask, dinuc]), df.loc[train_mask,'y_isfv'])
    pred = clf.predict(sc.transform(df.loc[hold, dinuc]))
    n = int(hold.sum())
    rows.append(dict(species=sp, n=n, n_correct=int((pred==1).sum()), acc=(pred==1).sum()/n))
lo = pd.DataFrame(rows)
sp_ok = (lo.acc >= 0.5).sum()
print("\n=== LOSO-CV (%d ISFV species, dinucleotide-only) ===" % len(lo))
print("  species majority-correct: %d/%d = %.1f%%" % (sp_ok, len(lo), 100*sp_ok/len(lo)))
print("  genomes correct: %d/%d = %.1f%%" % (lo.n_correct.sum(), lo.n.sum(),
                                              100*lo.n_correct.sum()/lo.n.sum()))
print("  misclassified species:", lo[lo.acc<0.5].species.tolist())
