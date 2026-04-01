"""
============================================================
  Loan Default Prediction & Expected Loss Calculator
  Retail Banking â€” Risk Team Prototype
  Author: QR Team
============================================================

  EL  =  PD  x  LGD  x  EAD
  where:
    PD  = Probability of Default   (model output)
    LGD = Loss Given Default       = 1 - Recovery Rate = 1 - 0.10 = 0.90
    EAD = Exposure at Default      = loan_amt_outstanding
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics         import (roc_auc_score, roc_curve,
                                      average_precision_score,
                                      brier_score_loss,
                                      classification_report,
                                      precision_recall_curve)
from sklearn.pipeline        import Pipeline
from sklearn.calibration     import calibration_curve

# â”€â”€ palette â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BG     = '#0d1117'
PANEL  = '#161b22'
GRID   = '#21262d'
TEXT   = '#e6edf3'
MUTED  = '#8b949e'
COLORS = ['#58a6ff', '#f85149', '#3fb950', '#d29922']

plt.rcParams.update({
    'figure.facecolor': BG,   'axes.facecolor':  PANEL,
    'axes.edgecolor':   GRID, 'axes.labelcolor': MUTED,
    'xtick.color':      MUTED,'ytick.color':     MUTED,
    'grid.color':       GRID, 'grid.linewidth':  0.6,
    'text.color':       TEXT, 'font.family':     'DejaVu Sans',
    'legend.framealpha':0.3,  'legend.edgecolor':GRID,
})

RECOVERY_RATE = 0.10
LGD           = 1.0 - RECOVERY_RATE   # 0.90

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  LOAD DATA
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DATA_PATH = '/mnt/user-data/uploads/1774895656950_Task_3_and_4_Loan_Data.csv'
df = pd.read_csv(DATA_PATH)

print("=" * 62)
print("  RETAIL BANKING â€” LOAN DEFAULT PROTOTYPE")
print("=" * 62)
print(f"\n  Rows: {len(df):,}   |   Columns: {df.shape[1]}")
print(f"  Default rate: {df['default'].mean():.2%}")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  FEATURE ENGINEERING
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d['debt_to_income'] = d['total_debt_outstanding'] / (d['income'] + 1)
    d['loan_to_income'] = d['loan_amt_outstanding']   / (d['income'] + 1)
    d['loan_to_debt']   = d['loan_amt_outstanding']   / (d['total_debt_outstanding'] + 1)
    d['fico_band']      = pd.cut(d['fico_score'],
                                  bins=[300, 579, 619, 659, 699, 739, 850],
                                  labels=[1, 2, 3, 4, 5, 6]).astype(float)
    return d

df_eng = engineer(df)

TARGET       = 'default'
DROP_COLS    = ['customer_id', TARGET]
FEATURE_COLS = [c for c in df_eng.columns if c not in DROP_COLS]

X = df_eng[FEATURE_COLS]
y = df_eng[TARGET]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  TRAIN / TEST SPLIT  (80 / 20, stratified)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,}   Test: {len(X_test):,}")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  MODELS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
models = {
    'Logistic Regression': Pipeline([
        ('sc',  StandardScaler()),
        ('clf', LogisticRegression(C=0.5, max_iter=1000,
                                    class_weight='balanced', random_state=42))
    ]),
    'Decision Tree': Pipeline([
        ('clf', DecisionTreeClassifier(max_depth=6, min_samples_leaf=30,
                                        class_weight='balanced', random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('clf', RandomForestClassifier(n_estimators=300, max_depth=8,
                                        min_samples_leaf=20,
                                        class_weight='balanced',
                                        n_jobs=-1, random_state=42))
    ]),
    'Gradient Boosting': Pipeline([
        ('clf', GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                            learning_rate=0.05,
                                            subsample=0.8, random_state=42))
    ]),
}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5.  TRAINING & EVALUATION
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\n" + "-" * 62)
print("  MODEL TRAINING & EVALUATION")
print("-" * 62)

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    probs  = pipe.predict_proba(X_test)[:, 1]
    preds  = pipe.predict(X_test)
    auc    = roc_auc_score(y_test, probs)
    ap     = average_precision_score(y_test, probs)
    brier  = brier_score_loss(y_test, probs)
    cv_auc = cross_val_score(pipe, X_train, y_train,
                              cv=cv, scoring='roc_auc').mean()
    results[name] = dict(pipe=pipe, probs=probs, preds=preds,
                          auc=auc, ap=ap, brier=brier, cv_auc=cv_auc)
    print(f"\n  [{name}]")
    print(f"    AUC-ROC     : {auc:.4f}")
    print(f"    CV AUC      : {cv_auc:.4f}")
    print(f"    Avg Prec    : {ap:.4f}")
    print(f"    Brier Score : {brier:.4f}  (lower = better)")

best_name  = max(results, key=lambda k: results[k]['auc'])
best_pipe  = results[best_name]['pipe']
best_probs = results[best_name]['probs']

print(f"\n  Best model: {best_name}  (AUC = {results[best_name]['auc']:.4f})")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 6.  FIG 1 â€” EDA
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
fig1.suptitle('Exploratory Data Analysis â€” Loan Book',
               fontsize=15, fontweight='bold', color=TEXT, y=1.01)

num_features = ['fico_score', 'income', 'loan_amt_outstanding',
                 'total_debt_outstanding', 'years_employed',
                 'credit_lines_outstanding']
for ax, feat in zip(axes.flat, num_features):
    d0 = df[df['default'] == 0][feat]
    d1 = df[df['default'] == 1][feat]
    ax.hist(d0, bins=35, alpha=0.65, color=COLORS[0], label='No Default', density=True)
    ax.hist(d1, bins=35, alpha=0.65, color=COLORS[1], label='Default',    density=True)
    ax.set_title(feat.replace('_', ' ').title(), color=TEXT, fontsize=11, fontweight='bold')
    ax.set_xlabel('Value', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)

plt.tight_layout()
fig1.savefig('/mnt/user-data/outputs/fig1_eda.png', dpi=140,
             bbox_inches='tight', facecolor=BG)
print("\n  Fig 1 (EDA) saved.")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 7.  FIG 2 â€” MODEL PERFORMANCE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
fig2 = plt.figure(figsize=(20, 18))
gs   = gridspec.GridSpec(3, 2, figure=fig2, hspace=0.45, wspace=0.35)

# 2A ROC
ax_roc = fig2.add_subplot(gs[0, 0])
ax_roc.plot([0,1],[0,1], '--', color=MUTED, lw=1)
for (name, res), col in zip(results.items(), COLORS):
    fpr, tpr, _ = roc_curve(y_test, res['probs'])
    ax_roc.plot(fpr, tpr, color=col, lw=2,
                label=f"{name}  AUC={res['auc']:.3f}")
ax_roc.set_title('ROC Curves', fontsize=13, fontweight='bold', color=TEXT)
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.legend(fontsize=8); ax_roc.grid(True, alpha=0.4)

# 2B Precision-Recall
ax_pr = fig2.add_subplot(gs[0, 1])
baseline = y_test.mean()
ax_pr.axhline(baseline, ls='--', color=MUTED, lw=1, label=f'Baseline ({baseline:.2f})')
for (name, res), col in zip(results.items(), COLORS):
    prec, rec, _ = precision_recall_curve(y_test, res['probs'])
    ax_pr.plot(rec, prec, color=col, lw=2, label=f"{name}  AP={res['ap']:.3f}")
ax_pr.set_title('Precision-Recall Curves', fontsize=13, fontweight='bold', color=TEXT)
ax_pr.set_xlabel('Recall')
ax_pr.set_ylabel('Precision')
ax_pr.legend(fontsize=8); ax_pr.grid(True, alpha=0.4)

# 2C Calibration
ax_cal = fig2.add_subplot(gs[1, 0])
ax_cal.plot([0,1],[0,1], '--', color=MUTED, lw=1, label='Perfect calibration')
for (name, res), col in zip(results.items(), COLORS):
    frac_pos, mean_pred = calibration_curve(y_test, res['probs'], n_bins=10)
    ax_cal.plot(mean_pred, frac_pos, 'o-', color=col, lw=2, ms=4, label=name)
ax_cal.set_title('Calibration Curves', fontsize=13, fontweight='bold', color=TEXT)
ax_cal.set_xlabel('Mean Predicted Probability')
ax_cal.set_ylabel('Fraction of Positives')
ax_cal.legend(fontsize=8); ax_cal.grid(True, alpha=0.4)

# 2D Model comparison bars
ax_bar = fig2.add_subplot(gs[1, 1])
metrics       = ['auc', 'cv_auc', 'ap']
metric_labels = ['AUC-ROC', 'CV AUC', 'Avg Precision']
x = np.arange(len(metrics)); w = 0.18
for i, (name, res) in enumerate(results.items()):
    vals = [res[m] for m in metrics]
    ax_bar.bar(x + i*w, vals, w, label=name, color=COLORS[i],
               alpha=0.85, edgecolor=BG, lw=0.5)
ax_bar.set_xticks(x + w*1.5); ax_bar.set_xticklabels(metric_labels)
ax_bar.set_ylim(0.5, 1.0)
ax_bar.set_title('Model Comparison', fontsize=13, fontweight='bold', color=TEXT)
ax_bar.set_ylabel('Score')
ax_bar.legend(fontsize=8); ax_bar.grid(True, alpha=0.4, axis='y')

# 2E Feature importance
ax_fi = fig2.add_subplot(gs[2, 0])
clf = best_pipe.named_steps['clf']
if hasattr(clf, 'feature_importances_'):
    fi = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values()
elif hasattr(clf, 'coef_'):
    fi = pd.Series(np.abs(clf.coef_[0]), index=FEATURE_COLS).sort_values()
else:
    fi = None
if fi is not None:
    cmap = plt.cm.Blues(np.linspace(0.35, 0.9, len(fi)))
    ax_fi.barh(fi.index, fi.values, color=cmap, edgecolor=BG, lw=0.5)
    ax_fi.set_title(f'Feature Importance ({best_name})',
                    fontsize=13, fontweight='bold', color=TEXT)
    ax_fi.set_xlabel('Importance')
    ax_fi.grid(True, alpha=0.4, axis='x')

# 2F Expected Loss distribution
ax_el = fig2.add_subplot(gs[2, 1])
ead_test = X_test['loan_amt_outstanding'].values
EL_test  = best_probs * LGD * ead_test
ax_el.hist(EL_test[y_test == 0], bins=50, alpha=0.65, density=True,
           color=COLORS[0], label='No Default')
ax_el.hist(EL_test[y_test == 1], bins=50, alpha=0.65, density=True,
           color=COLORS[1], label='Default')
ax_el.axvline(EL_test.mean(), color=COLORS[3], lw=2, ls='--',
              label=f"Mean EL = Â£{EL_test.mean():,.0f}")
ax_el.set_title('Expected Loss Distribution', fontsize=13, fontweight='bold', color=TEXT)
ax_el.set_xlabel('Expected Loss (Â£)')
ax_el.set_ylabel('Density')
ax_el.legend(fontsize=8); ax_el.grid(True, alpha=0.4)

fig2.suptitle(
    f'Model Performance â€” Best: {best_name}  (AUC = {results[best_name]["auc"]:.4f})',
    fontsize=14, fontweight='bold', color=TEXT, y=1.005)
fig2.savefig('/mnt/user-data/outputs/fig2_model_performance.png', dpi=140,
             bbox_inches='tight', facecolor=BG)
print("  Fig 2 (Model Performance) saved.")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 8.  CLASSIFICATION REPORT
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n{'-'*62}")
print(f"  CLASSIFICATION REPORT  ({best_name})")
print(f"{'-'*62}")
print(classification_report(y_test, results[best_name]['preds'],
                              target_names=['No Default', 'Default']))

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 9.  PORTFOLIO EXPECTED LOSS SUMMARY
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
df_all     = engineer(df)
X_all      = df_all[FEATURE_COLS]
df['pd']   = best_pipe.predict_proba(X_all)[:, 1]
df['el']   = df['pd'] * LGD * df['loan_amt_outstanding']
df['band'] = pd.cut(df['pd'],
                     bins=[-np.inf, 0.10, 0.25, 0.50, np.inf],
                     labels=['Low (<10%)', 'Medium (10-25%)',
                              'High (25-50%)', 'Very High (>50%)'])

print(f"\n{'-'*62}")
print("  PORTFOLIO EXPECTED LOSS SUMMARY")
print(f"{'-'*62}")
print(f"  Recovery Rate      : {RECOVERY_RATE:.0%}")
print(f"  LGD                : {LGD:.0%}")
print(f"  Total EAD          : Â£{df['loan_amt_outstanding'].sum():>15,.2f}")
print(f"  Total Expected Loss: Â£{df['el'].sum():>15,.2f}")
print(f"  Mean PD            : {df['pd'].mean():.2%}")
print(f"\n  Risk Band Distribution:")
band_summary = df.groupby('band', observed=True).agg(
    Count=('customer_id', 'count'),
    Total_EL=('el', 'sum'),
    Mean_PD=('pd', 'mean')
)
print(band_summary.to_string())

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 10.  PREDICTION FUNCTION  (Charlie's plug-in)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def predict_expected_loss(
    credit_lines_outstanding: int,
    loan_amt_outstanding:     float,
    total_debt_outstanding:   float,
    income:                   float,
    years_employed:           int,
    fico_score:               int,
    recovery_rate:            float = RECOVERY_RATE,
) -> dict:
    """
    Estimate the Expected Loss for a single loan.

    EL = PD x LGD x EAD
      PD  = model-predicted probability of default
      LGD = 1 - recovery_rate
      EAD = loan_amt_outstanding

    Parameters
    ----------
    credit_lines_outstanding : Number of open credit lines
    loan_amt_outstanding     : Current loan balance (GBP)
    total_debt_outstanding   : Total borrower debt (GBP)
    income                   : Annual income (GBP)
    years_employed           : Years at current employer
    fico_score               : FICO score (300-850)
    recovery_rate            : Fraction recovered on default (default 0.10)

    Returns
    -------
    dict with keys: pd, lgd, ead, el, risk_band, model
    """
    raw = pd.DataFrame([{
        'credit_lines_outstanding': credit_lines_outstanding,
        'loan_amt_outstanding':     loan_amt_outstanding,
        'total_debt_outstanding':   total_debt_outstanding,
        'income':                   income,
        'years_employed':           years_employed,
        'fico_score':               fico_score,
    }])
    feat   = engineer(raw)[FEATURE_COLS]
    pd_val = float(best_pipe.predict_proba(feat)[0, 1])
    lgd_val = 1.0 - recovery_rate
    ead_val = loan_amt_outstanding
    el_val  = pd_val * lgd_val * ead_val

    if pd_val < 0.10:   band = 'Low'
    elif pd_val < 0.25: band = 'Medium'
    elif pd_val < 0.50: band = 'High'
    else:               band = 'Very High'

    return {
        'pd':        round(pd_val,  4),
        'lgd':       round(lgd_val, 4),
        'ead':       round(ead_val, 2),
        'el':        round(el_val,  2),
        'risk_band': band,
        'model':     best_name,
    }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 11.  DEMO
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
demo = [
    dict(label='Low-risk borrower',
         credit_lines_outstanding=1, loan_amt_outstanding=4500,
         total_debt_outstanding=5000, income=95000,
         years_employed=8, fico_score=740),
    dict(label='Medium-risk borrower',
         credit_lines_outstanding=3, loan_amt_outstanding=5500,
         total_debt_outstanding=15000, income=52000,
         years_employed=3, fico_score=620),
    dict(label='High-risk borrower',
         credit_lines_outstanding=5, loan_amt_outstanding=8000,
         total_debt_outstanding=30000, income=28000,
         years_employed=1, fico_score=490),
]

print(f"\n{'-'*62}")
print(f"  EXAMPLE PREDICTIONS  (Recovery Rate = {RECOVERY_RATE:.0%})")
print(f"{'-'*62}")
for d in demo:
    label = d.pop('label')
    out   = predict_expected_loss(**d)
    print(f"\n  {label}")
    print(f"    Income : Â£{d['income']:>8,.0f}   FICO: {d['fico_score']}")
    print(f"    Loan   : Â£{d['loan_amt_outstanding']:>8,.2f}   Debt: Â£{d['total_debt_outstanding']:,.2f}")
    print(f"    PD     : {out['pd']:.2%}   LGD: {out['lgd']:.0%}   EAD: Â£{out['ead']:,.2f}")
    print(f"    Expected Loss : Â£{out['el']:,.2f}   Risk Band: {out['risk_band']}")

print("\n" + "=" * 62)
print("  predict_expected_loss() is ready for integration.")
print("=" * 62)