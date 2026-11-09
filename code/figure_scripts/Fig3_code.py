#!/usr/bin/env python3
"""
Figure 3. Classification performance for distinguishing ISFVs from dual-host flaviviruses.
Publication-quality figure for Journal of Virology (ASM).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── ASM-compliant settings ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 7,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

DUAL_COLOR = '#D62728'
ISFV_COLOR = '#1F77B4'
LR_COLOR   = '#1F77B4'   # Blue for Logistic Regression
RF_COLOR   = '#FF7F0E'   # Orange for Random Forest (colorblind-safe, not green)

# ── Load data ───────────────────────────────────────────────────────────
pr_df = pd.read_csv('figure_data/Fig3A_pr_curve_data.csv')
roc_df = pd.read_csv('figure_data/Fig3B_roc_curve_data.csv')
pr_auc = pd.read_csv('figure_data/Fig3A_pr_auc_summary.csv')
roc_auc = pd.read_csv('figure_data/Fig3B_roc_auc_summary.csv')
cm_lr = pd.read_csv('figure_data/Fig3C_confusion_matrix_lr.csv', index_col=0)
perf = pd.read_csv('figure_data/Fig3_performance_summary.csv')

# Extract curve data per model
pr_lr = pr_df[pr_df['model'] == 'Logistic Regression']
pr_rf = pr_df[pr_df['model'] == 'Random Forest']
roc_lr = roc_df[roc_df['model'] == 'Logistic Regression']
roc_rf = roc_df[roc_df['model'] == 'Random Forest']

# AUC values
lr_pr_auc = pr_auc[pr_auc['Model'] == 'Logistic Regression']['PR_AUC'].values[0]
rf_pr_auc = pr_auc[pr_auc['Model'] == 'Random Forest']['PR_AUC'].values[0]
baseline_pr = pr_auc[pr_auc['Model'] == 'Random Baseline']['PR_AUC'].values[0]
lr_roc_auc = roc_auc[roc_auc['Model'] == 'Logistic Regression']['ROC_AUC'].values[0]
rf_roc_auc = roc_auc[roc_auc['Model'] == 'Random Forest']['ROC_AUC'].values[0]

# Confusion matrix
cm = cm_lr.values  # rows: true ISFV/dual-host; columns: predicted ISFV/dual-host -> [[15, 2], [0, 103]]

# ── Figure layout ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(6.875, 2.8))

gs = gridspec.GridSpec(
    1, 3, figure=fig,
    width_ratios=[1, 1, 0.85],
    wspace=0.38,
)

ax_pr  = fig.add_subplot(gs[0, 0])
ax_roc = fig.add_subplot(gs[0, 1])
ax_cm  = fig.add_subplot(gs[0, 2])


# ════════════════════════════════════════════════════════════════════════
# PANEL A: Precision-Recall Curve
# ════════════════════════════════════════════════════════════════════════
ax_pr.text(-0.18, 1.10, 'A', transform=ax_pr.transAxes,
           fontsize=12, fontweight='bold', va='top')

# Plot curves
ax_pr.step(pr_lr['recall'], pr_lr['precision'], where='post',
           color=LR_COLOR, linewidth=1.2, label=f'LogReg (AUC = {lr_pr_auc:.3f})', zorder=3)
ax_pr.step(pr_rf['recall'], pr_rf['precision'], where='post',
           color=RF_COLOR, linewidth=1.2, label=f'RF (AUC = {rf_pr_auc:.3f})', zorder=2)

# Random baseline
ax_pr.axhline(y=baseline_pr, color='#999999', linewidth=0.7, linestyle='--',
              label=f'Random ({baseline_pr:.3f})', zorder=1)

ax_pr.set_xlabel('Recall', fontsize=8)
ax_pr.set_ylabel('Precision', fontsize=8)
ax_pr.set_title('Precision-Recall Curve', fontsize=9, fontweight='bold', pad=6)
ax_pr.set_xlim(-0.02, 1.02)
ax_pr.set_ylim(-0.02, 1.08)
ax_pr.spines['top'].set_visible(False)
ax_pr.spines['right'].set_visible(False)

leg_pr = ax_pr.legend(loc='center right', frameon=True, framealpha=0.95,
                       edgecolor='#CCC', fontsize=5.8, handlelength=1.5,
                       borderpad=0.4, labelspacing=0.3)
leg_pr.get_frame().set_linewidth(0.5)


# ════════════════════════════════════════════════════════════════════════
# PANEL B: ROC Curve
# ════════════════════════════════════════════════════════════════════════
ax_roc.text(-0.18, 1.10, 'B', transform=ax_roc.transAxes,
            fontsize=12, fontweight='bold', va='top')

ax_roc.plot(roc_lr['fpr'], roc_lr['tpr'],
            color=LR_COLOR, linewidth=1.2, label=f'LogReg (AUC = {lr_roc_auc:.3f})', zorder=3)
ax_roc.plot(roc_rf['fpr'], roc_rf['tpr'],
            color=RF_COLOR, linewidth=1.2, label=f'RF (AUC = {rf_roc_auc:.3f})', zorder=2)

# Diagonal reference
ax_roc.plot([0, 1], [0, 1], color='#999999', linewidth=0.7, linestyle='--',
            label='Random', zorder=1)

ax_roc.set_xlabel('False Positive Rate', fontsize=8)
ax_roc.set_ylabel('True Positive Rate', fontsize=8)
ax_roc.set_title('ROC Curve', fontsize=9, fontweight='bold', pad=6)
ax_roc.set_xlim(-0.02, 1.02)
ax_roc.set_ylim(-0.02, 1.08)
ax_roc.spines['top'].set_visible(False)
ax_roc.spines['right'].set_visible(False)

leg_roc = ax_roc.legend(loc='lower right', frameon=True, framealpha=0.95,
                         edgecolor='#CCC', fontsize=5.8, handlelength=1.5,
                         borderpad=0.4, labelspacing=0.3)
leg_roc.get_frame().set_linewidth(0.5)


# ════════════════════════════════════════════════════════════════════════
# PANEL C: Confusion Matrix (Logistic Regression)
# ════════════════════════════════════════════════════════════════════════
ax_cm.text(-0.22, 1.16, 'C', transform=ax_cm.transAxes,
           fontsize=12, fontweight='bold', va='top')

# Normalized confusion matrix for coloring
cm_norm = cm.astype(float)
cm_norm[0, :] = cm_norm[0, :] / cm_norm[0, :].sum()
cm_norm[1, :] = cm_norm[1, :] / cm_norm[1, :].sum()

# Color: blue palette
from matplotlib.colors import LinearSegmentedColormap
cm_cmap = LinearSegmentedColormap.from_list('custom_blue',
    ['#FFFFFF', '#BBDEFB', '#64B5F6', '#1565C0'], N=100)

im = ax_cm.imshow(cm_norm, cmap=cm_cmap, vmin=0, vmax=1, aspect='equal', zorder=2)
ax_cm.set_anchor('N')

# Annotate with raw counts and percentages
labels = [['ISFV', 'Dual-host'], ['ISFV', 'Dual-host']]
for i in range(2):
    for j in range(2):
        count = cm[i, j]
        pct = cm_norm[i, j] * 100
        text_color = 'white' if cm_norm[i, j] > 0.6 else 'black'
        ax_cm.text(j, i, f'{count}\n({pct:.1f}%)',
                   ha='center', va='center', fontsize=7,
                   fontweight='bold', color=text_color, zorder=3)

ax_cm.set_xticks([0, 1])
ax_cm.set_xticklabels(['ISFV', 'Dual-host'], fontsize=7)
ax_cm.set_yticks([0, 1])
ax_cm.set_yticklabels(['ISFV', 'Dual-host'], fontsize=7)
ax_cm.set_xlabel('Predicted', fontsize=8)
ax_cm.set_ylabel('True', fontsize=8)
ax_cm.set_title('Confusion Matrix (LR)', fontsize=9, fontweight='bold', pad=6)

# Grid lines
for spine in ax_cm.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.5)
    spine.set_color('#CCCCCC')

# Add thin lines between cells
ax_cm.set_xticks([0.5], minor=True)
ax_cm.set_yticks([0.5], minor=True)
ax_cm.grid(which='minor', color='white', linewidth=1.5)
ax_cm.tick_params(which='minor', length=0)

# Subtitle with key metrics
ax_cm.text(0.5, -0.30, f'Sensitivity: 88.2%  |  Specificity: 100%\n'
           f'Bal. Acc.: 0.941  |  MCC: 0.930',
           transform=ax_cm.transAxes, ha='center', va='top',
           fontsize=5.5, color='#555')


# ── Save ────────────────────────────────────────────────────────────────
import os
for fmt, dpi, ext in [('tiff', 600, 'tiff'), ('pdf', 600, 'pdf'), ('png', 200, 'png')]:
    fname = f'figure_data/Fig3_final.{ext}'
    kw = dict(pil_kwargs={'compression': 'tiff_lzw'}) if fmt == 'tiff' else {}
    fig.savefig(fname, dpi=dpi, format=fmt, bbox_inches='tight', pad_inches=0.08, **kw)

print("✓ Figure 3 saved:")
for ext in ['tiff', 'pdf', 'png']:
    mb = os.path.getsize(f'figure_data/Fig3_final.{ext}') / (1024**2)
    print(f"  Fig3_final.{ext}: {mb:.1f} MB")
