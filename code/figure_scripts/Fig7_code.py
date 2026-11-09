#!/usr/bin/env python3
"""
Figure 7. Genus-level classification across Flaviviridae.
Layout: 2×2 grid (A confusion matrix, B per-class F1, C PCA, D feature importance).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "figure_data"
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 7,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'legend.fontsize': 6,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# ── Load data ───────────────────────────────────────────────────────────
cm_norm = pd.read_csv(DATA / 'Fig7A_confusion_matrix_normalized.csv', index_col=0)
cm_raw = pd.read_csv(DATA / 'Fig7A_confusion_matrix_raw.csv', index_col=0)
cv_summary = pd.read_csv(DATA / 'Fig7B_cv_summary.csv')
per_class = pd.read_csv(DATA / 'Fig7B_per_class_metrics.csv')
pca_coords = pd.read_csv(DATA / 'Fig7C_pca_coordinates.csv')
pca_var = pd.read_csv(DATA / 'Fig7C_pca_variance.csv')
rf_top15 = pd.read_csv(DATA / 'Fig7D_feature_importance_top15.csv')

genera = ['Flavivirus', 'Hepacivirus', 'Pegivirus', 'Pestivirus']
genus_colors = {
    'Flavivirus': '#D62728',
    'Hepacivirus': '#1F77B4',
    'Pestivirus': '#2CA02C',
    'Pegivirus': '#FF7F0E',
}

LR_COLOR = '#1F77B4'
RF_COLOR = '#D62728'

# ── Figure layout ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(6.875, 7.0))

gs = gridspec.GridSpec(
    2, 2, figure=fig,
    width_ratios=[1, 1],
    height_ratios=[1, 1],
    hspace=0.40, wspace=0.38,
)

ax_cm   = fig.add_subplot(gs[0, 0])   # A
ax_f1   = fig.add_subplot(gs[0, 1])   # B
ax_pca  = fig.add_subplot(gs[1, 0])   # C
ax_imp  = fig.add_subplot(gs[1, 1])   # D


# ════════════════════════════════════════════════════════════════════════
# PANEL A: Normalized Confusion Matrix
# ════════════════════════════════════════════════════════════════════════
ax_cm.text(-0.08, 1.12, 'A', transform=ax_cm.transAxes,
           fontsize=12, fontweight='bold', va='top')

norm_vals = cm_norm.values.astype(float)
raw_vals = cm_raw.values.astype(int)

im = ax_cm.imshow(norm_vals, cmap='Blues', vmin=0, vmax=1, aspect='equal')

# Annotate cells
for i in range(4):
    for j in range(4):
        nv = norm_vals[i, j]
        rv = raw_vals[i, j]
        color = 'white' if nv > 0.5 else 'black'
        ax_cm.text(j, i, f'{nv:.3f}\n({rv})',
                   ha='center', va='center', fontsize=5.5, color=color)

ax_cm.set_xticks(range(4))
ax_cm.set_xticklabels(genera, fontsize=6, rotation=30, ha='right')
ax_cm.set_yticks(range(4))
ax_cm.set_yticklabels(genera, fontsize=6)
ax_cm.set_xlabel('Predicted', fontsize=8)
ax_cm.set_ylabel('True', fontsize=8)
ax_cm.set_title('Confusion Matrix (RF)', fontsize=9, fontweight='bold', pad=8)

# Colorbar
cbar = fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04, shrink=0.85)
cbar.ax.tick_params(labelsize=5.5)
cbar.set_label('Proportion', fontsize=6)


# ════════════════════════════════════════════════════════════════════════
# PANEL B: Per-Class F1 Scores
# ════════════════════════════════════════════════════════════════════════
ax_f1.text(-0.08, 1.12, 'B', transform=ax_f1.transAxes,
           fontsize=12, fontweight='bold', va='top')

lr_f1 = per_class[per_class['Model'] == 'Logistic Regression'].set_index('Genus')
rf_f1 = per_class[per_class['Model'] == 'Random Forest'].set_index('Genus')

x_f1 = np.arange(4)
w_f1 = 0.35

lr_vals_f1 = [lr_f1.loc[g, 'F1_Score'] for g in genera]
rf_vals_f1 = [rf_f1.loc[g, 'F1_Score'] for g in genera]

ax_f1.bar(x_f1 - w_f1/2, lr_vals_f1, w_f1, color=LR_COLOR, edgecolor='white',
          linewidth=0.3, label='Logistic Regression', zorder=2)
ax_f1.bar(x_f1 + w_f1/2, rf_vals_f1, w_f1, color=RF_COLOR, edgecolor='white',
          linewidth=0.3, label='Random Forest', zorder=2)

# Value labels
for i in range(4):
    ax_f1.text(x_f1[i] - w_f1/2, lr_vals_f1[i] + 0.001, f'{lr_vals_f1[i]:.3f}',
               ha='center', va='bottom', fontsize=4.8, color=LR_COLOR)
    ax_f1.text(x_f1[i] + w_f1/2, rf_vals_f1[i] + 0.001, f'{rf_vals_f1[i]:.3f}',
               ha='center', va='bottom', fontsize=4.8, color=RF_COLOR)

# Random baseline
ax_f1.axhline(y=0.25, color='#999', linewidth=0.6, linestyle='--', zorder=1)
ax_f1.text(3.6, 0.255, 'Random\nbaseline', fontsize=4.5, color='#999', va='bottom')

ax_f1.set_xticks(x_f1)
ax_f1.set_xticklabels(genera, fontsize=6, fontstyle='italic')
ax_f1.set_ylabel('F1 Score', fontsize=8)
ax_f1.set_title('Per-Class F1 Scores', fontsize=9, fontweight='bold', pad=8)
ax_f1.set_ylim(0.2, 1.06)
ax_f1.spines['top'].set_visible(False)
ax_f1.spines['right'].set_visible(False)

# Place the model key in the inter-panel whitespace, clear of the plotted bars.
leg_f1 = ax_f1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2,
                       frameon=True, framealpha=0.95, edgecolor='#CCC', fontsize=5.5, handlelength=1.0)
leg_f1.get_frame().set_linewidth(0.5)


# ════════════════════════════════════════════════════════════════════════
# PANEL C: PCA
# ════════════════════════════════════════════════════════════════════════
ax_pca.text(-0.08, 1.12, 'C', transform=ax_pca.transAxes,
            fontsize=12, fontweight='bold', va='top')

pc1_var = pca_var.loc[pca_var['Component']=='PC1', 'Variance_Explained_Percent'].values[0]
pc2_var = pca_var.loc[pca_var['Component']=='PC2', 'Variance_Explained_Percent'].values[0]

for genus in genera:
    gdata = pca_coords[pca_coords['genus'] == genus]
    ax_pca.scatter(gdata['PC1'], gdata['PC2'],
                   s=5, alpha=0.5, color=genus_colors[genus],
                   label=genus, edgecolors='none',
                   zorder=2, rasterized=True)

ax_pca.set_xlabel(f'PC1 ({pc1_var:.1f}%)', fontsize=8)
ax_pca.set_ylabel(f'PC2 ({pc2_var:.1f}%)', fontsize=8)
ax_pca.set_title('PCA of Compositional Features', fontsize=9, fontweight='bold', pad=8)
ax_pca.spines['top'].set_visible(False)
ax_pca.spines['right'].set_visible(False)

leg_pca = ax_pca.legend(loc='upper right', frameon=True, framealpha=0.95,
                         edgecolor='#CCC', fontsize=5.5, handlelength=0.8,
                         markerscale=1.5, scatterpoints=1)
leg_pca.get_frame().set_linewidth(0.5)
# Italicize genus names in legend
for text in leg_pca.get_texts():
    text.set_fontstyle('italic')


# ════════════════════════════════════════════════════════════════════════
# PANEL D: Top 15 RF Feature Importances
# ════════════════════════════════════════════════════════════════════════
ax_imp.text(-0.08, 1.12, 'D', transform=ax_imp.transAxes,
            fontsize=12, fontweight='bold', va='top')

rf_sorted = rf_top15.sort_values('Rank', ascending=False).reset_index(drop=True)
y_imp = np.arange(len(rf_sorted))

def clean_name(feat):
    name = feat.replace('RSCU_', '').replace('_freq', ' freq').replace('_OE', ' O/E')
    name = name.replace('genome_length', 'Genome length').replace('gc_content', 'GC content')
    name = name.replace('UpA', 'UpA').replace('CpG', 'CpG')
    return name

names_imp = [clean_name(f) for f in rf_sorted['Feature']]

# Color by feature type
def feat_color(f):
    if f.startswith('RSCU_'):
        return '#FF7F0E'
    elif f in ('genome_length', 'gc_content'):
        return '#2CA02C'
    else:
        return '#9467BD'

colors_imp = [feat_color(f) for f in rf_sorted['Feature']]

ax_imp.barh(y_imp, rf_sorted['Importance'], height=0.7, color=colors_imp,
            edgecolor='none', zorder=2)

ax_imp.set_yticks(y_imp)
ax_imp.set_yticklabels(names_imp, fontsize=6)
ax_imp.set_xlabel('Feature Importance (MDI)', fontsize=7.5)
ax_imp.set_title('Top 15 RF Feature Importances', fontsize=9, fontweight='bold', pad=8)
ax_imp.spines['top'].set_visible(False)
ax_imp.spines['right'].set_visible(False)
ax_imp.tick_params(axis='y', length=0, pad=3)

leg_imp = ax_imp.legend(
    handles=[Patch(facecolor='#9467BD', label='Dinucleotide'),
             Patch(facecolor='#2CA02C', label='Compositional'),
             Patch(facecolor='#FF7F0E', label='RSCU')],
    loc='lower right', frameon=True, framealpha=0.95,
    edgecolor='#CCC', fontsize=5.5, handlelength=1.0)
leg_imp.get_frame().set_linewidth(0.5)

# Summary annotation
fig.text(0.5, 0.005,
         'Species-stratified 5-fold CV with balanced class weights; '
         f'RF balanced accuracy = 0.980 ± 0.020, macro F1 = 0.984 ± 0.013',
         ha='center', va='bottom', fontsize=6, fontstyle='italic', color='#555')


# ── Save ────────────────────────────────────────────────────────────────
import os
outputs = {
    'tiff': (OUT / 'Fig7_publication.tiff', 600),
    'pdf': (OUT / 'Fig7_publication.pdf', 600),
    'png': (OUT / 'Fig7_preview.png', 200),
}
for fmt, (path, dpi) in outputs.items():
    kw = dict(pil_kwargs={'compression': 'tiff_lzw'}) if fmt == 'tiff' else {}
    fig.savefig(path, dpi=dpi, format=fmt, bbox_inches='tight', pad_inches=0.08, **kw)

print("Figure 7 saved:")
for path, _ in outputs.values():
    mb = path.stat().st_size / (1024**2)
    print(f"  {path.name}: {mb:.1f} MB")
