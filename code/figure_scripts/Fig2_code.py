#!/usr/bin/env python3
"""
Figure 2. Feature distributions reveal significant compositional differences
between host range classes.
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
DUAL_LIGHT = '#D6272833'  # with alpha
ISFV_LIGHT = '#1F77B433'

# ── Load data ───────────────────────────────────────────────────────────
gc_df = pd.read_csv('figure_data/Fig2A_gc_length_data.csv')
cpg_df = pd.read_csv('figure_data/Fig2B_cpg_upa_data.csv')
hm_df = pd.read_csv('figure_data/Fig2C_dinucleotide_heatmap_pivot.csv')
pca_df = pd.read_csv('figure_data/Fig2D_pca_coordinates.csv')
pca_var = pd.read_csv('figure_data/Fig2D_pca_variance.csv')
stats_df = pd.read_csv('figure_data/Fig2_statistical_tests.csv')

# Split data by class
gc_dual = gc_df[gc_df['host_range_label'] == 'dual_host']
gc_isfv = gc_df[gc_df['host_range_label'] == 'invertebrate_only']
cpg_dual = cpg_df[cpg_df['host_range_label'] == 'dual_host']
cpg_isfv = cpg_df[cpg_df['host_range_label'] == 'invertebrate_only']
pca_dual = pca_df[pca_df['host_range_label'] == 'dual_host']
pca_isfv = pca_df[pca_df['host_range_label'] == 'invertebrate_only']

# PCA variance
pc1_var = pca_var.iloc[0]['Variance_Explained_Percent']
pc2_var = pca_var.iloc[1]['Variance_Explained_Percent']
total_var = pc1_var + pc2_var

# Stats
stats = {}
for _, row in stats_df.iterrows():
    stats[row['Feature']] = {
        'dual_mean': row['Dual_host_mean'], 'dual_std': row['Dual_host_std'],
        'isfv_mean': row['ISFV_mean'], 'isfv_std': row['ISFV_std'],
        'd': row['Cohens_d'], 'effect': row['Effect_size']
    }

# ── Figure layout ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(6.875, 6.5))

# Top row: A (2 subplots) | B (2 subplots)
# Bottom row: C (heatmap) | D (PCA)
gs = gridspec.GridSpec(
    2, 2, figure=fig,
    width_ratios=[1, 1],
    height_ratios=[1, 1],
    hspace=0.38, wspace=0.35,
)

# Top-left: Panel A with 2 subplots (GC content, Genome length)
gs_a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0], wspace=0.45)
ax_gc = fig.add_subplot(gs_a[0, 0])
ax_len = fig.add_subplot(gs_a[0, 1])

# Top-right: Panel B with 2 subplots (CpG, UpA)
gs_b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], wspace=0.45)
ax_cpg = fig.add_subplot(gs_b[0, 0])
ax_upa = fig.add_subplot(gs_b[0, 1])

# Bottom-left: Panel C (heatmap)
ax_hm = fig.add_subplot(gs[1, 0])

# Bottom-right: Panel D (PCA)
ax_pca = fig.add_subplot(gs[1, 1])


# ── Helper: Publication box plot with jittered points ───────────────────
def pub_boxplot(ax, data_dual, data_isfv, ylabel, title, cohen_d,
                show_signif=True, ylim=None, dashed_line=None):
    """Create a clean box plot with individual points."""
    positions = [0, 1]
    labels = ['Dual-host', 'ISFV']
    colors = [DUAL_COLOR, ISFV_COLOR]

    for i, (data, color, pos) in enumerate(zip(
            [data_dual, data_isfv], colors, positions)):
        # Box plot
        bp = ax.boxplot([data], positions=[pos], widths=0.45,
                        patch_artist=True, showfliers=False, zorder=3)
        bp['boxes'][0].set(facecolor=color, alpha=0.25, edgecolor=color, linewidth=0.8)
        bp['medians'][0].set(color=color, linewidth=1.2)
        bp['whiskers'][0].set(color=color, linewidth=0.7)
        bp['whiskers'][1].set(color=color, linewidth=0.7)
        bp['caps'][0].set(color=color, linewidth=0.7)
        bp['caps'][1].set(color=color, linewidth=0.7)

        # Jittered points
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(data))
        ax.scatter(pos + jitter, data, s=3, alpha=0.3, color=color,
                   edgecolors='none', zorder=2, rasterized=True)

    # Dashed reference line
    if dashed_line is not None:
        ax.axhline(y=dashed_line, color='#999', linewidth=0.5, linestyle='--', zorder=1)

    # Significance bracket
    if show_signif:
        if ylim:
            yrange = ylim[1] - ylim[0]
            bracket_y = ylim[1] - yrange * 0.12
        else:
            yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
            bracket_y = ax.get_ylim()[1] - yrange * 0.12
        ax.plot([0, 0, 1, 1], [bracket_y, bracket_y + yrange*0.02,
                bracket_y + yrange*0.02, bracket_y],
                color='black', linewidth=0.6, zorder=4)
        ax.text(0.5, bracket_y + yrange*0.03, '***',
                ha='center', va='bottom', fontsize=8, fontweight='bold', zorder=4)

    # Cohen's d annotation
    d_val = cohen_d
    d_str = f'd = {abs(d_val):.2f}'
    if d_val < 0:
        d_str = f'd = {d_val:.2f}'
    ax.text(0.5, 0.02, d_str, transform=ax.transAxes,
            ha='center', va='bottom', fontsize=6, fontstyle='italic', color='#555')

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel(ylabel, fontsize=7.5)
    ax.set_title(title, fontsize=8, fontweight='bold', pad=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ylim:
        ax.set_ylim(ylim)


# ════════════════════════════════════════════════════════════════════════
# PANEL A: GC Content + Genome Length
# ════════════════════════════════════════════════════════════════════════
ax_gc.text(-0.25, 1.15, 'A', transform=ax_gc.transAxes,
           fontsize=12, fontweight='bold', va='top')

pub_boxplot(ax_gc,
            gc_dual['gc_content'].values,
            gc_isfv['gc_content'].values,
            'GC content (%)', 'GC Content',
            cohen_d=stats['GC content']['d'],
            ylim=(44, 58))

pub_boxplot(ax_len,
            gc_dual['genome_length'].values,
            gc_isfv['genome_length'].values,
            'Genome length (nt)', 'Genome Length',
            cohen_d=stats['Genome length']['d'],
            ylim=(9800, 11500))


# ════════════════════════════════════════════════════════════════════════
# PANEL B: CpG and UpA Suppression
# ════════════════════════════════════════════════════════════════════════
ax_cpg.text(-0.25, 1.15, 'B', transform=ax_cpg.transAxes,
            fontsize=12, fontweight='bold', va='top')

pub_boxplot(ax_cpg,
            cpg_dual['CpG_OE'].values,
            cpg_isfv['CpG_OE'].values,
            'CpG O/E ratio', 'CpG Suppression',
            cohen_d=stats['CpG O/E']['d'],
            ylim=(0.25, 1.18),
            dashed_line=1.0)

pub_boxplot(ax_upa,
            cpg_dual['UpA_OE'].values,
            cpg_isfv['UpA_OE'].values,
            'UpA O/E ratio', 'UpA Suppression',
            cohen_d=stats['UpA O/E']['d'],
            ylim=(0.3, 1.15),
            dashed_line=1.0)


# ════════════════════════════════════════════════════════════════════════
# PANEL C: Dinucleotide O/E Heatmap
# ════════════════════════════════════════════════════════════════════════
ax_hm.text(-0.08, 1.12, 'C', transform=ax_hm.transAxes,
           fontsize=12, fontweight='bold', va='top')

dinucs = hm_df['dinucleotide'].tolist()
dual_vals = hm_df['dual_host'].values
isfv_vals = hm_df['invertebrate_only'].values

# Build 2-row matrix: row 0 = dual-host, row 1 = ISFV
hm_matrix = np.array([dual_vals, isfv_vals])

# Use diverging colormap centered at 1.0
from matplotlib.colors import TwoSlopeNorm
norm = TwoSlopeNorm(vmin=0.4, vcenter=1.0, vmax=1.4)

im = ax_hm.imshow(hm_matrix, aspect='auto', cmap='RdBu_r', norm=norm, zorder=2)

# Annotate cells - use smaller font, 2 decimal places
for i in range(2):
    for j in range(len(dinucs)):
        val = hm_matrix[i, j]
        text_color = 'white' if abs(val - 1.0) > 0.35 else 'black'
        ax_hm.text(j, i, f'{val:.2f}', ha='center', va='center',
                   fontsize=4.0, color=text_color, zorder=3)

ax_hm.set_xticks(np.arange(len(dinucs)))
ax_hm.set_xticklabels(dinucs, fontsize=5.8, rotation=45, ha='right')
ax_hm.set_yticks([0, 1])
ax_hm.set_yticklabels(['Dual-host', 'ISFV'], fontsize=7)
ax_hm.set_title('Dinucleotide O/E Ratios', fontsize=9, fontweight='bold', pad=8)
ax_hm.tick_params(axis='x', length=0, pad=2)
ax_hm.tick_params(axis='y', length=0, pad=3)

# Colorbar
cbar = fig.colorbar(im, ax=ax_hm, shrink=0.55, aspect=12, pad=0.03)
cbar.set_label('O/E ratio', fontsize=6.5)
cbar.ax.tick_params(labelsize=5.5)


# ════════════════════════════════════════════════════════════════════════
# PANEL D: PCA of RSCU
# ════════════════════════════════════════════════════════════════════════
ax_pca.text(-0.18, 1.12, 'D', transform=ax_pca.transAxes,
            fontsize=12, fontweight='bold', va='top')

# Plot dual-host first (larger group, background)
ax_pca.scatter(pca_dual['PC1'], pca_dual['PC2'],
               s=8, alpha=0.4, color=DUAL_COLOR, edgecolors='none',
               label='Dual-host', zorder=2, rasterized=True)
# Plot ISFV on top
ax_pca.scatter(pca_isfv['PC1'], pca_isfv['PC2'],
               s=12, alpha=0.6, color=ISFV_COLOR, edgecolors='none',
               label='ISFV', zorder=3, rasterized=True)

ax_pca.set_xlabel(f'PC1 ({pc1_var:.1f}%)', fontsize=8)
ax_pca.set_ylabel(f'PC2 ({pc2_var:.1f}%)', fontsize=8)
ax_pca.set_title('RSCU Principal Component Analysis', fontsize=9, fontweight='bold', pad=8)
ax_pca.spines['top'].set_visible(False)
ax_pca.spines['right'].set_visible(False)

# Total variance annotation
ax_pca.text(0.03, 0.03, f'Total variance: {total_var:.1f}%',
            transform=ax_pca.transAxes, fontsize=6, fontstyle='italic', color='#555')

leg = ax_pca.legend(loc='upper right', frameon=True, framealpha=0.95,
                     edgecolor='#CCC', fontsize=6.5, markerscale=1.5,
                     handletextpad=0.3)
leg.get_frame().set_linewidth(0.5)

# ── Footer annotation ──────────────────────────────────────────────────
fig.text(0.5, -0.01,
         'Mann-Whitney U test: ***p < 0.001, **p < 0.01, *p < 0.05, ns = not significant. '
         'd = Cohen\'s d effect size.',
         ha='center', va='top', fontsize=5.5, fontstyle='italic', color='#555')

# ── Save ────────────────────────────────────────────────────────────────
import os
for fmt, dpi, ext in [('tiff', 600, 'tiff'), ('pdf', 600, 'pdf'), ('png', 200, 'png')]:
    fname = f'figure_data/Fig2_final.{ext}'
    kw = dict(pil_kwargs={'compression': 'tiff_lzw'}) if fmt == 'tiff' else {}
    fig.savefig(fname, dpi=dpi, format=fmt, bbox_inches='tight', pad_inches=0.08, **kw)

print("✓ Figure 2 saved:")
for ext in ['tiff', 'pdf', 'png']:
    mb = os.path.getsize(f'figure_data/Fig2_final.{ext}') / (1024**2)
    print(f"  Fig2_final.{ext}: {mb:.1f} MB")
