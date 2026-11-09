#!/usr/bin/env python3
"""
Figure 1. Dataset overview and curation workflow.
Publication-quality figure for Viruses (MDPI).
Uses exact data from Fig1_Data CSVs.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
import numpy as np
import csv, os
from pathlib import Path
from zipfile import ZipFile
from io import TextIOWrapper
from PIL import Image

# ── ASM-compliant settings ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 7,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
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

# ── Load real data ──────────────────────────────────────────────────────
def title_case_species(s):
    """Convert species_key to display name."""
    # Title case each word, preserving known abbreviations
    words = s.strip().split()
    result = []
    for w in words:
        if w.lower() in ('ho',):
            result.append(w.capitalize())
        elif w.lower() == 't':
            result.append("T'")  # T'Ho virus
        else:
            result.append(w.capitalize())
    name = ' '.join(result)
    # Fix T' Ho -> T'Ho
    name = name.replace("T' Ho", "T'Ho")
    return name

# Read species distribution
dual_species = []
isfv_species = []
DATA_ARCHIVE = Path(__file__).resolve().parents[1] / 'Figure1_Data.zip'
OUTPUT_DIR = Path(__file__).resolve().parent

with ZipFile(DATA_ARCHIVE) as data_zip:
    with data_zip.open('Fig1B_species_distribution.csv') as raw:
        rows = list(csv.DictReader(TextIOWrapper(raw, encoding='utf-8')))
    for row in rows:
        name = title_case_species(row['species_key'])
        count = int(row['genome_count'])
        if row['host_range_label'] == 'dual_host':
            dual_species.append((name, count))
        else:
            isfv_species.append((name, count))

# Sort each group by count (descending)
dual_species.sort(key=lambda x: x[1], reverse=True)
isfv_species.sort(key=lambda x: x[1], reverse=True)

# Read split composition
splits = {}
with ZipFile(DATA_ARCHIVE) as data_zip:
    with data_zip.open('Fig1C_split_composition.csv') as raw:
        rows = list(csv.DictReader(TextIOWrapper(raw, encoding='utf-8')))
    for row in rows:
        splits[row['split']] = {
            'dual': int(row['dual_host_genomes']),
            'isfv': int(row['isfv_genomes']),
            'total': int(row['total_genomes']),
            'species': int(row['n_species'])
        }

# Verify
total_dual = sum(c for _, c in dual_species)
total_isfv = sum(c for _, c in isfv_species)
print(f"Dual-host: {total_dual} genomes, {len(dual_species)} species")
print(f"ISFV: {total_isfv} genomes, {len(isfv_species)} species")
print(f"Total: {total_dual + total_isfv} genomes, {len(dual_species)+len(isfv_species)} species")

# ── Figure layout ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(6.875, 9.0))

gs = gridspec.GridSpec(
    2, 2, figure=fig,
    width_ratios=[0.33, 0.67],
    height_ratios=[0.62, 0.38],
    hspace=0.25, wspace=0.22,
)

ax_flow  = fig.add_subplot(gs[0, 0])
ax_sp    = fig.add_subplot(gs[0, 1])
ax_split = fig.add_subplot(gs[1, 0])
ax_table = fig.add_subplot(gs[1, 1])

# ════════════════════════════════════════════════════════════════════════
# PANEL A: Flowchart
# ════════════════════════════════════════════════════════════════════════
ax_flow.set_xlim(0, 10)
ax_flow.set_ylim(0, 16)
ax_flow.axis('off')
ax_flow.text(-0.02, 1.07, 'A', transform=ax_flow.transAxes,
             fontsize=12, fontweight='bold', va='top')

def add_box(ax, x, y, w, h, text, fc='white', ec='black',
            fs=5.5, fw='normal', tc='black', lw=0.7):
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.10",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2, clip_on=True
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            fontweight=fw, color=tc, zorder=3, linespacing=1.2, clip_on=True)

def arrow_d(ax, x, y1, y2):
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='->', color='#555', lw=0.8))

cx = 3.5

# 1. NCBI Query
add_box(ax_flow, cx, 15.2, 6.5, 1.0,
        'NCBI GenBank Query\nFlaviviridae (all genera)',
        fc='#E3F2FD', ec='#1565C0', fs=5.5, fw='bold')
arrow_d(ax_flow, cx, 14.7, 14.0)

# 2. Records
add_box(ax_flow, cx, 13.5, 5.5, 0.8,
        'Records retrieved: n = 4,634',
        fc='#F5F5F5', ec='#999', fs=5.3)
arrow_d(ax_flow, cx, 13.1, 12.4)

# 3. Filter
add_box(ax_flow, cx, 11.8, 6.8, 1.0,
        'Filter: Flavivirus genus only\nComplete genomes + CDS\n'
        '(excl. non-Flavivirus, segmented, partial)',
        fc='#FFF3E0', ec='#E65100', fs=4.6)
arrow_d(ax_flow, cx, 11.3, 10.55)

# 4. Host range
add_box(ax_flow, cx, 10.0, 6.8, 0.9,
        'Host range labeling (evidence-based)\n'
        '(excl. ambiguous host, unverified)',
        fc='#FFF3E0', ec='#E65100', fs=4.6)
arrow_d(ax_flow, cx, 9.55, 8.55)

# 5. Final dataset
add_box(ax_flow, cx, 7.95, 6.5, 1.1,
        'Final Dataset\n1,285 genomes | 53 species',
        fc='#E8F5E9', ec='#2E7D32', fs=6.0, fw='bold')

# Split arrows
arrow_d(ax_flow, 1.8, 7.4, 6.55)
arrow_d(ax_flow, 5.2, 7.4, 6.55)

# Dual-host
add_box(ax_flow, 1.8, 5.85, 3.0, 1.3,
        'Dual-host\n1,178 genomes\n29 species',
        fc='#FFCDD2', ec=DUAL_COLOR, fs=5.3, fw='bold', tc='#B71C1C')

# ISFV
add_box(ax_flow, 5.2, 5.85, 3.0, 1.3,
        'ISFV\n107 genomes\n24 species',
        fc='#BBDEFB', ec=ISFV_COLOR, fs=5.3, fw='bold', tc='#0D47A1')


# ════════════════════════════════════════════════════════════════════════
# PANEL B: Species Distribution (REAL DATA)
# ════════════════════════════════════════════════════════════════════════
ax_sp.text(-0.02, 1.07, 'B', transform=ax_sp.transAxes,
           fontsize=12, fontweight='bold', va='top')

# ISFV at bottom, dual-host at top (reversed for barh)
all_names  = [s for s, _ in reversed(isfv_species)] + [s for s, _ in reversed(dual_species)]
all_counts = [c for _, c in reversed(isfv_species)] + [c for _, c in reversed(dual_species)]
all_colors = [ISFV_COLOR]*len(isfv_species) + [DUAL_COLOR]*len(dual_species)

y_pos = np.arange(len(all_names))
ax_sp.barh(y_pos, all_counts, height=0.72, color=all_colors, edgecolor='none', zorder=2)

# Separator line
sep_y = len(isfv_species) - 0.5
ax_sp.axhline(y=sep_y, color='black', linewidth=0.8, zorder=3)

ax_sp.set_yticks(y_pos)
ax_sp.set_yticklabels(all_names, fontsize=4.8)
ax_sp.set_xlabel('Number of genomes', fontsize=7.5)
ax_sp.set_title('Species Distribution', fontsize=9, fontweight='bold', pad=6)
ax_sp.set_xlim(0, max(all_counts) * 1.08)
ax_sp.set_ylim(-0.7, len(all_names) - 0.3)
ax_sp.spines['top'].set_visible(False)
ax_sp.spines['right'].set_visible(False)
ax_sp.tick_params(axis='y', length=0, pad=2)
ax_sp.grid(axis='x', alpha=0.15, linewidth=0.3, zorder=0)

# Class labels on right margin
dual_mid = len(isfv_species) + len(dual_species) / 2
isfv_mid = len(isfv_species) / 2
ax_sp.text(max(all_counts)*1.06, dual_mid, 'Dual-host',
           fontsize=7, color=DUAL_COLOR, fontweight='bold',
           va='center', ha='left', rotation=270)
ax_sp.text(max(all_counts)*1.06, isfv_mid, 'ISFV',
           fontsize=7, color=ISFV_COLOR, fontweight='bold',
           va='center', ha='left', rotation=270)


# ════════════════════════════════════════════════════════════════════════
# PANEL C: Species-Stratified Data Splits (REAL DATA)
# ════════════════════════════════════════════════════════════════════════
ax_split.text(-0.02, 1.07, 'C', transform=ax_split.transAxes,
              fontsize=12, fontweight='bold', va='top')

partitions = ['Train', 'Validation', 'Test']
dual_c = [splits['train']['dual'], splits['val']['dual'], splits['test']['dual']]
isfv_c = [splits['train']['isfv'], splits['val']['isfv'], splits['test']['isfv']]
totals = [splits['train']['total'], splits['val']['total'], splits['test']['total']]

x = np.arange(len(partitions))
w = 0.52

ax_split.bar(x, dual_c, w, color=DUAL_COLOR, edgecolor='white',
             linewidth=0.3, label='Dual-host', zorder=2)
ax_split.bar(x, isfv_c, w, bottom=dual_c, color=ISFV_COLOR,
             edgecolor='white', linewidth=0.3, label='ISFV', zorder=2)

for i in range(3):
    # Total above bar
    ax_split.text(x[i], totals[i] + 18, f'{totals[i]}',
                  ha='center', va='bottom', fontsize=6.5, fontweight='bold')
    # Dual count inside (if bar is tall enough)
    if dual_c[i] > 60:
        ax_split.text(x[i], dual_c[i]/2, f'{dual_c[i]}',
                      ha='center', va='center', fontsize=5.5,
                      color='white', fontweight='bold')
    # ISFV count
    if isfv_c[i] >= 15:
        ax_split.text(x[i], dual_c[i] + isfv_c[i]/2, f'{isfv_c[i]}',
                      ha='center', va='center', fontsize=5.5,
                      color='white', fontweight='bold')
    elif isfv_c[i] > 0:
        ax_split.text(x[i]+0.32, dual_c[i] + isfv_c[i]/2, f'{isfv_c[i]}',
                      ha='left', va='center', fontsize=5, color=ISFV_COLOR)

ax_split.set_xticks(x)
ax_split.set_xticklabels(partitions, fontsize=7.5)
ax_split.set_ylabel('Number of genomes', fontsize=7.5)
ax_split.set_title('Species-Stratified Data Splits', fontsize=9, fontweight='bold', pad=8, loc='left', x=0.10)
ax_split.set_ylim(0, 1350)
ax_split.spines['top'].set_visible(False)
ax_split.spines['right'].set_visible(False)

leg = ax_split.legend(loc='upper right', frameon=True, framealpha=0.95,
                       edgecolor='#CCC', fontsize=6.5, handlelength=1.0)
leg.get_frame().set_linewidth(0.5)


# ════════════════════════════════════════════════════════════════════════
# PANEL D: Dataset Summary
# ════════════════════════════════════════════════════════════════════════
ax_table.axis('off')
ax_table.text(-0.02, 1.07, 'D', transform=ax_table.transAxes,
              fontsize=12, fontweight='bold', va='top')

box = FancyBboxPatch(
    (0.02, 0.02), 0.96, 0.96,
    boxstyle="round,pad=0.02",
    facecolor='#FAFAFA', edgecolor='#BBBBBB', linewidth=0.6,
    transform=ax_table.transAxes, zorder=0
)
ax_table.add_patch(box)

ax_table.text(0.50, 0.95, 'Dataset Summary', transform=ax_table.transAxes,
              fontsize=8, fontweight='bold', ha='center', va='top')
ax_table.plot([0.06, 0.94], [0.90, 0.90], color='#CCCCCC', linewidth=0.5,
              transform=ax_table.transAxes)

rows = [
    ('Total genomes:', '1,285', False),
    ('Total species:', '53', False),
    (None, None, None),
    ('HOST RANGE CLASSES', '', True),
    ('  Dual-host:', '1,178 genomes (29 spp.)', False),
    ('  ISFV:', '107 genomes (24 spp.)', False),
    ('  Class imbalance:', '~11:1', False),
    (None, None, None),
    ('DATA PARTITIONS', '', True),
    ('  Train:', '1,148 (89.3%)', False),
    ('  Validation:', '17 (1.3%)', False),
    ('  Test:', '120 (9.3%)', False),
    (None, None, None),
    ('FEATURES (n = 97)', '', True),
    ('  Compositional:', '2', False),
    ('  Dinucleotide:', '34', False),
    ('  RSCU:', '61', False),
]

y = 0.86
dy = 0.045
for label, value, is_section in rows:
    if label is None:
        y -= dy * 0.35
        continue
    fw = 'bold' if is_section else 'normal'
    fs = 6.2 if is_section else 6.0
    ax_table.text(0.06, y, label, transform=ax_table.transAxes,
                  fontsize=fs, fontweight=fw, va='top', color='#333')
    if value:
        ax_table.text(0.94, y, value, transform=ax_table.transAxes,
                      fontsize=fs, va='top', ha='right', color='#333')
    y -= dy


# ── Save ────────────────────────────────────────────────────────────────
for fmt, dpi, ext in [('tiff', 600, 'tiff'), ('pdf', 600, 'pdf'), ('png', 200, 'png')]:
    fname = OUTPUT_DIR / f'Fig1_publication.{ext}'
    kw = dict(pil_kwargs={'compression': 'tiff_lzw'}) if fmt == 'tiff' else {}
    fig.savefig(fname, dpi=dpi, format=fmt, bbox_inches='tight', pad_inches=0.05, facecolor='white', transparent=False, **kw)
    if fmt == 'tiff':
        with Image.open(fname) as img:
            img.convert('RGB').save(fname, format='TIFF', compression='tiff_lzw', dpi=(dpi, dpi))

print("\n✓ Figure 1 (real data) saved:")
for ext in ['tiff', 'pdf', 'png']:
    mb = os.path.getsize(OUTPUT_DIR / f'Fig1_publication.{ext}') / (1024**2)
    print(f"  Fig1_publication.{ext}: {mb:.1f} MB")
