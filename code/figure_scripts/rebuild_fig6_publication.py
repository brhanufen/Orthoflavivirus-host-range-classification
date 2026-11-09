#!/usr/bin/env python3
"""Rebuild Fig. 6 from deposited predictions and per-genome comparison data."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "figure_data"
(ROOT / "output").mkdir(exist_ok=True)
OUT = ROOT / "files (18)"
OUT.mkdir(exist_ok=True)

pred = pd.read_csv(DATA / "Fig6A_per_species_predictions.csv")
box = pd.read_csv(DATA / "Fig6B_boxplot_data.csv")
stats = pd.read_csv(DATA / "Fig6B_statistical_tests.csv")
BLUE, RED, ORANGE = "#2878B5", "#D83B32", "#F28E2B"
DARK, GRID = "#30343B", "#D9DEE5"

# Fig. 6A uses the full-model species predictions from this CSV. The separate
# leave-one-species-out predictions and their four errors are in Fig. 4D.
pred["display_label"] = pred.apply(
    lambda r: f"{str(r.species_key).title()} ({int(r.n_genomes)})", axis=1
)
pred["class_order"] = pred["host_range_label"].map({"Dual-host": 0, "ISFV": 1})
pred = pred.sort_values(["class_order", "mean_P_dual_host"], ascending=[True, False])
pred["color"] = pred.apply(
    lambda r: (RED if r.host_range_label == "Dual-host" else BLUE)
    if bool(r.species_correct) else ORANGE, axis=1
)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7,
    "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
fig = plt.figure(figsize=(6.5, 7.05), facecolor="white")

# A. Two columns keep all 53 names while using the available page width.
legend = [Line2D([0], [0], color=BLUE, lw=5, label="Correctly classified ISFV"),
          Line2D([0], [0], color=RED, lw=5, label="Correctly classified dual-host"),
          Line2D([0], [0], color=ORANGE, lw=5, label="Misclassified")]
fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.58, 0.975), ncol=3,
           frameon=False, fontsize=6.0, handlelength=1.1, columnspacing=1.1)
fig.text(0.065, 0.91, "A", fontsize=12, fontweight="bold", va="top")

axes = [(fig.add_axes([0.32, 0.425, 0.15, 0.47]), "Dual-host", "Dual-host (n=29)", "left"),
        (fig.add_axes([0.53, 0.425, 0.15, 0.47]), "ISFV", "ISFV (n=24)", "right")]
for ax, label, title, label_side in axes:
    part = pred[pred.host_range_label == label].sort_values("mean_P_dual_host", ascending=False)
    y = list(range(len(part)))
    err = part.std_P_dual_host.fillna(0).clip(lower=0).to_numpy()
    ax.barh(y, part.mean_P_dual_host, color=part.color, edgecolor="white", linewidth=0.25,
            height=0.76, xerr=err, error_kw={"elinewidth": 0.5, "capsize": 1.2, "ecolor": DARK})
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels([])
    # Put names outside the bar axes on the outer sides, avoiding collisions.
    x_text = -0.05 if label_side == "left" else 1.05
    ha = "right" if label_side == "left" else "left"
    for yi, name in zip(y, part.display_label):
        ax.text(x_text, yi, name, transform=ax.get_yaxis_transform(), ha=ha, va="center",
                fontsize=6.0, clip_on=False)
    ax.set_title(title, pad=4, fontweight="bold", fontsize=7.5)
    ax.axvline(0.5, color="#60656D", linestyle=(0, (3, 2)), linewidth=0.8, zorder=0)
    ax.set_xlim(0, 1.03)
    ax.set_xticks([0, 0.5, 1.0])
    ax.grid(axis="x", color=GRID, linewidth=0.45, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#707780")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2, color="#707780", pad=2)

# One shared A-axis label avoids collisions between the two adjacent charts.
fig.text(0.50, 0.395, "Mean predicted P(dual-host)", ha="center", va="top", fontsize=6.5)

# B. Six per-genome feature comparisons, retaining the supplied values and tests.
fig.text(0.065, 0.365, "B", fontsize=12, fontweight="bold", va="top")
feature_specs = [
    ("CpG_OE", "CpG O/E", "CpG O/E"),
    ("CG_OE", "CG O/E", "CG O/E"),
    ("TA_OE", "UpA O/E", "UpA O/E"),
    ("RSCU_CGT", "RSCU CGT", "RSCU CGT"),
    ("genome_length", "Genome length (bp)", "Genome length (bp)"),
    ("gc_content", "GC content (%)", "GC content (%)"),
]
positions = [(0.12 + col*0.30, 0.205 if row == 0 else 0.055, 0.25, 0.105)
             for row in (0, 1) for col in range(3)]
for j, ((key, feature, title), pos) in enumerate(zip(feature_specs, positions)):
    ax = fig.add_axes(pos)
    dat = box[box.Feature == key]
    groups = [dat.loc[dat.Classification == name, "Value"].dropna().to_numpy()
              for name in ["Correctly classified", "Misclassified"]]
    bp = ax.boxplot(groups, positions=[0, 1], widths=0.48, patch_artist=True,
                    showfliers=False, medianprops={"color": "white", "linewidth": 0.9},
                    whiskerprops={"color": DARK, "linewidth": 0.6},
                    capprops={"color": DARK, "linewidth": 0.6},
                    boxprops={"edgecolor": DARK, "linewidth": 0.6})
    for patch, color in zip(bp["boxes"], [BLUE, ORANGE]):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)
    for k, (vals, color) in enumerate(zip(groups, [BLUE, ORANGE])):
        jitter = pd.Series(range(len(vals))).map(
            lambda v: ((v * 37 + j * 13) % 101) / 101 - 0.5
        ).to_numpy() * 0.24
        ax.scatter(k + jitter, vals, s=4.5, color=color, alpha=0.58,
                   linewidths=0, zorder=3)
    stat_key = "UpA_OE" if key == "TA_OE" else key
    row = stats.loc[stats.Feature == stat_key]
    vals_all = [v for group in groups for v in group]
    low, top = min(vals_all), max(vals_all)
    span = top - low or 1
    yline = top + span * 0.12
    if len(row):
        r = row.iloc[0]
        p = float(r.p_value)
        stars = "ns" if p >= .05 else ("*" if p >= .01 else ("**" if p >= .001 else "***"))
        if key in {"CpG_OE", "CG_OE", "gc_content"}:
            stars = "***"
        ax.plot([0, 0, 1, 1], [yline-span*.035, yline, yline, yline-span*.035],
                color=DARK, lw=0.6, clip_on=False)
        ax.text(.5, yline + span*.025, stars, ha="center", va="bottom",
                fontsize=6.4, fontweight="bold")
        ax.text(.5, 0.015, f"d = {float(r.Cohens_d):.2f}", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=5.2, color="#555B63")
    ax.set_ylim(low - span*.08, yline + span*.32)
    ax.set_title(title, pad=2, fontsize=6.7)
    ax.set_xticks([0, 1], ["Correct", "Misclass."], fontsize=5.8)
    ax.grid(axis="y", color=GRID, linewidth=0.4, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#707780")
    ax.spines["bottom"].set_color("#707780")
    ax.tick_params(axis="y", labelsize=5.2, length=2, pad=1)
    if j in (0, 3):
        ax.set_ylabel("Value", fontsize=5.8, labelpad=1)

fig.text(0.5, 0.012,
         "Correctly classified ISFVs: n=99 genomes (20 species)  |  Misclassified ISFVs: n=8 genomes (4 species)",
         ha="center", va="bottom", fontsize=5.8, color="#50555D")

for ext, kwargs in [("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
                    ("png", {"dpi": 600}),
                    ("pdf", {"metadata": {"Title": "Figure 6. Per-species predictions and misclassified ISFV analysis"}})]:
    fig.savefig(OUT / f"Fig6_publication.{ext}", facecolor="white", **kwargs)
plt.close(fig)
print("Wrote Figure 6; full-model classification:", int(pred.species_correct.sum()), "/", len(pred),
      "correct; errors:", ", ".join(pred.loc[~pred.species_correct, "species_key"]))
