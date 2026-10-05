import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from panel import clade_colours, load_panel, motif_clade, motif_labels, tile

TILE_LEN = 600
WIN = 25
THRESHOLD = 1.0
FLANK = 35
C228T_REL_ATG = -124
C250T_REL_ATG = -146
VARIANT_COLOURS = {"WT": "black", "C228T": "#B85042", "C250T": "#2C5F2D", "C228T + C250T": "#065A82"}
PLOT_ORDER = ["C228T + C250T", "C228T", "C250T", "WT"]


def g4hunter_per_base(seq):
    seq = seq.upper()
    n = len(seq)
    scores = np.zeros(n)
    i = 0
    while i < n:
        b = seq[i]
        if b in "GC":
            run = 1
            while i + run < n and seq[i + run] == b:
                run += 1
            scores[i:i + run] = (1 if b == "G" else -1) * min(run, 4)
            i += run
        else:
            i += 1
    return scores


def g4hunter_window(seq, window=WIN):
    s = g4hunter_per_base(seq)
    if len(s) < window:
        return np.array([s.mean()])
    return np.convolve(s, np.ones(window) / window, mode="valid")


def telomeric_scores(df):
    per_species = df[["species", "motif", "clade"]].copy()
    per_species["g4h_mean"] = [g4hunter_per_base(tile(m, TILE_LEN)).mean() for m in df["motif"]]
    per_motif = (per_species.groupby("motif", sort=False)
                 .agg(n_species=("species", "size"),
                      species=("species", "; ".join),
                      clades=("clade", lambda c: "; ".join(sorted(set(c)))),
                      g4h_mean=("g4h_mean", "first"))
                 .reset_index()
                 .sort_values("g4h_mean", ascending=False))
    per_motif["above_threshold"] = per_motif["g4h_mean"] > THRESHOLD
    return per_species.sort_values("g4h_mean", ascending=False), per_motif


def htert_tracks(path="htert_promoter.json"):
    with open(path) as f:
        htert = json.load(f)
    plus = htert["plus_strand"]
    n = len(plus)
    i228 = n - 1 - htert["pos_C228T_minus"]
    i250 = n - 1 - htert["pos_C250T_minus"]
    if plus[i228] != "G" or plus[i250] != "G":
        raise ValueError("expected G at both hotspot positions on the plus strand")

    rel = C228T_REL_ATG - (np.arange(n) - i228)
    if rel[i250] != C250T_REL_ATG:
        raise ValueError("C250T does not map to -146 relative to the ATG")

    def mutate(seq, pos):
        return seq[:pos] + "A" + seq[pos + 1:]

    seqs = {"WT": plus, "C228T": mutate(plus, i228), "C250T": mutate(plus, i250),
            "C228T + C250T": mutate(mutate(plus, i228), i250)}
    tracks = {k: g4hunter_window(s) for k, s in seqs.items()}
    centre = rel[WIN // 2: WIN // 2 + len(tracks["WT"])]
    return htert, tracks, centre


def hotspot_table(tracks, centre):
    rows = []
    for site, pos in [("C228T", C228T_REL_ATG), ("C250T", C250T_REL_ATG)]:
        k = int(np.where(centre == pos)[0][0])
        for variant, track in tracks.items():
            rows.append({"site": site, "rel_atg": pos, "variant": variant,
                         "score_window_centred_on_site": track[k]})
    return pd.DataFrame(rows)


def label_for(motif, labels):
    label = labels[motif]
    if label.endswith("species"):
        return f"{motif}  ({label})"
    return f"{motif}  ($\\it{{{label.replace(' ', '~')}}}$)"


def plot_motifs(ax, per_motif, labels, clades):
    data = per_motif.sort_values("g4h_mean")
    colour = clade_colours(set(clades.values()))
    groups = list(colour)
    y = np.arange(len(data))
    ax.barh(y, data["g4h_mean"], color=[colour[clades[m]] for m in data["motif"]],
            edgecolor="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([label_for(m, labels) for m in data["motif"]], fontsize=10, family="monospace")
    ax.axvline(THRESHOLD, color="red", lw=0.8, ls="--", alpha=0.7)
    ax.set_xlabel("G4Hunter mean score (tiled repeat, G-rich strand)", fontsize=13)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colour[g], edgecolor="black") for g in groups]
    ax.legend(handles, groups, loc="lower right", fontsize=10, frameon=True,
              edgecolor="black", title="Clade", title_fontsize=11)


def plot_htert(ax, tracks, centre, zoom=False):
    lo, hi = C250T_REL_ATG - FLANK, C228T_REL_ATG + FLANK
    mask = (centre >= lo) & (centre <= hi) if zoom else np.ones_like(centre, dtype=bool)
    for label in PLOT_ORDER:
        wt = label == "WT"
        ax.plot(centre[mask], tracks[label][mask], label=label, color=VARIANT_COLOURS[label],
                lw=3.0 if wt else 1.5, alpha=1.0 if wt else 0.85, zorder=5 if wt else 3,
                marker="o" if zoom else None, markersize=4)
    ax.axhline(THRESHOLD, color="red", lw=0.6, ls="--", alpha=0.6)
    if not zoom:
        ax.axhline(-THRESHOLD, color="red", lw=0.6, ls="--", alpha=0.6)
    for pos, name, col, ha in [(C228T_REL_ATG, "C228T", "#d95f02", "left"),
                               (C250T_REL_ATG, "C250T", "#1b9e77", "right")]:
        ax.axvline(pos, color=col, lw=0.8, ls=":", alpha=0.9)
        ax.text(pos, 1.01, name, transform=ax.get_xaxis_transform(), color=col,
                fontsize=11, va="bottom", ha=ha, fontweight="bold")
    ax.set_xlabel("Position relative to the TERT ATG (bp)", fontsize=13)
    ax.set_ylabel(f"G4Hunter score ({WIN}-nt window)", fontsize=13)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=10, frameon=True,
              edgecolor="black", title="Variant", title_fontsize=11)


def save(fig, stem):
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    df = load_panel()
    per_species, per_motif = telomeric_scores(df)
    per_species.to_csv("results/g4hunter_telomeric_species.csv", index=False, float_format="%.3f")
    per_motif.to_csv("results/g4hunter_telomeric_motifs.csv", index=False, float_format="%.3f")

    htert, tracks, centre = htert_tracks()
    pd.DataFrame({"rel_atg": centre, **tracks}).to_csv(
        "results/htert_g4hunter_tracks.csv", index=False, float_format="%.4f")
    hotspots = hotspot_table(tracks, centre)
    hotspots.to_csv("results/htert_hotspot_scores.csv", index=False, float_format="%.3f")

    labels, clades = motif_labels(df), motif_clade(df)

    fig, ax = plt.subplots(figsize=(11, 8))
    plot_motifs(ax, per_motif, labels, clades)
    save(fig, "figures/Fig_g4hunter_motifs")

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.85], hspace=0.35)
    plot_htert(fig.add_subplot(gs[0]), tracks, centre)
    plot_htert(fig.add_subplot(gs[1]), tracks, centre, zoom=True)
    save(fig, "figures/Fig_htert_promoter")
    fig, ax = plt.subplots(figsize=(14, 5))
    plot_htert(ax, tracks, centre)
    save(fig, "figures/Fig_htert_promoter_full")

    fig, ax = plt.subplots(figsize=(14, 4.5))
    plot_htert(ax, tracks, centre, zoom=True)
    save(fig, "figures/Fig_htert_promoter_zoom")

    fig = plt.figure(figsize=(14, 17))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.3, 1, 0.85], hspace=0.35)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    plot_motifs(axes[0], per_motif, labels, clades)
    plot_htert(axes[1], tracks, centre)
    plot_htert(axes[2], tracks, centre, zoom=True)
    for ax, letter in zip(axes, "ABC"):
        ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=18, fontweight="bold",
                ha="right", va="bottom")
    save(fig, "figures/Fig_insilico_g4")

    print(per_motif[["motif", "n_species", "g4h_mean", "above_threshold"]].to_string(index=False))
    print()
    print(hotspots.pivot(index="variant", columns="site",
                         values="score_window_centred_on_site").round(3).to_string())


if __name__ == "__main__":
    main()
