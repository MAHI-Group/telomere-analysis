import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- G4Hunter ----------
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
            val = min(run, 4)
            sign = 1 if b == "G" else -1
            scores[i:i + run] = sign * val
            i += run
        else:
            i += 1
    return scores

def g4hunter_window(seq, window=25):
    s = g4hunter_per_base(seq)
    if len(s) < window:
        return np.array([s.mean()])
    kernel = np.ones(window) / window
    return np.convolve(s, kernel, mode="valid")

# ---------- (i) Telomeric repeats ----------
# SPECIES = [
#     ("Tetrahymena thermophila", "TTGGGG", "Ciliate"),
#     ("Oxytricha nova", "TTTTGGGG", "Ciliate"),
#     ("Paramecium tetraurelia", "TTGGGT", "Ciliate"),
#     ("Euplotes aediculatus", "TTTTGGGG", "Ciliate"),
#     ("Stylonychia mytilus", "TTTTGGGG", "Ciliate"),
#     ("Trypanosoma brucei", "TTAGGG", "Kinetoplastid"),
#     ("Leishmania major", "TTAGGG", "Kinetoplastid"),
#     ("Saccharomyces cerevisiae", "TGTGGGTGTGGTGTG", "Fungi (Asco.)"),
#     ("Schizosaccharomyces pombe", "TTACAGGG", "Fungi (Asco.)"),
#     ("Candida albicans", "GGTGTACGGATGTCACGATCATT", "Fungi (Asco.)"),
#     ("Candida glabrata", "GGGGTCTGGGTGCTG", "Fungi (Asco.)"),
#     ("Neurospora crassa", "TTAGGG", "Fungi (Asco.)"),
#     ("Aspergillus nidulans", "TTAGGG", "Fungi (Asco.)"),
#     ("Ustilago maydis", "TTAGGG", "Fungi (Basidio.)"),
#     ("Chlamydomonas reinhardtii", "TTTTAGGG", "Green alga"),
#     ("Caenorhabditis elegans", "TTAGGC", "Nematode"),
#     ("Bombyx mori", "TTAGG", "Insect"),
#     ("Apis mellifera", "TTAGG", "Insect"),
#     ("Tribolium castaneum", "TCAGG", "Insect"),
#     ("Locusta migratoria", "TTAGG", "Insect"),
#     ("Strongylocentrotus purpuratus", "TTAGGG", "Echinoderm"),
#     ("Crassostrea gigas", "TTAGGG", "Mollusc"),
#     ("Homo sapiens", "TTAGGG", "Vertebrate"),
#     ("Mus musculus", "TTAGGG", "Vertebrate"),
#     ("Gallus gallus", "TTAGGG", "Vertebrate"),
#     ("Danio rerio", "TTAGGG", "Vertebrate"),
#     ("Xenopus laevis", "TTAGGG", "Vertebrate"),
# ]

SPECIES = [
    # Ciliates
    ("Tetrahymena thermophila", "TTGGGG", "Ciliate"),
    ("Oxytricha nova", "TTTTGGGG", "Ciliate"),
    ("Paramecium tetraurelia", "TTGGGT", "Ciliate"),
    ("Euplotes aediculatus", "TTTTGGGG", "Ciliate"),
    ("Stylonychia mytilus", "TTTTGGGG", "Ciliate"),
    # Kinetoplastids / other protists
    ("Trypanosoma brucei", "TTAGGG", "Kinetoplastid"),
    ("Trypanosoma cruzi", "TTAGGG", "Kinetoplastid"),
    ("Leishmania major", "TTAGGG", "Kinetoplastid"),
    ("Giardia lamblia", "TAGGG", "Other protist"),
    ("Plasmodium falciparum", "TTTAGGG", "Other protist"),
    # Fungi - Ascomycota
    ("Saccharomyces cerevisiae", "TGTGGGTGTGGTGTG", "Fungi (Ascomycota)"),
    ("Saccharomyces castellii", "TCTGGGTG", "Fungi (Ascomycota)"),
    ("Schizosaccharomyces pombe", "TTACAGGG", "Fungi (Ascomycota)"),
    ("Candida albicans", "GGTGTACGGATGTCACGATCATT", "Fungi (Ascomycota)"),
    ("Candida glabrata", "GGGGTCTGGGTGCTG", "Fungi (Ascomycota)"),
    ("Candida tropicalis", "TTACGGATGTCTAACTCTTT", "Fungi (Ascomycota)"),
    ("Kluyveromyces lactis", "TTGATTAGGTATGTGGTGT", "Fungi (Ascomycota)"),
    ("Yarrowia lipolytica", "GGGTTAGTCA", "Fungi (Ascomycota)"),
    ("Pichia pastoris", "TTGGGTGCTGTGTGGGT", "Fungi (Ascomycota)"),
    ("Neurospora crassa", "TTAGGG", "Fungi (Ascomycota)"),
    ("Aspergillus nidulans", "TTAGGG", "Fungi (Ascomycota)"),
    ("Aspergillus fumigatus", "TTAGGG", "Fungi (Ascomycota)"),
    ("Magnaporthe oryzae", "TTAGGG", "Fungi (Ascomycota)"),
    ("Fusarium oxysporum", "TTAGGG", "Fungi (Ascomycota)"),
    ("Trichoderma reesei", "TTAGGG", "Fungi (Ascomycota)"),
    # Fungi - Basidiomycota
    ("Ustilago maydis", "TTAGGG", "Fungi (Basidiomycota)"),
    ("Cryptococcus neoformans", "TTAGGG", "Fungi (Basidiomycota)"),
    ("Coprinus cinereus", "TTAGGG", "Fungi (Basidiomycota)"),
    # Green alga
    ("Chlamydomonas reinhardtii", "TTTTAGGG", "Green alga"),
    # Nematodes
    ("Caenorhabditis elegans", "TTAGGC", "Nematode"),
    ("Caenorhabditis briggsae", "TTAGGC", "Nematode"),
    # Insects
    ("Bombyx mori", "TTAGG", "Insect"),
    ("Apis mellifera", "TTAGG", "Insect"),
    ("Tribolium castaneum", "TCAGG", "Insect"),
    ("Locusta migratoria", "TTAGG", "Insect"),
    ("Anopheles gambiae", "TTAGGC", "Insect"),
    # Other invertebrates
    ("Strongylocentrotus purpuratus", "TTAGGG", "Echinoderm"),
    ("Crassostrea gigas", "TTAGGG", "Mollusc"),
    ("Daphnia pulex", "TTAGGG", "Crustacean"),
    ("Hydra vulgaris", "TTAGGG", "Cnidarian"),
    ("Nematostella vectensis", "TTAGGG", "Cnidarian"),
    # Vertebrates
    ("Homo sapiens", "TTAGGG", "Vertebrate"),
    ("Mus musculus", "TTAGGG", "Vertebrate"),
    ("Bos taurus", "TTAGGG", "Vertebrate"),
    ("Ornithorhynchus anatinus", "TTAGGG", "Vertebrate"),
    ("Gallus gallus", "TTAGGG", "Vertebrate"),
    ("Anolis carolinensis", "TTAGGG", "Vertebrate"),
    ("Xenopus laevis", "TTAGGG", "Vertebrate"),
    ("Danio rerio", "TTAGGG", "Vertebrate"),
    ("Petromyzon marinus", "TTAGGG", "Vertebrate"),
]

TARGET_LEN = 600
def tile(motif, length=TARGET_LEN):
    return (motif * (length // len(motif) + 1))[:length]

rows = []
for sp, motif, clade in SPECIES:
    rows.append({"species": sp, "motif": motif, "clade": clade,
                 "g4h_mean": g4hunter_per_base(tile(motif)).mean()})
tel_df = pd.DataFrame(rows).sort_values("g4h_mean", ascending=True)
tel_df.to_csv("g4hunter_telomeric.csv", index=False)

# ---------- (ii)/(iii) hTERT proximal promoter ----------
with open("htert_promoter.json") as f:
    htert = json.load(f)

# We score the G-rich (plus) strand because G4 forms on the G-rich strand.
# Mutations C228T and C250T are C>T on minus, equivalent to G>A on plus.
PLUS_WT = htert["plus_strand"]
N = len(PLUS_WT)
POS_G_C228T_PLUS = N - 1 - htert["pos_C228T_minus"]
POS_G_C250T_PLUS = N - 1 - htert["pos_C250T_minus"]
assert PLUS_WT[POS_G_C228T_PLUS] == "G"
assert PLUS_WT[POS_G_C250T_PLUS] == "G"

def mutate(seq, pos, alt):
    return seq[:pos] + alt + seq[pos + 1:]

PLUS_C228T = mutate(PLUS_WT, POS_G_C228T_PLUS, "A")
PLUS_C250T = mutate(PLUS_WT, POS_G_C250T_PLUS, "A")
PLUS_BOTH = mutate(PLUS_C228T, POS_G_C250T_PLUS, "A")

WIN = 25
tracks = {
    "WT": g4hunter_window(PLUS_WT, WIN),
    "C228T": g4hunter_window(PLUS_C228T, WIN),
    "C250T": g4hunter_window(PLUS_C250T, WIN),
    "C228T + C250T": g4hunter_window(PLUS_BOTH, WIN),
}
x = np.arange(len(tracks["WT"])) + WIN // 2

# ---------- Plot ----------
fig = plt.figure(figsize=(15, 20))
gs = fig.add_gridspec(3, 1, height_ratios=[1.4, 1, 0.85], hspace=0.2)

ax1 = fig.add_subplot(gs[0])
clade_colors = {c: plt.get_cmap("tab10")(i)
                for i, c in enumerate(sorted(tel_df["clade"].unique()))}
y_pos = np.arange(len(tel_df))
ax1.barh(y_pos, tel_df["g4h_mean"],
         color=[clade_colors[c] for c in tel_df["clade"]],
         edgecolor="black", linewidth=0.6)
ax1.set_yticks(y_pos)
ax1.set_yticklabels([f"{r.species}  ({r.motif})" for r in tel_df.itertuples()],
                    fontsize=8, style="italic")
ax1.axvline(1.0, color="red", lw=0.8, ls="--", alpha=0.7)
ax1.axvline(-1.0, color="red", lw=0.8, ls="--", alpha=0.7)
ax1.set_xlabel("G4Hunter mean score (G-rich strand)")
ax1.set_title("(i) Predicted G4-forming propensity of tiled telomeric repeats",
              loc="left", fontsize=11)
handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in clade_colors.values()]
ax1.legend(handles, clade_colors.keys(), loc="lower right",
           fontsize=7, frameon=False, ncol=2)

ax2 = fig.add_subplot(gs[1])
colors = {"WT": "black", "C228T": "#d95f02",
          "C250T": "#1b9e77", "C228T + C250T": "#7570b3"}
for label, track in tracks.items():
    ax2.plot(x, track, label=label, color=colors[label],
             lw=1.8 if label == "WT" else 1.2,
             alpha=1.0 if label == "WT" else 0.85)
ax2.axhline(1.0, color="red", lw=0.6, ls="--", alpha=0.6)
ax2.axhline(-1.0, color="red", lw=0.6, ls="--", alpha=0.6)
ax2.axvline(POS_G_C228T_PLUS, color="#d95f02", lw=0.7, ls=":", alpha=0.8)
ax2.axvline(POS_G_C250T_PLUS, color="#1b9e77", lw=0.7, ls=":", alpha=0.8)
ax2.set_xlabel(f"Position in hTERT promoter window "
               f"(chr5:{htert['start']}-{htert['end']}, hg38, plus strand)")
ax2.set_ylabel("G4Hunter score (25 bp window)")
ax2.set_title("(ii) G4-forming potential across the hTERT proximal promoter "
              "with C228T / C250T overlays", loc="left", fontsize=11)
ax2.legend(loc="lower right", fontsize=8, frameon=False, ncol=4)

ax3 = fig.add_subplot(gs[2])
lo = min(POS_G_C228T_PLUS, POS_G_C250T_PLUS) - 35
hi = max(POS_G_C228T_PLUS, POS_G_C250T_PLUS) + 35
mask = (x >= lo) & (x <= hi)
for label, track in tracks.items():
    ax3.plot(x[mask], track[mask], label=label, color=colors[label],
             lw=1.8 if label == "WT" else 1.2, marker="o", markersize=3,
             alpha=1.0 if label == "WT" else 0.85)
ax3.axvline(POS_G_C228T_PLUS, color="#d95f02", lw=0.7, ls=":", alpha=0.8)
ax3.axvline(POS_G_C250T_PLUS, color="#1b9e77", lw=0.7, ls=":", alpha=0.8)
ax3.axhline(1.0, color="red", lw=0.6, ls="--", alpha=0.6)
ax3.set_xlabel("Position (bp, plus strand)")
ax3.set_ylabel("G4Hunter")
ax3.set_title("(iii) Zoom on the C228T / C250T hotspot region",
              loc="left", fontsize=11)

plt.savefig("FigB_g4hunter.pdf", bbox_inches="tight")
plt.savefig("FigB_g4hunter.png", dpi=300, bbox_inches="tight")
print(tel_df.to_string(index=False))