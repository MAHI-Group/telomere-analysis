import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForMaskedLM
from umap import UMAP
from matplotlib.lines import Line2D
from adjustText import adjust_text

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

TARGET_LEN = 1200
MODEL = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"

def tile(motif, length=TARGET_LEN):
    return (motif * (length // len(motif) + 1))[:length]

df = pd.DataFrame(SPECIES, columns=["species", "motif", "group"])
df["sequence"] = df["motif"].apply(tile)

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(
    MODEL, trust_remote_code=True
).to(device).eval()

@torch.inference_mode()
def embed(seqs, batch_size=4):
    out = []
    for i in range(0, len(seqs), batch_size):
        tok = tokenizer.batch_encode_plus(
            seqs[i:i+batch_size], return_tensors="pt",
            padding="longest", truncation=True, max_length=2048
        )
        input_ids = tok["input_ids"].to(device)
        attn = (input_ids != tokenizer.pad_token_id).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outs = model(input_ids, attention_mask=attn,
                         encoder_attention_mask=attn, output_hidden_states=True)
        h = outs.hidden_states[-1]
        mask = attn.unsqueeze(-1).float()
        pooled = (h.float() * mask).sum(1) / mask.sum(1)
        out.append(pooled.cpu().numpy())
    return np.concatenate(out)

embs = embed(df["sequence"].tolist())
np.save("telomere_embeddings.npy", embs)

reducer = UMAP(n_neighbors=15, min_dist=0.6, metric="cosine",
               spread=1.5, random_state=42)
xy = reducer.fit_transform(embs)
df["umap_1"], df["umap_2"] = xy[:, 0], xy[:, 1]
df.to_csv("telomere_umap.csv", index=False)

#UMAP
df = pd.read_csv("telomere_umap.csv")
#df = df[df["group"] != "Plant"].reset_index(drop=True)

agg = (df.groupby(["motif", "umap_1", "umap_2"], as_index=False)
         .agg(species=("species", list),
              groups=("group", lambda g: sorted(set(g))),
              n=("species", "count")))
agg["color_group"] = agg.apply(
    lambda r: r["groups"][0] if len(r["groups"]) == 1 else "Cross-clade",
    axis=1,
)
agg = agg.sort_values("n", ascending=False).reset_index(drop=True)

groups = sorted(agg["color_group"].unique())
cmap = plt.get_cmap("tab20")
color_map = {g: cmap(i) for i, g in enumerate(groups)}

fig = plt.figure(figsize=(17, 9))
gs = fig.add_gridspec(1, 2, width_ratios=[3, 2.2], wspace=0.05)
ax = fig.add_subplot(gs[0])
ax_legend = fig.add_subplot(gs[1])
ax_legend.axis("off")

texts = []
for _, r in agg.iterrows():
    ax.scatter(r["umap_1"], r["umap_2"], s=140,
               color=color_map[r["color_group"]],
               edgecolor="black", linewidth=0.6, alpha=0.9, zorder=3)
    label = r["motif"] if r["n"] == 1 else f"{r['motif']} (n={r['n']})"
    texts.append(ax.text(r["umap_1"], r["umap_2"], label,
                         fontsize=9, family="monospace"))

adjust_text(
    texts, ax=ax,
    arrowprops=dict(arrowstyle="-", color="grey", lw=0.5,
                    shrinkA=8, shrinkB=4, connectionstyle="arc3"),
    expand_points=(1.6, 1.6), expand_text=(1.2, 1.2),
)

ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_title("Telomeric tandem-repeat embeddings across eukaryotes\n"
             "(Nucleotide Transformer v2, 500M, multi-species)")

handles = [Line2D([0], [0], marker="o", color="w",
                  markerfacecolor=color_map[g], markeredgecolor="black",
                  markersize=9, label=g) for g in groups]
ax.legend(handles=handles, loc="lower left", fontsize=8,
          frameon=False, title="Clade", title_fontsize=9)

def wrap(species_list, width=42):
    lines, current = [], ""
    for s in species_list:
        candidate = s if not current else current + ", " + s
        if len(candidate) > width and current:
            lines.append(current)
            current = s
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines

ax_legend.text(0.0, 1.0, "Motif → species", fontsize=12, weight="bold",
               transform=ax_legend.transAxes)

y = 0.96
line_height = 0.030
for _, r in agg.iterrows():
    ax_legend.text(0.0, y, r["motif"], fontsize=9, family="monospace",
                   weight="bold", color=color_map[r["color_group"]],
                   transform=ax_legend.transAxes, va="top")
    species_lines = wrap(r["species"], width=42)
    for j, line in enumerate(species_lines):
        ax_legend.text(0.40, y - j * line_height, line, fontsize=8,
                       style="italic", transform=ax_legend.transAxes,
                       va="top")
    y -= line_height * (len(species_lines) + 0.4)

plt.savefig("FigA_telomere_umap.pdf", bbox_inches="tight")
plt.savefig("FigA_telomere_umap.png", dpi=300, bbox_inches="tight")