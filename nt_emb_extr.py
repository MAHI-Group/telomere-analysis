import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

from panel import clade_colours, load_panel, motif_clade, motif_labels, tile, unique_motifs

MODEL = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
TARGET_LEN = 1200
SEED = 42
EMB_FILE = "results/motif_embeddings.npz"


def embed_motifs(motifs, batch_size=4):
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(MODEL, trust_remote_code=True).to(device).eval()

    seqs = [tile(m, TARGET_LEN) for m in motifs]
    out = []
    with torch.inference_mode():
        for i in range(0, len(seqs), batch_size):
            tok = tokenizer.batch_encode_plus(
                seqs[i:i + batch_size], return_tensors="pt",
                padding="longest", truncation=True, max_length=2048,
            )
            input_ids = tok["input_ids"].to(device)
            attn = (input_ids != tokenizer.pad_token_id).to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                outs = model(input_ids, attention_mask=attn,
                             encoder_attention_mask=attn, output_hidden_states=True)
            h = outs.hidden_states[-1]
            mask = attn.unsqueeze(-1).float()
            out.append(((h.float() * mask).sum(1) / mask.sum(1)).cpu().numpy())
    return np.concatenate(out)


def load_or_embed(motifs, recompute):
    if os.path.isfile(EMB_FILE) and not recompute:
        cached = np.load(EMB_FILE, allow_pickle=False)
        if list(cached["motifs"]) == motifs:
            print(f"Loaded cached embeddings from {EMB_FILE}")
            return cached["embeddings"]
        print("Cached motifs differ from species.tsv; recomputing")
    emb = embed_motifs(motifs)
    np.savez(EMB_FILE, motifs=np.array(motifs), embeddings=emb,
             model=MODEL, tile_length=TARGET_LEN)
    print(f"Saved {emb.shape} embeddings to {EMB_FILE}")
    return emb


def tick_label(motif, label):
    if label.endswith("species"):
        return f"{motif}  ({label})"
    return f"{motif}  ($\\it{{{label.replace(' ', '~')}}}$)"


def plot_similarity(motifs, emb, labels, clades, stem):
    dist = pdist(emb, metric="cosine")
    sim = 1.0 - squareform(dist)
    pd.DataFrame(sim, index=motifs, columns=motifs).to_csv("results/motif_cosine_similarity.csv")

    Z = linkage(dist, method="average", optimal_ordering=True)
    n = len(motifs)
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 4], height_ratios=[0.035, 1],
                          wspace=0.02, hspace=0.04)
    ax_c = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_h = fig.add_subplot(gs[1, 1])

    dn = dendrogram(Z, orientation="left", ax=ax_d, no_labels=True,
                    color_threshold=0, above_threshold_color="black")
    order = dn["leaves"]
    ax_d.set_ylim(10 * n, 0)
    ax_d.axis("off")

    off = sim[~np.eye(n, dtype=bool)]
    im = ax_h.imshow(sim[np.ix_(order, order)], cmap="viridis", aspect="auto",
                     vmin=off.min(), vmax=off.max())
    ax_h.set_yticks(range(n))
    ax_h.set_yticklabels([tick_label(motifs[i], labels[motifs[i]]) for i in order],
                         fontsize=11, family="monospace")
    ax_h.yaxis.tick_right()
    ax_h.set_xticks(range(n))
    ax_h.set_xticklabels([motifs[i] for i in order], rotation=90, fontsize=10, family="monospace")
    cb = fig.colorbar(im, cax=ax_c, orientation="horizontal")
    cb.set_label("Cosine similarity of mean-pooled embeddings (diagonal saturated)", fontsize=11)
    ax_c.xaxis.set_ticks_position("top")
    ax_c.xaxis.set_label_position("top")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    return sim, order


def plot_umap(motifs, emb, labels, clades, stem, n_neighbors, min_dist):
    from umap import UMAP

    xy = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine",
              random_state=SEED).fit_transform(emb)
    pd.DataFrame({"motif": motifs, "umap_1": xy[:, 0], "umap_2": xy[:, 1]}).to_csv(
        "results/motif_umap.csv", index=False)

    colour = clade_colours(set(clades.values()))
    groups = list(colour)
    fig, ax = plt.subplots(figsize=(11, 9))
    for m, (x, y) in zip(motifs, xy):
        ax.scatter(x, y, s=160, color=colour[clades[m]], edgecolor="black", lw=0.8, zorder=3)
        ax.annotate(m, (x, y), xytext=(6, 4), textcoords="offset points",
                    fontsize=10, family="monospace")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colour[g],
                          markeredgecolor="black", markersize=10, label=g) for g in groups]
    ax.legend(handles=handles, loc="best", fontsize=11, frameon=False, title="Clade")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true", help="ignore cached embeddings")
    parser.add_argument("--umap", action="store_true", help="also plot a UMAP of the distinct motifs")
    parser.add_argument("--n-neighbors", type=int, default=5)
    parser.add_argument("--min-dist", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    df = load_panel()
    motifs = unique_motifs(df)
    print(f"{len(df)} species, {len(motifs)} distinct motifs")
    emb = load_or_embed(motifs, args.recompute)

    labels = motif_labels(df)
    clades = motif_clade(df)
    plot_similarity(motifs, emb, labels, clades, "figures/FigS1_nt_motif_similarity")
    if args.umap:
        n_neighbors = min(args.n_neighbors, len(motifs) - 1)
        plot_umap(motifs, emb, labels, clades, "figures/FigS1b_nt_motif_umap",
                  n_neighbors, args.min_dist)
        print(f"UMAP: n_neighbors={n_neighbors}, min_dist={args.min_dist}, cosine, seed={SEED}")


if __name__ == "__main__":
    main()
