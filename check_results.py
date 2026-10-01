import json
import os

import numpy as np
import pandas as pd

from panel import load_panel, unique_motifs

EXPECTED_G4H = {"TTGGGG": 2.667, "TTTTGGGG": 2.000, "TAGGG": 1.800, "TTAGGG": 1.500,
                "TTGGGT": 1.500, "TTTAGGG": 1.277, "TTTTAGGG": 1.125, "TTAGG": 0.800,
                "TCAGG": 0.600, "TTAGGC": 0.500, "CTGGGTGCTGTGGGGT": 1.555,
                "ACGGATGTCTAACTTCTTGGTGT": 0.258, "ACGGATGTCACGATCATTGGTGT": 0.302,
                "ACGGATTTGATTAGGTATGTGGTGT": 0.560}


def check(cond, msg):
    print(("ok    " if cond else "FAIL  ") + msg)
    return cond


def main():
    ok = True
    df = load_panel()
    motifs = unique_motifs(df)
    sp = dict(zip(df["species"], df["motif"]))
    ok &= check(len(df) == 49, f"panel has 49 species ({len(df)})")
    ok &= check(len(motifs) == 19, f"panel has 19 distinct motifs ({len(motifs)})")
    ok &= check("Anopheles gambiae" not in sp, "Anopheles gambiae removed")
    ok &= check(sp.get("Daphnia pulex") == "TTAGG", "Daphnia pulex is TTAGG")
    ok &= check((df["motif"] == "TTAGGG").sum() == 25, "25 species share TTAGGG")

    g4 = pd.read_csv("results/g4hunter_telomeric_motifs.csv").set_index("motif")["g4h_mean"]
    for m, v in EXPECTED_G4H.items():
        ok &= check(abs(g4[m] - v) < 0.002, f"G4Hunter {m} = {g4[m]:.3f} (expected {v})")

    with open("htert_promoter.json") as f:
        htert = json.load(f)
    plus, n = htert["plus_strand"], len(htert["plus_strand"])
    i228 = n - 1 - htert["pos_C228T_minus"]
    i250 = n - 1 - htert["pos_C250T_minus"]
    ok &= check(plus[i228] == "G" and plus[i250] == "G", "G at both hotspots on the plus (template) strand")
    ok &= check(i250 - i228 == 22, "C228T and C250T are 22 nt apart")
    ok &= check(plus.startswith("GCGGGGGTGG"), "plus strand starts just upstream of the ATG")

    hs = pd.read_csv("results/htert_hotspot_scores.csv")
    for site in ("C228T", "C250T"):
        s = hs[hs["site"] == site].set_index("variant")["score_window_centred_on_site"]
        ok &= check(s[site] < s["WT"], f"{site} lowers the local score ({s['WT']:.2f} to {s[site]:.2f})")

    if os.path.isfile("results/motif_embeddings.npz"):
        emb = np.load("results/motif_embeddings.npz")
        ok &= check(list(emb["motifs"]) == motifs, "embedding motifs match species.tsv")
        ok &= check(emb["embeddings"].shape == (19, 1024), f"embedding shape {emb['embeddings'].shape}")
        ok &= check(np.isfinite(emb["embeddings"]).all(), "embeddings are finite")
    else:
        print("skip  results/motif_embeddings.npz not found")

    if os.path.isfile("results/drug_descriptors.csv"):
        d = pd.read_csv("results/drug_descriptors.csv")
        n_comp = len(pd.read_csv("compounds.tsv", sep="\t"))
        ok &= check(len(d) == n_comp, f"{len(d)} compounds with descriptors")
        ok &= check(d[["MW", "cLogP"]].notna().all().all(), "no missing descriptors")
    else:
        print("skip  results/drug_descriptors.csv not found")

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
