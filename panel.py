import pandas as pd

PANEL_FILE = "species.tsv"


def load_panel(path=PANEL_FILE):
    df = pd.read_csv(path, sep="\t", dtype=str)
    df["motif"] = df["motif"].str.upper().str.strip()
    if df["species"].duplicated().any():
        raise ValueError("duplicate species in panel")
    if not df["motif"].str.fullmatch("[ACGT]+").all():
        raise ValueError("motifs must contain only A, C, G, T")
    return df


def unique_motifs(df):
    return list(dict.fromkeys(df["motif"]))


def tile(motif, length):
    return (motif * (length // len(motif) + 1))[:length]


def motif_labels(df):
    labels = {}
    for motif, grp in df.groupby("motif", sort=False):
        n = len(grp)
        labels[motif] = grp["species"].iloc[0] if n == 1 else f"{n} species"
    return labels


def motif_clade(df):
    clades = df.groupby("motif", sort=False)["clade"].agg(lambda c: sorted(set(c)))
    return {m: c[0] if len(c) == 1 else "Multiple clades" for m, c in clades.items()}


def clade_colours(groups):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    single = [g for g in sorted(groups) if g != "Multiple clades"]
    colours = {g: cmap(i % 10) for i, g in enumerate(single)}
    if "Multiple clades" in groups:
        colours["Multiple clades"] = "lightgrey"
    return colours
