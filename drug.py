import argparse
import os
import time
import urllib.parse

import matplotlib.pyplot as plt
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
COMPOUNDS = "compounds.tsv"
CACHE = "results/compounds_resolved.tsv"
COLOURS = {"Synthetic Telomere (G4)": "#e74c3c", "Synthetic Non-telomere": "#3498db",
           "Natural Telomere (G4)": "#f1c40f", "Natural Non-telomere": "#2ecc71"}


def pubchem_lookup(query):
    q = urllib.parse.quote(query, safe="")
    r = requests.get(f"{PUG}/compound/name/{q}/cids/JSON", timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    cids = r.json()["IdentifierList"]["CID"]
    cid = cids[0]
    time.sleep(0.25)
    for prop in ("SMILES", "IsomericSMILES"):
        r = requests.get(f"{PUG}/compound/cid/{cid}/property/{prop},MolecularFormula,Title/JSON",
                         timeout=30)
        if r.ok:
            break
    r.raise_for_status()
    props = r.json()["PropertyTable"]["Properties"][0]
    smiles = next(v for k, v in props.items() if k.endswith("SMILES"))
    time.sleep(0.25)
    return {"cid": cid, "n_cids_for_name": len(cids), "pubchem_title": props.get("Title", ""),
            "pubchem_formula": props["MolecularFormula"], "smiles": smiles}


def resolve(compounds, refresh):
    cache = {}
    if os.path.isfile(CACHE) and not refresh:
        cache = pd.read_csv(CACHE, sep="\t", dtype=str).set_index("name").to_dict("index")

    rows, missing = [], []
    for c in compounds.itertuples():
        override = c.smiles_override if isinstance(c.smiles_override, str) and c.smiles_override else None
        if override:
            if not (isinstance(c.override_source, str) and c.override_source):
                raise ValueError(f"{c.name}: smiles_override needs override_source")
            rows.append({"name": c.name, "cid": "", "n_cids_for_name": "", "pubchem_title": "",
                         "pubchem_formula": "", "smiles": override, "smiles_source": c.override_source})
            continue
        hit = cache.get(c.name)
        if hit is None or hit.get("smiles_source") != "PubChem":
            hit = pubchem_lookup(c.pubchem_query)
            if hit is None:
                missing.append(c.name)
                continue
            hit["smiles_source"] = "PubChem"
        rows.append({"name": c.name, **{k: hit[k] for k in
                     ("cid", "n_cids_for_name", "pubchem_title", "pubchem_formula", "smiles", "smiles_source")}})

    if missing:
        raise SystemExit(
            "Not found in PubChem by name: " + ", ".join(missing) +
            f"\nAdd smiles_override and override_source for these rows in {COMPOUNDS} "
            "(structure from the primary paper or its SI), then rerun.")
    resolved = pd.DataFrame(rows)
    resolved.to_csv(CACHE, sep="\t", index=False)
    return resolved


def descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    parent = rdMolStandardize.LargestFragmentChooser().choose(mol)
    return {
        "parent_smiles": Chem.MolToSmiles(parent),
        "parent_formula": rdMolDescriptors.CalcMolFormula(parent),
        "formal_charge": Chem.GetFormalCharge(parent),
        "MW": Descriptors.MolWt(parent),
        "cLogP": Crippen.MolLogP(parent),
        "TPSA": rdMolDescriptors.CalcTPSA(parent),
        "HBD": Lipinski.NumHDonors(parent),
        "HBA": Lipinski.NumHAcceptors(parent),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(parent),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(parent),
    }


def plot(df, stem):
    fig, ax = plt.subplots(figsize=(13, 8))
    for cat, colour in COLOURS.items():
        sub = df[df["category"] == cat]
        if len(sub):
            ax.scatter(sub["MW"], sub["cLogP"], s=sub["aromatic_rings"].clip(lower=1) * 100,
                       c=colour, label=cat, edgecolors="black", alpha=0.75)
    texts = [ax.text(r.MW, r.cLogP, r.name, fontsize=9) for r in df.itertuples()]
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))
    except ImportError:
        pass
    ax.axvline(500, color="grey", ls="--", alpha=0.4)
    ax.axhline(5, color="grey", ls="--", alpha=0.4)
    ax.set_xlabel("Molecular weight of parent structure (Da)", fontsize=12)
    ax.set_ylabel("Calculated logP (Crippen)", fontsize=12)
    ax.legend(title="Bubble size = aromatic rings", bbox_to_anchor=(1.02, 1), loc="upper left",
              labelspacing=1.5, frameon=True, fontsize=10)
    ax.grid(True, ls=":", alpha=0.5)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-query PubChem, ignoring the cache")
    args = parser.parse_args()
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    compounds = pd.read_csv(COMPOUNDS, sep="\t", dtype=str).fillna("")
    resolved = resolve(compounds, args.refresh)
    desc = pd.DataFrame([descriptors(s) for s in resolved["smiles"]])
    df = pd.concat([compounds[["name", "origin", "target"]].reset_index(drop=True),
                    resolved.drop(columns="name"), desc], axis=1)
    df["category"] = df["origin"] + " " + df["target"]
    if len(df) != len(compounds):
        raise RuntimeError("compound count mismatch")
    df.to_csv("results/drug_descriptors.csv", index=False, float_format="%.3f")
    plot(df, "figures/FigS2_drug_descriptors")

    print(f"{len(df)} compounds")
    print(df[["name", "cid", "pubchem_title", "pubchem_formula", "parent_formula",
              "formal_charge", "MW", "cLogP", "aromatic_rings"]].to_string(index=False))


if __name__ == "__main__":
    main()
