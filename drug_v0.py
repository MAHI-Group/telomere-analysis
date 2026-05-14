import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

# 1. Dataset Construction based on the Article
compounds = [
    # --- NATURAL ---
    {"Name": "Berberine", "Origin": "Natural", "Target": "Telomere (G4)", "SMILES": "COc1ccc2c(c1)C[n+]3ccc4c(c3C2)cc5c(c4)OCO5"},
    {"Name": "EGCG", "Origin": "Natural", "Target": "Non-Telomere", "SMILES": "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"},
    {"Name": "Curcumin", "Origin": "Natural", "Target": "Non-Telomere", "SMILES": "COC1=C(C=CC(=C1)/C=C/C(=O)CC(=O)/C=C/C2=CC(=C(C=C2)O)OC)O"},
    {"Name": "Resveratrol", "Origin": "Natural", "Target": "Non-Telomere", "SMILES": "Oc1cc(O)cc(\\C=C\\c2ccc(O)cc2)c1"},
    
    # --- SYNTHETIC ---
    {"Name": "BRACO-19", "Origin": "Synthetic", "Target": "Telomere (G4)", "SMILES": "CN(C)CCCN1c2cc(cc(c2c3ccc(cc13)NC(=O)c4ccc(cc4)N(C)C)NC(=O)c5ccc(cc5)N(C)C)NC(=O)c6ccc(cc6)N(C)C"},
    {"Name": "RHPS4", "Origin": "Synthetic", "Target": "Telomere (G4)", "SMILES": "C[N+]4=C2C3=C(C=C(C)C=C3C5=C4C=CC(F)=C5)N(C)C1=CC=C(F)C=C12"},
    {"Name": "Pyridostatin", "Origin": "Synthetic", "Target": "Telomere (G4)", "SMILES": "NCCOC1=CC(C(Nc2nc3ccccc3c(OCCN)n2)=O)=NC(C(Nc4nc5ccccc5c(OCCN)n4)=O)=C1"},
    {"Name": "BIBR1532", "Origin": "Synthetic", "Target": "Non-Telomere", "SMILES": "CC(=CC(=O)NC1=CC=CC=C1C(=O)O)C2=CC3=CC=CC=C3C=C2"},
    {"Name": "RG260", "Origin": "Synthetic", "Target": "Non-Telomere", "SMILES": "O=C(NC(NC1=CC=C(OC2=NC=C(Br)C=N2)C(C)=C1)=O)C3=CC=CC=C3NC([C@@H](N)C)=O"}
]

# 2. Descriptor Calculation
data = []
for cp in compounds:
    mol = Chem.MolFromSmiles(cp["SMILES"])
    if mol:
        data.append({
            "Name": cp["Name"],
            "Origin": cp["Origin"],
            "Target": cp["Target"],
            "Category": f"{cp['Origin']} {cp['Target']}",
            "MW": Descriptors.MolWt(mol),
            "Aromatic_Rings": Lipinski.NumAromaticRings(mol)
        })

df = pd.DataFrame(data)

# 3. Comprehensive Visualization
plt.figure(figsize=(12, 7))

# Mapping Styles
styles = {
    "Natural Non-Telomere": {"color": "#27ae60", "marker": "o"}, # Green Circle
    "Natural Telomere (G4)": {"color": "#27ae60", "marker": "D"}, # Green Diamond
    "Synthetic Non-Telomere": {"color": "#2980b9", "marker": "o"}, # Blue Circle
    "Synthetic Telomere (G4)": {"color": "#c0392b", "marker": "D"} # Red Diamond
}

for cat, style in styles.items():
    subset = df[df["Category"] == cat]
    plt.scatter(subset["MW"], subset["Aromatic_Rings"], 
                s=200, c=style["color"], marker=style["marker"], 
                label=cat, edgecolors='black', alpha=0.8)

# Annotations for each molecule
for i, row in df.iterrows():
    plt.annotate(row["Name"], (row["MW"], row["Aromatic_Rings"]), 
                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

plt.axvline(x=500, color='gray', linestyle='--', alpha=0.5, label="Lipinski MW Limit")
plt.xlabel("Molecular Weight (Da)", fontsize=12)
plt.ylabel("Number of Aromatic Rings", fontsize=12)
plt.title("Chemical Landscape of Telomere-Related Drug Molecules", fontsize=14, pad=20)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(title="Molecule Classification", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig("FigC_drug.pdf", bbox_inches="tight")
plt.savefig("FigC_drug.png", dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.close("all")