import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

# 1. Complete Molecule Dataset (Small Molecules & Natural)
compounds = [
    # --- NATURAL ---
    {"Name": "Berberine", "Type": "Natural", "Target": "Telomere (G4)", "SMILES": "COc1ccc2c(c1)C[n+]3ccc4c(c3C2)cc5c(c4)OCO5"},
    {"Name": "EGCG", "Type": "Natural", "Target": "Non-Telomere", "SMILES": "C1[C@H]([C@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O"},
    {"Name": "Curcumin", "Type": "Natural", "Target": "Non-Telomere", "SMILES": "COC1=C(C=CC(=C1)/C=C/C(=O)CC(=O)/C=C/C2=CC(=C(C=C2)O)OC)O"},
    {"Name": "Resveratrol", "Type": "Natural", "Target": "Non-Telomere", "SMILES": "Oc1cc(O)cc(\\C=C\\c2ccc(O)cc2)c1"},
    {"Name": "Camptothecin", "Type": "Natural", "Target": "Non-Telomere", "SMILES": "CC[C@@]1(C2=C(COC1=O)C(=O)N3CC4=CC5=CC=CC=C5N=C4C3=C2)O"},

    # --- SYNTHETIC G4 LIGANDS ---
    {"Name": "BRACO-19", "Type": "Synthetic", "Target": "Telomere (G4)", "SMILES": "CN(C)CCCN1c2cc(cc(c2c3ccc(cc13)NC(=O)c4ccc(cc4)N(C)C)NC(=O)c5ccc(cc5)N(C)C)NC(=O)c6ccc(cc6)N(C)C"},
    {"Name": "RHPS4", "Type": "Synthetic", "Target": "Telomere (G4)", "SMILES": "C[N+]4=C2C3=C(C=C(C)C=C3C5=C4C=CC(F)=C5)N(C)C1=CC=C(F)C=C12"},
    {"Name": "Pyridostatin", "Type": "Synthetic", "Target": "Telomere (G4)", "SMILES": "NCCOC1=CC(C(Nc2nc3ccccc3c(OCCN)n2)=O)=NC(C(Nc4nc5ccccc5c(OCCN)n4)=O)=C1"},
    {"Name": "Telomestatin", "Type": "Synthetic", "Target": "Telomere (G4)", "SMILES": "Cc1oc(-c2nc(-c3nc(-c4nc(-c5nc(-c6nc(-c7nc(C)oc7C)co6)co5)co4)co3)co2)co1"},
    {"Name": "GTC365", "Type": "Synthetic", "Target": "Telomere (G4)", "SMILES": "CC1=CC2=C(C=C1)N(C3=C2C=C(C=C3)C(=O)N)CCCNC(=N)N"},
    {"Name": "RG260", "Type": "Synthetic", "Target": "Telomere (G4)", "SMILES": "O=C(NC(NC1=CC=C(OC2=NC=C(Br)C=N2)C(C)=C1)=O)C3=CC=CC=C3NC([C@@H](N)C)=O"},
    {"Name": "TMPyP4", "Type": "Synthetic", "Target": "Telomere (G4)", "SMILES": "C[n+]1ccc(cc1)c2c3ccc(n3)c(c4ccc(n4)c(c5ccc(n5)c(c6ccc2n6)c7cc[n+](cc7)C)c8cc[n+](cc8)C)c9cc[n+](cc9)C"},

    # --- SYNTHETIC ENZYME/PATHWAY INHIBITORS ---
    {"Name": "BIBR1532", "Type": "Synthetic", "Target": "Non-Telomere", "SMILES": "CC(=CC(=O)NC1=CC=CC=C1C(=O)O)C2=CC3=CC=CC=C3C=C2"},
    {"Name": "MST-312", "Type": "Synthetic", "Target": "Non-Telomere", "SMILES": "O=C(NC1=CC=CC(NC(C2=CC=CC(O)=C2O)=O)=C1)C3=CC=CC(O)=C3O"},
    {"Name": "AG 1478", "Type": "Synthetic", "Target": "Non-Telomere", "SMILES": "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC=CC=C3)OC.Cl"},
    {"Name": "VE-821", "Type": "Synthetic", "Target": "Non-Telomere", "SMILES": "CS(=O)(=O)C1=CC=C(C=C1)C2=NC(=C(N=C2N)C3=CC=CC=C3)C4=CC=C(C=C4)S(=O)(=O)C"},
    {"Name": "Ceralasertib", "Type": "Synthetic", "Target": "Non-Telomere", "SMILES": "C[C@@H]1CN(CCO1)C2=NC(=C(C=N2)C3=CC=C(C=C3)S(=O)(=O)C4CC4)N"}
]

# 2. Calculation
data = []
for cp in compounds:
    mol = Chem.MolFromSmiles(cp["SMILES"])
    if mol:
        data.append({
            "Name": cp["Name"],
            "Category": f"{cp['Type']} {cp['Target']}",
            "MW": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "Aromatic": Lipinski.NumAromaticRings(mol)
        })

df = pd.DataFrame(data)

# 3. Figure Generation
plt.figure(figsize=(13, 8))
colors = {"Synthetic Telomere (G4)": "#e74c3c", "Synthetic Non-Telomere": "#3498db", 
          "Natural Telomere (G4)": "#f1c40f", "Natural Non-Telomere": "#2ecc71"}

for cat, color in colors.items():
    subset = df[df["Category"] == cat]
    plt.scatter(subset["MW"], subset["LogP"], s=subset["Aromatic"]*100, 
                c=color, label=cat, edgecolors='black', alpha=0.7)

for i, row in df.iterrows():
    plt.annotate(row["Name"], (row["MW"], row["LogP"]), xytext=(5,5), textcoords='offset points', fontsize=9)

plt.axvline(x=500, color='grey', linestyle='--', alpha=0.3)
plt.axhline(y=5, color='grey', linestyle='--', alpha=0.3)
plt.xlabel("Molecular Weight (Da)", fontsize=12)
plt.ylabel("Lipophilicity (LogP)", fontsize=12)
plt.title("Physicochemical Profile of All Telomere-Related Small Molecules\n(Bubble size = Aromatic Ring Count)", fontsize=14)
plt.legend(
    title="Molecule Classification", 
    bbox_to_anchor=(1.05, 1), 
    loc='upper left',
    labelspacing=1.8,    # Increases vertical space between items
    borderpad=1.5,       # Increases space between the legend border and content
    handletextpad=1.2,   # Increases space between the icon and the text
    frameon=True,
    fontsize=11
)
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig("FigC_drug.pdf", bbox_inches="tight")
plt.savefig("FigC_drug.png", dpi=300, bbox_inches="tight")
plt.close("all")