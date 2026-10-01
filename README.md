# telomere-analysis

Illustrative computational analyses for a review on telomere biology (Maji, Dash, Chattopadhyay; Biochimie, in revision). All outputs are predictions, not experimental measurements.

## Analyses

- **G4Hunter on telomeric motifs.** Telomeric repeat motifs from 49 eukaryotic species (19 distinct motifs; `species.tsv`) are tiled to 600 nt and scored with G4Hunter (Bedrat et al., NAR 2016).
- **G4Hunter on the hTERT proximal promoter.** A 400-nt window (chr5:1,294,991-1,295,390, GRCh38; -2 to -401 relative to the TERT ATG) is scored on the G-rich template (plus) strand, for the wild type and the C228T (-124C>T) and C250T (-146C>T) hotspot mutations, modelled as G>A on the plus strand.
- **DNA language-model embeddings.** Each distinct motif is tiled to 1,200 nt and embedded with [Nucleotide Transformer v2](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species) (500M, multi-species; attention-masked mean pooling of the final layer). Species that share a motif share an embedding, so only the 19 distinct motifs are embedded. Similarity is shown as a cosine-similarity heatmap with average-linkage clustering.
- **Physicochemical descriptors.** Structures of 17 small molecules are retrieved from PubChem by name (CIDs recorded in `results/compounds_resolved.tsv`), reduced to the parent structure (largest fragment), and described with RDKit.

## Files

| File | Description |
| --- | --- |
| `species.tsv` | Species panel: motif (5' to 3', G-rich strand), clade, primary source, verification status |
| `compounds.tsv` | Compound list, PubChem query names, optional SMILES overrides with their source |
| `panel.py` | Shared helpers (panel loading, tiling, labels, colours) |
| `g4hunter.py` | G4Hunter on telomeric motifs and the hTERT promoter |
| `nt_emb_extr.py` | Nucleotide Transformer embeddings and similarity figure (optional UMAP) |
| `drug.py` | PubChem retrieval and RDKit descriptors |
| `download_ucsc.py` | Fetches the hTERT promoter window from the UCSC REST API into `htert_promoter.json` |
| `check_results.py` | Consistency checks on the panel and all outputs |

## Reproducing

```bash
conda env create -f env.yml
conda activate nt-env
pip install rdkit

python g4hunter.py
python nt_emb_extr.py
python drug.py
python check_results.py
```

`nt_emb_extr.py` needs a GPU for reasonable speed (about a minute on 24 GB) and caches embeddings in `results/motif_embeddings.npz`; pass `--recompute` to ignore the cache and `--umap` for an additional UMAP of the distinct motifs. `drug.py` caches PubChem look-ups in `results/compounds_resolved.tsv`; pass `--refresh` to re-query. `download_ucsc.py` only needs rerunning to regenerate `htert_promoter.json`.

## Outputs

| Output | Use |
| --- | --- |
| `figures/Fig_insilico_g4` | Main in silico figure: (A) motif scores, (B) hTERT promoter, (C) hotspot zoom |
| `figures/Fig_g4hunter_motifs`, `figures/Fig_htert_promoter` | The same panels as separate files |
| `figures/FigS1_nt_motif_similarity` | Supplementary Figure S1 |
| `figures/FigS2_drug_descriptors` | Supplementary Figure S2 |
| `results/g4hunter_telomeric_motifs.csv`, `results/g4hunter_telomeric_species.csv` | G4Hunter scores (Tables S2 and S4) |
| `results/htert_g4hunter_tracks.csv`, `results/htert_hotspot_scores.csv` | hTERT window scores |
| `results/motif_embeddings.npz`, `results/motif_cosine_similarity.csv` | Embeddings (19 x 1,024) and similarity matrix |
| `results/compounds_resolved.tsv`, `results/drug_descriptors.csv` | Structures with PubChem CIDs, and descriptors |

## Environment

Ubuntu, single NVIDIA GPU (24 GB). Requires `transformers==4.44.2` (the NT v2 custom code is incompatible with transformers 5.x); weights are loaded in fp32 with `torch.autocast` for the forward pass.

## Limitations

- The embeddings reflect the sequence of the repeat unit only; they carry no information on phylogeny, repeat heterogeneity, telomere length or chromatin.
- G4Hunter is a sequence-based propensity score; the |score| > 1.0 cut-off is heuristic, and the mean score of a strictly periodic tract is fixed by the composition of its repeat unit.
- The hTERT analysis predicts local changes in G-richness only. The established mechanism by which C228T and C250T activate TERT is the creation of ETS/GABP binding sites (Bell et al., Science 2015).
- Descriptors summarise general drug-likeness and do not predict G4 binding or activity.

## References

- Dalla-Torre H. et al. Nucleotide Transformer: building and evaluating robust foundation models for human genomics. *Nat. Methods* 22, 287-297 (2025).
- Bedrat A., Lacroix L., Mergny J.-L. Re-evaluation of G-quadruplex propensity with G4Hunter. *Nucleic Acids Res.* 44, 1746-1759 (2016).
- Lyčka M. et al. TeloBase: a community-curated database of telomere sequences across the tree of life. *Nucleic Acids Res.* 52, D311-D321 (2024).
- Horn S. et al. TERT promoter mutations in familial and sporadic melanoma. *Science* 339, 959-961 (2013).
- Huang F. W. et al. Highly recurrent TERT promoter mutations in human melanoma. *Science* 339, 957-959 (2013).
- Bell R. J. A. et al. The transcription factor GABP selectively binds and activates the mutant TERT promoter in cancer. *Science* 348, 1036-1039 (2015).
- McInnes L., Healy J., Melville J. UMAP: Uniform Manifold Approximation and Projection for dimension reduction. arXiv:1802.03426 (2018).
- Landrum G. RDKit: open-source cheminformatics. https://www.rdkit.org
- Kim S. et al. PubChem 2023 update. *Nucleic Acids Res.* 51, D1373-D1380 (2023).
