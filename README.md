# telomere-analysis

Computational analyses contributed to a review on telomere biology. Two panel figures with supporting code and data.

## Figures

**Figure A: Telomeric repeat embeddings across eukaryotes.** Telomeric tandem repeats from 50 species are tiled to 1.2 kb, embedded with [Nucleotide Transformer v2](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species) (500M, multi-species), and projected to 2D with UMAP. Recovers cross-clade conservation of the canonical TTAGGG hexamer and reveals four motif-defined regions (canonical TTAGGG; TTAGGC; heterogeneous yeast/ciliate; pentameric insect).

**Figure B: Predicted G-quadruplex propensity.** G4Hunter (Bedrat et al., NAR 2016) applied to (i) the same 50 telomeric motifs and (ii) the human hTERT proximal promoter (UCSC, GRCh38, chr5:1,294,990–1,295,390), comparing wild-type to the C228T and C250T cancer hotspot mutations.

## Files

| File | Description |
|---|---|
| `nt_emb_extr.py` | Extracts NT v2 embeddings for the species panel and runs UMAP. |
| `g4hunter.py` | Pure-numpy G4Hunter implementation. Generates Figure B. |
| `download_ucsc.py` | Fetches the hTERT promoter window from the UCSC REST API. |
| `telomere_embeddings.npy` | NT v2 mean-pooled hidden states (50 × 1024). |
| `telomere_umap.csv` | UMAP coordinates with species/motif/clade metadata. |
| `g4hunter_telomeric.csv` | G4Hunter mean scores per species. |
| `htert_promoter.json` | hTERT promoter sequence and hotspot positions. |
| `FigA_*`, `FigB_*` | Final figures (PDF + PNG). |
| `env.yml` | Conda environment specification. |

## Reproducing

```bash
conda env create -f env.yml
conda activate nt-env

python nt_emb_extr.py        # ~1 min on a 24 GB GPU
python download_ucsc.py
python g4hunter.py
```

## Environment

Tested on Ubuntu 24.04 with an NVIDIA GPU (24 GB). Requires `transformers==4.44.2` (the NT v2 custom code is incompatible with transformers 5.x). Inference uses fp32 weights with `torch.autocast` for the forward pass.

## Notes

- The NT v2 projection is a sequence-similarity analysis, not a phylogenetic reconstruction.
- G4Hunter is a sequence-based predictor; scores are not experimental measurements.
- Telomeric motifs were sourced from primary literature and should be cross-checked against [TeloBase](https://cfb.ceitec.muni.cz/telobase/) before reuse.

## References

- Dalla-Torre H. et al. The Nucleotide Transformer: building and evaluating robust foundation models for human genomics. *Nat. Methods* (2025).
- Bedrat A., Lacroix L., Mergny J.-L. Re-evaluation of G-quadruplex propensity with G4Hunter. *Nucleic Acids Res.* 44, 1746–1759 (2016).
- McInnes L., Healy J., Melville J. UMAP: Uniform Manifold Approximation and Projection. arXiv:1802.03426 (2018).
- Horn S. et al. TERT promoter mutations in familial and sporadic melanoma. *Science* 339, 959–961 (2013).
- Huang F. W. et al. Highly recurrent TERT promoter mutations in human melanoma. *Science* 339, 957–959 (2013).
- Bell R. J. A. et al. The transcription factor GABP selectively binds and activates the mutant TERT promoter in cancer. *Science* 348, 1036–1039 (2015).
