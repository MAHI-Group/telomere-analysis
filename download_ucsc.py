import requests
import json

# UCSC REST API. hTERT is on the minus strand, so we fetch a window
# upstream of the TSS (higher genomic coordinates) and reverse-complement.
# Window: 400 bp around the proximal promoter to comfortably include
# both hotspots and surrounding G-tracts.
CHROM = "chr5"
START = 1294990   # 0-based
END   = 1295390
GENOME = "hg38"

url = (f"https://api.genome.ucsc.edu/getData/sequence?"
       f"genome={GENOME};chrom={CHROM};start={START};end={END}")
r = requests.get(url, timeout=30)
r.raise_for_status()
data = r.json()
plus_strand = data["dna"].upper()

COMP = str.maketrans("ACGT", "TGCA")
def revcomp(s):
    return s.translate(COMP)[::-1]

minus_strand = revcomp(plus_strand)

# Hotspot positions on the minus strand (relative to our window).
# The mutations are C>T on the minus strand (G>A on plus).
# Genomic positions of the C residues that mutate:
#   C228T at chr5:1,295,113
#   C250T at chr5:1,295,135
# Position in minus_strand = END - genomic_pos
POS_C228T_MINUS = END - 1295113   # index into minus_strand
POS_C250T_MINUS = END - 1295135

print(f"Window: {CHROM}:{START}-{END} ({GENOME}), {len(plus_strand)} bp")
print(f"Plus strand 5'->3':\n{plus_strand}\n")
print(f"Minus strand (transcribed strand) 5'->3':\n{minus_strand}\n")
print(f"C228T position in minus strand: {POS_C228T_MINUS} "
      f"(base = {minus_strand[POS_C228T_MINUS]}, expect C)")
print(f"C250T position in minus strand: {POS_C250T_MINUS} "
      f"(base = {minus_strand[POS_C250T_MINUS]}, expect C)")

with open("htert_promoter.json", "w") as f:
    json.dump({
        "genome": GENOME, "chrom": CHROM, "start": START, "end": END,
        "plus_strand": plus_strand, "minus_strand": minus_strand,
        "pos_C228T_minus": POS_C228T_MINUS,
        "pos_C250T_minus": POS_C250T_MINUS,
    }, f, indent=2)
print("\nSaved to htert_promoter.json")
