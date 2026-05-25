import json

with open("htert_promoter.json") as f:
    htert = json.load(f)

plus = htert["plus_strand"]
minus = htert["minus_strand"]
START = htert["start"]
END = htert["end"]
N = len(plus)

print(f"Window: chr5:{START}-{END} (hg38), length = {N} bp")
print(f"Plus strand length:  {len(plus)}")
print(f"Minus strand length: {len(minus)}")
print()

print("First 30 bp of plus :", plus[:30])
print("Last  30 bp of plus :", plus[-30:])
print("First 30 bp of minus:", minus[:30])
print()

assert plus == minus[::-1].translate(str.maketrans("ACGT", "TGCA")), \
    "revcomp mismatch — minus strand is not revcomp of plus"
print("revcomp check: OK")
print()

C228T_HG38 = 1295113
C250T_HG38 = 1295135

# Three index conventions to test against the literature site.
def candidates(g):
    return {
        "END - g          (current buggy)": END - g,
        "END - 1 - g      (off-by-one fix)": END - 1 - g,
        "g - START        (plus-strand index)": g - START,
    }

print("=" * 70)
for name, g in [("C228T", C228T_HG38), ("C250T", C250T_HG38)]:
    print(f"\n{name} at hg38 chr5:{g}")
    for label, idx in candidates(g).items():
        if 0 <= idx < N:
            on_minus = minus[idx]
            on_plus = plus[idx]
            print(f"  {label:40s}  idx={idx:3d}  "
                  f"minus[{idx}]={on_minus}  plus[{idx}]={on_plus}")
        else:
            print(f"  {label:40s}  idx={idx:3d}  out of range")

print()
print("=" * 70)
print("Sequence context (plus strand) around each candidate plus-strand index")
print("=" * 70)

def show_context(label, plus_idx, width=15):
    if not (0 <= plus_idx < N):
        print(f"{label}: idx {plus_idx} out of range")
        return
    lo = max(0, plus_idx - width)
    hi = min(N, plus_idx + width + 1)
    seq = plus[lo:hi]
    pointer = " " * (plus_idx - lo) + "^"
    print(f"\n{label}: plus-strand index {plus_idx}, base = {plus[plus_idx]}")
    print(f"  plus  : {seq}")
    print(f"          {pointer}")
    mlo, mhi = N - 1 - (hi - 1), N - 1 - lo
    mseq = minus[mlo:mhi + 1]
    mpointer_pos = (hi - 1) - plus_idx
    mpointer = " " * mpointer_pos + "^"
    print(f"  minus : {mseq}  (reversed view, 5'->3' of minus)")
    print(f"          {mpointer}")

for name, g in [("C228T", C228T_HG38), ("C250T", C250T_HG38)]:
    print(f"\n--- {name} ---")
    for label, j in candidates(g).items():
        plus_idx = N - 1 - j if "g - START" not in label else j
        show_context(f"  {label}", plus_idx)

print()
print("=" * 70)
print("Anchor search: locate canonical contexts in the minus strand")
print("=" * 70)

ANCHORS = {
    "C228T": ("CCCCCTCCGGGCC", 4),
    "C250T": ("CCCCCTCCCAGCC", 4),
}
for name, (anchor, mut_offset) in ANCHORS.items():
    hits = [i for i in range(len(minus) - len(anchor) + 1)
            if minus[i:i + len(anchor)] == anchor]
    print(f"\n{name}: anchor '{anchor}' on minus strand")
    print(f"  hits: {hits}")
    if len(hits) == 1:
        i = hits[0]
        mut_minus = i + mut_offset
        mut_plus = N - 1 - mut_minus
        genomic = END - 1 - mut_minus
        print(f"  mutated C at minus[{mut_minus}] = {minus[mut_minus]} (expect C)")
        print(f"  corresponds to plus[{mut_plus}] = {plus[mut_plus]} (expect G)")
        print(f"  genomic position (hg38) = chr5:{genomic}")
