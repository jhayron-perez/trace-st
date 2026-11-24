"""Night-8 S10: the v23 code (--trial-ids, parallel members) reproduces night5/t3/p05 trial 15 (both alphas): all non-time columns equal."""
import pandas as pd
R1 = "/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1/"
a = pd.read_csv(R1 + "night5/t3/p05/results_alphas_final.csv"); a = a[a.trial_id == 15].sort_values("alpha_mix").reset_index(drop=True)
b = pd.read_csv(R1 + "night8/s10_dry/byte_p05/results_alphas_final.csv").sort_values("alpha_mix").reset_index(drop=True)
cols = [c for c in a.columns if c in b.columns and "elapsed" not in c and "sec" not in c and "bundle" not in c and c != "updated_at"]
bad = []
for c in cols:
    x, y = a[c], b[c]
    ok = x.equals(y)
    if not ok:
        bad.append((c, x.tolist()[:2], y.tolist()[:2]))
print("rows", len(a), len(b), "columns compared", len(cols), "differing", len(bad))
for r in bad[:15]: print(r)
print("BYTE CHECK", "PASS" if not bad and len(a) == len(b) == 2 else "FAIL")
