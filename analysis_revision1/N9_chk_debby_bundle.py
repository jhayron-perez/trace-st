import numpy as np, json
d = "/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1/night9/n1_smoke/bundles/debby_trial_00001"
b = np.load(d + ".npz", allow_pickle=True)
print(list(b.keys()))
P = b["parents"]
print("parents shape", P.shape, "max", int(np.nanmax(P)), "counts", np.bincount(P[P >= 0].astype(int)) if P.size else None)
print(json.load(open(d + ".json")).get("var_names"))
