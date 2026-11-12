"""Night-3 N2: draw 5 extra PNW twin years (seed 2022, pool 1995-2020 minus the A1 years) and write the replace-design task files."""
import numpy as np
from pathlib import Path
A1 = [2006, 2011, 2012, 2018, 2020]
pool = np.array([y for y in range(1995, 2021) if y not in A1])
new = sorted(int(y) for y in np.random.default_rng(2022).choice(pool, 5, replace=False))
print("new years:", new)
T = Path("/glade/derecho/scratch/jhayron/DataCaStLeBTs/revision1/night3/tasks")
(T.parent / "ids" / "pnw_years10_new.txt").write_text(" ".join(map(str, new)) + "\n")
lines = []
for v in ("olr", "slhf", "sshf"):
    for y in (new if v != "sshf" else new + A1):
        lines.append(f"pnw_{v}_rep{y}|8_HW_PNW21_Runs_v2.py|--n-trials 300 --n-members 10 --nproc 30 --start-method fork "
                     f"--continue-if-exists --run-ids @PNW233@ --placebo-vars {v} --placebo-year {y} --placebo-mode replace")
for i in range(0, len(lines), 4):
    (T / f"n2y10_part_{i // 4:02d}").write_text("\n".join(lines[i:i + 4]) + "\n")
print(len(lines), "runs")
