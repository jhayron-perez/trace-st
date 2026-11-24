"""Night-9 N3 (fork F2): the supplements that stay, regenerated into figures_v9 with the night-6 T5 code (N6_T5), unchanged
inputs (no trajectory recomputation). Tables -> figures_v9/tables; intermediate tables -> night9/analysis/T5_supp.
python N9_supp.py --which TS1 T8 R1 R2 S13 S14 S15 S16 S17 S18 S19 [--nproc 16]   (PBS develop only)
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import N6_T5 as T  # noqa: E402
import N5_style as S  # noqa: E402

R1 = T.R1
T.FIG6 = R1 / "figures_v9"
T.TAB = R1 / "figures_v9" / "tables"
T.SUPP = R1 / "night9" / "analysis" / "T5_supp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["TS1", "T8", "R1", "R2", "S13", "S14", "S15", "S16", "S17", "S18", "S19"])
    ap.add_argument("--nproc", type=int, default=16)
    a = ap.parse_args()
    for d in (T.FIG6, T.TAB, T.SUPP):
        d.mkdir(parents=True, exist_ok=True)
    S.set_output(T.FIG6, T.RENAME)
    fns = {"S13": T.run_S13, "S14": T.run_S14, "S15": T.run_S15, "S16": T.run_S16, "S17": T.run_S17,
           "S18": lambda: T.run_S18(a.nproc), "S19": T.run_S19, "R1": T.run_R1, "R2": T.run_R2, "TS1": T.run_TS1, "T8": T.run_T8}
    for w in a.which:
        t0 = time.time()
        print(f"===== {w}", flush=True)
        try:
            fns[w]()
        except Exception as e:  # one failing item must not block the others
            import traceback; traceback.print_exc()
            print("FAILED", w, e, flush=True)
        print(f"===== {w} done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
