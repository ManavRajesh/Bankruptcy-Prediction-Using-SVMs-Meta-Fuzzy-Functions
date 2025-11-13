import argparse, numpy as np, pandas as pd
from pathlib import Path

def generate(n_banks=100, years=(1998,1999,2000,2001,2002,2003), n_ratios=17, fail_rate=0.2, seed=42):
    rng = np.random.default_rng(seed)
    rows = []

    banks = [f"Bank_{i+1}" for i in range(n_banks)]

    for bank in banks:
        baseline = rng.uniform(0.2,0.8)
        fail_flag = rng.random() < fail_rate

        for y in years:
            ratios = {
                f"ratio_{r+1}": np.clip(baseline + rng.normal(0,0.1), 0, 1)
                for r in range(n_ratios)
            }

            # gradual deterioration before failure
            if fail_flag and y >= years[-2]:
                for k in ratios:
                    ratios[k] = np.clip(ratios[k] - rng.uniform(0.05, 0.2), 0, 1)

            rows.append({
                "bank": bank,
                "year": y,
                **{k: round(v,3) for k,v in ratios.items()},
                "failed": 1 if (fail_flag and y == years[-1]) else 0
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_banks", type=int, default=200)
    ap.add_argument("--start_year", type=int, default=1998)
    ap.add_argument("--n_years", type=int, default=10)
    ap.add_argument("--n_ratios", type=int, default=17)
    ap.add_argument("--fail_rate", type=float, default=0.25)
    ap.add_argument("--out", type=str, default="data/synth_bank_failure.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    years = tuple(range(args.start_year, args.start_year + args.n_years))
    df = generate(
        n_banks=args.n_banks,
        years=years,
        n_ratios=args.n_ratios,
        fail_rate=args.fail_rate,
        seed=args.seed
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print("Wrote", args.out, "shape=", df.shape)
