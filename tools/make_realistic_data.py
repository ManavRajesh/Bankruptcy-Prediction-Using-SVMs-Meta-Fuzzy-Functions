import pandas as pd, numpy as np
from pathlib import Path

def make_data(n_banks=400, start=1999, end=2007, seed=42):
    rng = np.random.default_rng(seed)
    years = list(range(start, end+1))
    rows = []

    for b in range(n_banks):
        bank = f"Bank_{b+1}"
        base = rng.uniform(0.4, 0.9)
        profile = rng.choice(["healthy","gradual","shock","recover"], p=[0.5,0.2,0.2,0.1])
        fail_year = rng.choice([2003,2004,2005,2006,2007], p=[0.05,0.1,0.2,0.25,0.4])

        for y in years:
            ratios = base + rng.normal(0, 0.05, size=17)

            # gradual distress
            if profile=="gradual" and y>=2002:
                ratios -= (y-2002)*0.04

            # shock pattern
            if profile=="shock" and y in [2004,2005,2006]:
                ratios -= 0.15

            # recovery banks dip then improve
            if profile=="recover":
                if y<=2003: ratios -= 0.08
                if y>=2005: ratios += 0.1

            # failure event
            fail = 1 if y == fail_year and np.mean(ratios) < 0.45 else 0

            rows.append({ "bank": bank, "year": y,
                          **{f"ratio_{i+1}": float(r) for i,r in enumerate(ratios)},
                          "failed": fail })

    df = pd.DataFrame(rows)
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/bank_failure_realistic.csv", index=False)
    print("✅ Created:", df.shape)

if __name__ == "__main__":
    make_data()
