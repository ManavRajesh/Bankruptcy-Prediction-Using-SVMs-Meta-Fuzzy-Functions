# Bank Failure: SVM Ensemble + Meta Fuzzy Functions

A complete, batteries-included implementation matching the paper's idea:

- Many SVMs (RBF + Linear) with diverse hyperparameters
- Functional margin matrix Z
- Fuzzy C-Means to learn which SVMs deserve higher weight
- Meta Fuzzy Functions produce final blended scores
- Baselines: L1 / L2 / L∞ ensemble rules

## Quickstart
```bash
pip install -r requirements.txt
python train.py --data data/bank_failure_dataset.csv --train_years 1998 --val_year 2000 --test_year 2001 --outdir outputs
```

## Streamlit App
```bash
streamlit run app/streamlit_app.py
```

## GPU Variant
```bash
pip install thundersvm  # or RAPIDS cuML
python gpu/train_gpu.py --data data/bank_failure_dataset.csv
```

## Structure
```
bank_failure_project/
  data/
  src/
  app/
  gpu/
  tools/
  notebooks/
  outputs/
  kaggle/
```

## Notes
- This uses **decision_function** margins, as in the paper.
- Fuzzy weights sweep c=2..5 and m=1.3..3.0.
- You can generate bigger data:
  ```bash
  python tools/synth_data.py --n_banks 500 --n_years 15 --out data/big.csv
  ```
