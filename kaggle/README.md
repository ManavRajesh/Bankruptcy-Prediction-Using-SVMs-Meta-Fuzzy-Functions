# Bank Failure Prediction: SVM + Meta Fuzzy Functions

This dataset and code implement an ensemble of SVMs combined with Meta Fuzzy Functions (MFF) to predict bank failures.

## Files
- `data/bank_failure_dataset.csv` — synthetic dataset (41 banks × 4 years, 17 ratios)
- `train.py` — trains SVM grids, builds MFF, outputs metrics
- `src/` — reusable modules
- `notebooks/MFF_SVM_BankFailure.ipynb` — runnable notebook

## How to Run on Kaggle
1. Upload this folder as a Dataset or use it in a Kaggle Notebook.
2. `pip install -r requirements.txt`
3. Run `python train.py --data data/bank_failure_dataset.csv --train_years 1998 --val_year 2000 --test_year 2001`

## License
Synthetic data provided for educational use.
