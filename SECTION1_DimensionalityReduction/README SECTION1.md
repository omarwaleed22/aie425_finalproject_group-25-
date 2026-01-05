# Collaborative Filtering Experiments (SVD + PCA)

This repo contains Jupyter notebooks that explore **collaborative filtering** on an explicit user–item rating dataset (e.g., MovieLens-style `ratings.csv`).  
You will find implementations and experiments for:

- **SVD for Collaborative Filtering**
- **PCA with Mean Filling**
- **PCA with Maximum Likelihood Estimation (MLE)**

## Notebooks

- `Singular_Value_Decomposition_(SVD)_for_Collaborative_Filtering.ipynb`
- `PCA_Method_with_Mean_Filling.ipynb`
- `PCA_Method_with_Maximum_Likelihood_Estimation.ipynb`

> Tip: run the notebooks in the order above if you want a clean progression from baseline SVD → PCA variants.

## Expected Input Data

The notebooks assume a ratings table with (at least) these columns:

- `userId` (int)
- `movieId` (int)
- `rating` (float)

Common filename: `ratings.csv`

If your file uses different column names, adjust the column names in the notebooks (or use the helper functions in `utils.py`).

## Outputs

The notebooks write results into an `out/` directory. Common outputs include (may vary by notebook):

- User/item statistics: `n_u.csv`, `n_i.csv`, `rbar_u.csv`, `rbar_i.csv`
- Popularity / grouping outputs: `popularity_items.csv`, `item_groups.csv`, `group_counts.csv`, `group_total_ratings*.csv`
- Target evaluation outputs:
  - `target_users.csv`, `target_items.csv`
  - `thresholds_per_target_user.csv`
  - `truncSVD_target_predictions.csv`
  - `truncSVD_target_predictions_with_groundtruth.csv`
  - `truncSVD_target_accuracy.csv`
- Reconstruction error logs:
  - `reconstruction_errors_observed_only.csv`
  - `reconstruction_errors_all_entries.csv`
  - `truncatedSVD_reconstruction_errors.csv`

Some notebooks also write intermediate artifacts (pickles) such as:
- `out/preprocessing_results_part1.pkl`
- `out/intermediate_results.pkl`

## Setup

### 1) Create an environment (recommended)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
```

### 2) Install requirements

```bash
pip install -r requirements.txt
```

### 3) Run

Open Jupyter and execute the notebooks:

```bash
jupyter notebook
```

## Project Structure (suggested)

```
.
├─ *.ipynb
├─ utils.py
├─ requirements.txt
├─ ratings.csv               # (or place data under ./data and update paths)
└─ out/
```

## Notes

- Large datasets can take time/memory. If you run into issues, try sampling fewer rows (the notebooks often have a `MAX_RATINGS` or similar setting).
- All outputs are written to `out/`. If you change output locations, update the helper functions accordingly.

