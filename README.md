# Polynomial-Regression-Machine-Learning-Assignment-
# Polynomial Regression: Degree vs MSE Plots

Companion script to the polynomial regression pipeline (Ridge/Lasso with `PolynomialFeatures`). `plot_degree_mse.py` sweeps polynomial degree for each regression problem, picks the best regularized model at every degree by cross-validation, and plots the results.

## Files

| File | Description |
|---|---|
| `plot_degree_mse.py` | Degree sweep and plotting script |
| `BT2024194_train_var{1,2}.csv` | Training data (input; must contain a `y` column) |
| `BT2024194_var{1,2}_degree_vs_mse.png` | Output: degree vs MSE plot |
| `BT2024194_var{1,2}_best_degree_fit.png` | Output: fit at the best degree |

## Requirements

- Python 3.8+
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Usage

Place the training CSVs in the same directory as the script (or change `D`), then run:

```bash
python plot_degree_mse.py
```

The script prints the best CV MSE for each degree and saves four PNGs (two per problem).

## Method

1. **Degree sweep:** var1 uses degrees 1-20 and var2 uses degrees 1-10 (set in `CFG`).
2. **Model selection per degree:** every `(model, alpha)` pair in the grid is scored with 5-fold CV repeated 3 times (`RepeatedKFold`, `random_state=0`). The pair with the lowest CV MSE is kept for that degree.
3. **Pipeline:** `PolynomialFeatures` -> `StandardScaler` -> `Ridge` or `Lasso`.
4. **Best degree:** the degree with the lowest CV MSE is refit on the full training set for the second plot.

## Outputs

**Degree vs MSE** (`*_degree_vs_mse.png`)
- Train MSE and CV MSE per degree, on a log y-axis.
- The best degree is marked with a dashed line and a red point.

**Best-degree fit** (`*_best_degree_fit.png`)
- One input feature: training data with the fitted curve.
- Multiple input features: predicted vs actual `y` with the `y = ŷ` reference line.

## Configuration

Edit `CFG` at the top of the script:

```python
CFG = {1: dict(degrees=range(1, 21), models=[...]),
       2: dict(degrees=range(1, 11), models=[...])}
```

- `degrees`: degrees to sweep.
- `models`: list of `('ridge' | 'lasso', alpha)` candidates tried at each degree.
- `ROLL`, `D`: file prefix and data directory.

## Notes

- **Runtime:** with many input features, high degrees create a very large number of polynomial terms, and every degree fits the full model grid. If it is slow, shrink the model list or cap the maximum degree.
- **Regularization effect:** because alpha is tuned at every degree, high degrees can look better than they would unregularized. For a plainer overfitting curve, use a single fixed `Ridge(alpha=1e-6)` instead of the grid.
- **Metric:** MSE is computed on CV folds (test) and on the training portion of each fold (train).
