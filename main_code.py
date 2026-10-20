import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

ROLL, D = 'BT2024194', '.'
CFG = {1: dict(degrees=range(3, 9), models=[('ridge', a) for a in (1, 3, 10, 30, 100)] + [('lasso', a) for a in (0.003, 0.006, 0.01, 0.02)]),
       2: dict(degrees=range(7, 13), models=[('ridge', a) for a in (0.03, 0.1, 0.3, 1, 3)])}

def build(deg, kind, a):
    reg = Ridge(alpha=a) if kind == 'ridge' else Lasso(alpha=a, max_iter=200000)
    return make_pipeline(PolynomialFeatures(deg, include_bias=(kind == 'ridge')), StandardScaler(), reg)

rk = RepeatedKFold(n_splits=5, n_repeats=3, random_state=0)
for v, cfg in CFG.items():
    tr = pd.read_csv(f'{D}/{ROLL}_train_var{v}.csv'); te = pd.read_csv(f'{D}/{ROLL}_test_var{v}.csv')
    X, y = tr.drop(columns='y').values, tr.y.values
    best = None
    for deg in cfg['degrees']:
        for kind, a in cfg['models']:
            mse = -cross_val_score(build(deg, kind, a), X, y, cv=rk, scoring='neg_mean_squared_error', n_jobs=-1).mean()
            if best is None or mse < best[0]: best = (mse, deg, kind, a)
    mse, deg, kind, a = best
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, random_state=42)
    h = build(deg, kind, a).fit(Xa, ya).predict(Xb)
    print(f'var{v}: deg={deg} {kind} alpha={a} | CV MSE={mse:.4f} | holdout MSE={mean_squared_error(yb, h):.4f} R2={r2_score(yb, h):.4f}')
    model = build(deg, kind, a).fit(X, y)
    pd.DataFrame({'y': model.predict(te.values)}).to_csv(f'{D}/{ROLL}_pred_var{v}.csv', index=False)
