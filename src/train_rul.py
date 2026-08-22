"""
Train an XGBoost RUL regressor on N-CMAPSS cycle-level features.

Evaluation: leave-one-unit-out cross-validation on the dev split,
then final evaluation on the held-out test split.

Metrics:
  RMSE  — root mean squared error (cycles)
  Score — NASA asymmetric s-score (penalises late predictions more)

Outputs (saved to results/):
  rul_predictions_DS01.csv / rul_predictions_DS02.csv
  rul_cv_results_DS01.csv  / rul_cv_results_DS02.csv
  rul_pred_vs_actual_DS01.png / rul_pred_vs_actual_DS02.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

PROC_DIR    = Path(__file__).resolve().parents[1] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DATASETS = ["DS01", "DS02"]

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

NON_FEATURE_COLS = {"unit", "cycle", "RUL", "split", "Fc", "hs"}


def nasa_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """NASA asymmetric s-score: penalises late predictions more than early ones."""
    d = predicted - actual
    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(score.sum())


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def louo_cv(dev_df: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-unit-out cross-validation on the dev split."""
    feat_cols = get_feature_cols(dev_df)
    units = sorted(dev_df["unit"].unique())
    records = []

    for val_unit in units:
        train = dev_df[dev_df["unit"] != val_unit]
        val   = dev_df[dev_df["unit"] == val_unit]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[feat_cols])
        X_val   = scaler.transform(val[feat_cols])
        y_train = train["RUL"].values
        y_val   = val["RUL"].values

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)

        preds = model.predict(X_val).clip(0, 125)
        rmse  = float(np.sqrt(mean_squared_error(y_val, preds)))
        score = nasa_score(y_val, preds)

        print(f"  Unit {int(val_unit):2d}  RMSE={rmse:.2f}  Score={score:.0f}")
        records.append({
            "unit": val_unit,
            "n_cycles": len(val),
            "rmse": rmse,
            "nasa_score": score,
        })

    return pd.DataFrame(records)


def train_final(dev_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, object, StandardScaler]:
    """Train on full dev split, evaluate on test split."""
    feat_cols = get_feature_cols(dev_df)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(dev_df[feat_cols])
    X_test  = scaler.transform(test_df[feat_cols])
    y_train = dev_df["RUL"].values
    y_test  = test_df["RUL"].values

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train, verbose=False)

    preds = model.predict(X_test).clip(0, 125)
    result_df = test_df[["unit", "cycle", "RUL"]].copy()
    result_df["predicted_RUL"] = preds
    result_df["error"] = preds - y_test

    rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
    score = nasa_score(y_test, preds)
    print(f"  Test RMSE={rmse:.2f}  NASA Score={score:.0f}")

    return result_df, model, scaler


def plot_predictions(result_df: pd.DataFrame, ds_name: str, out: Path) -> None:
    """Scatter of predicted vs actual RUL, coloured by engine unit."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    units = sorted(result_df["unit"].unique())
    cmap  = plt.cm.get_cmap("tab10", len(units))

    ax = axes[0]
    for i, uid in enumerate(units):
        sub = result_df[result_df["unit"] == uid]
        ax.scatter(sub["RUL"], sub["predicted_RUL"],
                   s=10, alpha=0.5, color=cmap(i), label=f"Unit {int(uid)}")
    lims = [0, 125]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual RUL (cycles)"); ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title(f"{ds_name} — Predicted vs Actual RUL")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    for i, uid in enumerate(units):
        sub = result_df[result_df["unit"] == uid].sort_values("cycle")
        ax.plot(sub["cycle"], sub["RUL"], "--", color=cmap(i), linewidth=1, alpha=0.7)
        ax.plot(sub["cycle"], sub["predicted_RUL"], "-", color=cmap(i), linewidth=1.5,
                label=f"Unit {int(uid)}")
    ax.set_xlabel("Flight Cycle"); ax.set_ylabel("RUL (cycles)")
    ax.set_title(f"{ds_name} — RUL over time (— pred, -- actual)")
    ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out.name}")


def run_dataset(ds_name: str) -> None:
    feat_path = PROC_DIR / f"features_{ds_name}.parquet"
    if not feat_path.exists():
        print(f"Missing {feat_path.name} — run src/feature_engineering.py first.")
        return

    df      = pd.read_parquet(feat_path)
    dev_df  = df[df["split"] == "dev"].copy()
    test_df = df[df["split"] == "test"].copy()

    print(f"\n=== {ds_name} ===")
    print(f"Dev:  {len(dev_df):,} cycle rows | {dev_df['unit'].nunique()} units")
    print(f"Test: {len(test_df):,} cycle rows | {test_df['unit'].nunique()} units")

    print("\nLeave-one-unit-out CV (dev split):")
    cv_results = louo_cv(dev_df)
    print(f"\n  Mean CV RMSE:  {cv_results['rmse'].mean():.2f} cycles")
    print(f"  Mean CV Score: {cv_results['nasa_score'].mean():.0f}")
    cv_results.to_csv(RESULTS_DIR / f"rul_cv_results_{ds_name}.csv", index=False)

    print("\nFinal model (train=dev, eval=test):")
    result_df, _, _ = train_final(dev_df, test_df)
    result_df.to_csv(RESULTS_DIR / f"rul_predictions_{ds_name}.csv", index=False)

    plot_predictions(result_df, ds_name,
                     RESULTS_DIR / f"rul_pred_vs_actual_{ds_name}.png")


def main() -> None:
    for ds_name in DATASETS:
        run_dataset(ds_name)


if __name__ == "__main__":
    main()
