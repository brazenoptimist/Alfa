from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from train_credit_offer_model import (
    CAT_COLS,
    DATE_COL,
    ID_COL,
    TARGET,
    add_context_features,
    add_features,
    blend_prediction_dict,
    prepare_lgbm_frames,
    write_submission,
)


RAW_DATE_FEATURES = [
    "decision_year",
    "decision_dayofyear",
    "decision_weekofyear",
    "decision_days_since_2024",
]


def build_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    drop_raw_date: bool,
    drop_month_context: bool,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    train_fe = add_features(train)
    test_fe = add_features(test)
    train_fe, test_fe = add_context_features(train_fe, test_fe)
    if drop_raw_date:
        train_fe = train_fe.drop(columns=RAW_DATE_FEATURES, errors="ignore")
        test_fe = test_fe.drop(columns=RAW_DATE_FEATURES, errors="ignore")
    if drop_month_context:
        month_context_cols = [
            c
            for c in train_fe.columns
            if c.endswith("_rank_in_month")
            or c.endswith("_z_in_month")
            or c.endswith("_month_count_all_log")
        ]
        train_fe = train_fe.drop(columns=month_context_cols, errors="ignore")
        test_fe = test_fe.drop(columns=month_context_cols, errors="ignore")
    features = [c for c in train_fe.columns if c not in {ID_COL, TARGET}]
    return train_fe[features], train_fe[TARGET].astype(int), test_fe[features], features


def fit_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    *,
    name: str,
    depth: int,
    learning_rate: float,
    l2_leaf_reg: float,
    iterations: int,
    seed: int,
) -> tuple[str, pd.Series]:
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        auto_class_weights="SqrtBalanced",
        random_seed=seed,
        allow_writing_files=False,
        thread_count=-1,
        verbose=False,
    )
    model.fit(X, y, cat_features=[c for c in CAT_COLS if c in X.columns], verbose=False)
    return name, model.predict_proba(X_test)[:, 1]


def fit_lgbm_deep(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame) -> tuple[str, pd.Series]:
    params = {
        "objective": "binary",
        "n_estimators": 360,
        "learning_rate": 0.012,
        "num_leaves": 47,
        "max_depth": -1,
        "min_child_samples": 70,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.76,
        "reg_alpha": 1.2,
        "reg_lambda": 6.0,
        "random_state": 49,
        "n_jobs": -1,
        "verbosity": -1,
    }
    X_lgb, _, X_test_lgb = prepare_lgbm_frames(X, X.iloc[:1].copy(), X_test)
    model = LGBMClassifier(**params)
    model.fit(X_lgb, y, categorical_feature=[c for c in CAT_COLS if c in X.columns])
    return "lgbm_deep", model.predict_proba(X_test_lgb)[:, 1]


def fit_xgb(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame) -> tuple[str, pd.Series]:
    encoded = pd.concat([X, X_test], ignore_index=True)
    encoded = pd.get_dummies(encoded, columns=[c for c in CAT_COLS if c in encoded.columns], dummy_na=False)
    X_enc = encoded.iloc[: len(X)]
    X_test_enc = encoded.iloc[len(X) :]
    model = XGBClassifier(
        n_estimators=1200,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        max_depth=5,
        learning_rate=0.02,
        min_child_weight=40,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.5,
        reg_lambda=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_enc, y, verbose=False)
    return "xgb", model.predict_proba(X_test_enc)[:, 1]


def make_variant(args: argparse.Namespace) -> None:
    train = pd.read_csv(args.data_dir / "train_apps.csv")
    test = pd.read_csv(args.data_dir / "test_apps.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")

    if args.train_start_date:
        train = train[pd.to_datetime(train[DATE_COL]) >= pd.Timestamp(args.train_start_date)].copy()

    X, y, X_test, _ = build_features(train, test, args.drop_raw_date, args.drop_month_context)

    preds: dict[str, pd.Series] = {}
    for spec in [
        ("cat_sqrt_d7_seed77", 7, 0.02, 14, 960, 77),
        ("cat_sqrt_d7_seed99", 7, 0.02, 14, 745, 99),
        ("cat_sqrt_d8_seed77", 8, 0.016, 20, 1215, 77),
        ("cat_sqrt_d8_seed123", 8, 0.016, 20, 1280, 123),
        ("cat_sqrt_d6_seed51", 6, 0.02, 12, 1000, 51),
    ]:
        name, depth, lr, l2, iterations, seed = spec
        print(f"training {name}")
        pred_name, pred = fit_catboost(
            X,
            y,
            X_test,
            name=name,
            depth=depth,
            learning_rate=lr,
            l2_leaf_reg=l2,
            iterations=iterations,
            seed=seed,
        )
        preds[pred_name] = pred

    print("training xgb")
    name, pred = fit_xgb(X, y, X_test)
    preds[name] = pred

    print("training lgbm_deep")
    name, pred = fit_lgbm_deep(X, y, X_test)
    preds[name] = pred

    weights = {
        "cat_sqrt_d7_seed77": 0.25,
        "cat_sqrt_d7_seed99": 0.22,
        "cat_sqrt_d8_seed77": 0.18,
        "cat_sqrt_d8_seed123": 0.18,
        "cat_sqrt_d6_seed51": 0.10,
        "xgb": 0.05,
        "lgbm_deep": 0.02,
    }
    blended = blend_prediction_dict(preds, weights, "rank")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    write_submission(sample, test, train, y, blended, output_path)
    pd.DataFrame({ID_COL: test[ID_COL], **preds}).to_csv(
        args.output_dir / args.prediction_name,
        index=False,
    )
    print(f"saved {output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--train-start-date", default=None)
    parser.add_argument("--drop-raw-date", action="store_true")
    parser.add_argument("--drop-month-context", action="store_true")
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--prediction-name", required=True)
    args = parser.parse_args()
    make_variant(args)


if __name__ == "__main__":
    main()
