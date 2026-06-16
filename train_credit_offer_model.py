from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "target_value"
ID_COL = "front_id"
DATE_COL = "decision_day"
CAT_COLS = ["db_group_last", "fl_adminarea"]
RANDOM_STATE = 42
MONTH_CONTEXT_COLS = [
    "loan_amount_last",
    "days_from_authperson_registration",
    "app_term_mean_360",
    "corp_credit_products",
    "balance_rur_amt_30_min",
    "sum_deb_ul_90",
    "sum_deb_ul_30",
    "offered_rate",
    "rate_minus_cb",
    "count_all_corp_dashboard_events",
    "p75_time_spent_minutes",
]


@dataclass
class FittedModel:
    name: str
    model: Any
    valid_pred: np.ndarray
    test_pred: np.ndarray | None = None
    auc: float | None = None
    weight: float | None = None


def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / (den.abs() + 1e-6)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    dates = pd.to_datetime(out[DATE_COL], errors="coerce")
    min_anchor = pd.Timestamp("2024-01-01")
    out["decision_year"] = dates.dt.year
    out["decision_month"] = dates.dt.month
    out["decision_dayofmonth"] = dates.dt.day
    out["decision_dayofweek"] = dates.dt.dayofweek
    out["decision_dayofyear"] = dates.dt.dayofyear
    out["decision_weekofyear"] = dates.dt.isocalendar().week.astype("float")
    out["decision_days_since_2024"] = (dates - min_anchor).dt.days
    out["decision_is_month_start"] = dates.dt.is_month_start.astype("float")
    out["decision_is_month_end"] = dates.dt.is_month_end.astype("float")
    out["decision_month_sin"] = np.sin(2 * np.pi * out["decision_month"] / 12)
    out["decision_month_cos"] = np.cos(2 * np.pi * out["decision_month"] / 12)

    source_cols = [
        c
        for c in out.columns
        if c not in {TARGET, ID_COL}
        and not c.startswith("decision_")
        and c != DATE_COL
    ]
    out["missing_count"] = out[source_cols].isna().sum(axis=1)
    out["missing_share"] = out["missing_count"] / max(len(source_cols), 1)
    for col in source_cols:
        out[f"is_missing__{col}"] = out[col].isna().astype("int8")

    bank_activity_cols = [
        "sum_deb_ul_90",
        "sum_deb_ul_30",
        "cnt_deb_loan_90",
        "cnt_deb_ul_ip_90",
        "cnt_deb_ul_ip_30",
        "balance_rur_amt_30_min",
        "cnt_cred_loan_90",
        "sum_deb_investment_90",
    ]
    corp_bundle_cols = [
        "corp_credit_products",
        "corp_list",
        "count_all_corp_dashboard_events",
        "p75_time_spent_minutes",
    ]
    term_cols = ["app_term_mean_360", "overdraft_app_term_max_360"]
    revenue_cols = ["loan_rev_max_start_non_fin", "loan_rev_min_start_fin"]
    identity_geo_cols = ["days_from_authperson_registration", "fl_adminarea"]
    missing_groups = {
        "bank_activity": bank_activity_cols,
        "corp_bundle": corp_bundle_cols,
        "term_group": term_cols,
        "revenue_group": revenue_cols,
        "identity_geo": identity_geo_cols,
    }
    for name, cols in missing_groups.items():
        available = [col for col in cols if col in out.columns]
        if available:
            out[f"{name}_missing_count"] = out[available].isna().sum(axis=1)
            out[f"{name}_all_missing"] = out[available].isna().all(axis=1).astype("int8")
            out[f"{name}_any_present"] = out[available].notna().any(axis=1).astype("int8")

    for col in [
        "sum_deb_ul_90",
        "sum_deb_ul_30",
        "cnt_deb_ul_ip_90",
        "cnt_deb_ul_ip_30",
        "count_all_corp_dashboard_events",
        "p75_time_spent_minutes",
        "corp_credit_products",
        "corp_list",
        "loan_rev_max_start_non_fin",
        "loan_rev_min_start_fin",
        "app_term_mean_360",
        "overdraft_app_term_max_360",
    ]:
        if col in out.columns:
            out[f"has_{col}"] = out[col].notna().astype("int8")
            out[f"is_positive__{col}"] = (out[col].fillna(0) > 0).astype("int8")

    out["rate_minus_cb"] = out["offered_rate"] - out["cb_rate"]
    out["rate_to_cb_abs"] = safe_divide(out["offered_rate"], out["cb_rate"])
    out["overdraft_limit_spread"] = out["overdraft_limit_max"] - out["overdraft_limit_min"]
    out["overdraft_limit_mid"] = (out["overdraft_limit_max"] + out["overdraft_limit_min"]) / 2
    out["loan_minus_od_min"] = out["loan_amount_last"] - out["overdraft_limit_min"]
    out["loan_minus_od_max"] = out["loan_amount_last"] - out["overdraft_limit_max"]
    out["loan_above_od_max"] = (out["loan_amount_last"] > out["overdraft_limit_max"]).astype("int8")
    out["loan_below_od_min"] = (out["loan_amount_last"] < out["overdraft_limit_min"]).astype("int8")
    out["loan_between_od_limits"] = (
        (out["loan_amount_last"] >= out["overdraft_limit_min"])
        & (out["loan_amount_last"] <= out["overdraft_limit_max"])
    ).astype("int8")
    out["loan_to_od_min_abs"] = safe_divide(out["loan_amount_last"], out["overdraft_limit_min"])
    out["loan_to_od_max_abs"] = safe_divide(out["loan_amount_last"], out["overdraft_limit_max"])
    out["loan_to_od_mid_abs"] = safe_divide(out["loan_amount_last"], out["overdraft_limit_mid"])

    out["sum_deb_ul_delta_90_30"] = out["sum_deb_ul_90"] - out["sum_deb_ul_30"]
    out["sum_deb_ul_prior60_per_day"] = out["sum_deb_ul_delta_90_30"] / 60
    out["sum_deb_ul_30_per_day"] = out["sum_deb_ul_30"] / 30
    out["sum_deb_ul_30_vs_prior60_per_day"] = safe_divide(
        out["sum_deb_ul_30_per_day"], out["sum_deb_ul_prior60_per_day"]
    )
    out["sum_deb_ul_30_minus_prior60_per_day"] = (
        out["sum_deb_ul_30_per_day"] - out["sum_deb_ul_prior60_per_day"]
    )
    out["sum_deb_ul_30_to_90_abs"] = safe_divide(out["sum_deb_ul_30"], out["sum_deb_ul_90"])
    out["cnt_deb_ul_ip_delta_90_30"] = out["cnt_deb_ul_ip_90"] - out["cnt_deb_ul_ip_30"]
    out["cnt_deb_ul_ip_prior60_per_day"] = out["cnt_deb_ul_ip_delta_90_30"] / 60
    out["cnt_deb_ul_ip_30_per_day"] = out["cnt_deb_ul_ip_30"] / 30
    out["cnt_deb_ul_ip_30_vs_prior60_per_day"] = safe_divide(
        out["cnt_deb_ul_ip_30_per_day"], out["cnt_deb_ul_ip_prior60_per_day"]
    )
    out["cnt_deb_ul_ip_30_minus_prior60_per_day"] = (
        out["cnt_deb_ul_ip_30_per_day"] - out["cnt_deb_ul_ip_prior60_per_day"]
    )
    out["cnt_deb_ul_ip_30_to_90_abs"] = safe_divide(out["cnt_deb_ul_ip_30"], out["cnt_deb_ul_ip_90"])
    out["cnt_deb_activity_90"] = out["cnt_deb_loan_90"] + out["cnt_deb_ul_ip_90"]
    out["cnt_credit_activity_90"] = out["cnt_cred_loan_90"] + out["cnt_deb_loan_90"]
    out["deb_ul_amount_per_tx_30"] = safe_divide(out["sum_deb_ul_30"], out["cnt_deb_ul_ip_30"])
    out["deb_ul_amount_per_tx_90"] = safe_divide(out["sum_deb_ul_90"], out["cnt_deb_ul_ip_90"])
    out["cred_to_deb_loan_cnt"] = safe_divide(out["cnt_cred_loan_90"], out["cnt_deb_loan_90"])
    out["cnt_deb_loan_share_in_deb90"] = safe_divide(out["cnt_deb_loan_90"], out["cnt_deb_activity_90"])
    out["balance_to_deb90"] = safe_divide(out["balance_rur_amt_30_min"], out["sum_deb_ul_90"])
    out["investment_to_deb90"] = safe_divide(out["sum_deb_investment_90"], out["sum_deb_ul_90"])

    out["corp_activity_total"] = (
        out["corp_credit_products"]
        + out["corp_list"]
        + out["count_all_corp_dashboard_events"]
    )
    out["financial_activity_total"] = (
        out["sum_deb_ul_90"]
        + out["sum_deb_investment_90"]
        + out["balance_rur_amt_30_min"]
    )
    out["loan_rev_start_span"] = (
        out["loan_rev_max_start_non_fin"] - out["loan_rev_min_start_fin"]
    )
    out["app_term_minus_od_term"] = (
        out["app_term_mean_360"] - out["overdraft_app_term_max_360"]
    )
    out["balance_to_loan_abs"] = safe_divide(
        out["balance_rur_amt_30_min"], out["loan_amount_last"]
    )
    out["dashboard_time_product"] = (
        out["count_all_corp_dashboard_events"] * out["p75_time_spent_minutes"]
    )

    out = out.drop(columns=[DATE_COL], errors="ignore").copy()
    for col in CAT_COLS:
        if col in out.columns:
            out[col] = out[col].astype("object").where(out[col].notna(), "__MISSING__")
    return out


def add_context_features(
    train_fe: pd.DataFrame, test_fe: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train_fe.copy()
    test_out = test_fe.copy()
    train_out["_is_train_context"] = 1
    test_out["_is_train_context"] = 0
    combined = pd.concat([train_out, test_out], ignore_index=True, sort=False)
    month_key = combined["decision_year"].astype("Int64").astype(str) + "_" + combined[
        "decision_month"
    ].astype("Int64").astype(str)

    for col in MONTH_CONTEXT_COLS:
        if col not in combined.columns:
            continue
        grouped = combined.groupby(month_key, dropna=False)[col]
        combined[f"{col}_rank_in_month"] = grouped.rank(pct=True, method="average")
        means = grouped.transform("mean")
        stds = grouped.transform("std").replace(0, np.nan)
        combined[f"{col}_z_in_month"] = (combined[col] - means) / (stds + 1e-6)

    for col in CAT_COLS:
        if col not in combined.columns:
            continue
        cat_values = combined[col].astype("object").fillna("__MISSING__")
        counts = cat_values.map(cat_values.value_counts(dropna=False))
        combined[f"{col}_count_all_log"] = np.log1p(counts.astype(float))
        combined[f"{col}_freq_all"] = counts.astype(float) / len(combined)
        month_cat = month_key.astype(str) + "__" + cat_values.astype(str)
        month_counts = month_cat.map(month_cat.value_counts(dropna=False))
        combined[f"{col}_month_count_all_log"] = np.log1p(month_counts.astype(float))

    split_pos = len(train_fe)
    train_context = combined.iloc[:split_pos].drop(columns=["_is_train_context"])
    test_context = combined.iloc[split_pos:].drop(columns=["_is_train_context"])
    test_context.index = test_fe.index
    return train_context, test_context


def feature_columns(train_fe: pd.DataFrame) -> list[str]:
    return [c for c in train_fe.columns if c not in {ID_COL, TARGET}]


def split_time_validation(train: pd.DataFrame, valid_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray, str]:
    ordered = train.sort_values(DATE_COL).reset_index()
    cutoff_pos = int(len(ordered) * (1 - valid_frac))
    cutoff_date = ordered.loc[cutoff_pos, DATE_COL]
    valid_mask = pd.to_datetime(train[DATE_COL]) >= pd.to_datetime(cutoff_date)
    train_idx = train.index[~valid_mask].to_numpy()
    valid_idx = train.index[valid_mask].to_numpy()
    return train_idx, valid_idx, str(cutoff_date)


def prepare_lgbm_frames(
    X_train: pd.DataFrame, X_valid: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train_lgb = X_train.copy()
    X_valid_lgb = X_valid.copy()
    X_test_lgb = X_test.copy()
    for col in CAT_COLS:
        if col in X_train_lgb.columns:
            categories = pd.Index(
                pd.concat(
                    [X_train_lgb[col], X_valid_lgb[col], X_test_lgb[col]],
                    ignore_index=True,
                )
                .astype("object")
                .fillna("__MISSING__")
                .unique()
            )
            for frame in [X_train_lgb, X_valid_lgb, X_test_lgb]:
                frame[col] = pd.Categorical(
                    frame[col].astype("object").fillna("__MISSING__"),
                    categories=categories,
                )
    return X_train_lgb, X_valid_lgb, X_test_lgb


def fit_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> FittedModel:
    numeric_cols = [c for c in X_train.columns if c not in CAT_COLS]
    cat_cols = [c for c in CAT_COLS if c in X_train.columns]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=30,
                                sparse_output=True,
                            ),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                LogisticRegression(
                    C=0.35,
                    max_iter=2500,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_valid)[:, 1]
    return FittedModel("logistic", model, pred, auc=roc_auc_score(y_valid, pred))


def fit_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    params: dict[str, Any],
    name: str,
) -> FittedModel:
    X_train_lgb, X_valid_lgb, X_test_lgb = prepare_lgbm_frames(X_train, X_valid, X_test)
    model = LGBMClassifier(**params)
    model.fit(
        X_train_lgb,
        y_train,
        eval_set=[(X_valid_lgb, y_valid)],
        eval_metric="auc",
        categorical_feature=[c for c in CAT_COLS if c in X_train.columns],
        callbacks=[early_stopping(120), log_evaluation(0)],
    )
    pred = model.predict_proba(X_valid_lgb)[:, 1]
    test_pred = model.predict_proba(X_test_lgb)[:, 1]
    return FittedModel(name, model, pred, test_pred, roc_auc_score(y_valid, pred))


def fit_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    params: dict[str, Any],
    name: str,
) -> FittedModel:
    cat_features = [c for c in CAT_COLS if c in X_train.columns]
    model = CatBoostClassifier(**params)
    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        verbose=False,
    )
    pred = model.predict_proba(X_valid)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]
    return FittedModel(name, model, pred, test_pred, roc_auc_score(y_valid, pred))


def fit_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    params: dict[str, Any],
    name: str,
) -> FittedModel:
    numeric_cols = [c for c in X_train.columns if c not in CAT_COLS]
    cat_cols = [c for c in CAT_COLS if c in X_train.columns]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=30, sparse_output=True),
                cat_cols,
            ),
        ],
        sparse_threshold=1.0,
    )
    X_train_xgb = preprocessor.fit_transform(X_train)
    X_valid_xgb = preprocessor.transform(X_valid)
    X_test_xgb = preprocessor.transform(X_test)
    model = XGBClassifier(**params)
    model.fit(X_train_xgb, y_train, eval_set=[(X_valid_xgb, y_valid)], verbose=False)
    pred = model.predict_proba(X_valid_xgb)[:, 1]
    test_pred = model.predict_proba(X_test_xgb)[:, 1]
    return FittedModel(name, (preprocessor, model), pred, test_pred, roc_auc_score(y_valid, pred))


def transform_predictions(preds: list[np.ndarray], method: str) -> list[np.ndarray]:
    if method == "probability":
        return preds
    if method == "rank":
        return [
            pd.Series(pred).rank(pct=True, method="average").to_numpy(dtype=float)
            for pred in preds
        ]
    if method == "logit":
        eps = 1e-6
        return [np.log(np.clip(pred, eps, 1 - eps) / np.clip(1 - pred, eps, 1 - eps)) for pred in preds]
    raise ValueError(f"Unknown blend method: {method}")


def finish_blend(pred: np.ndarray, method: str) -> np.ndarray:
    if method == "logit":
        return 1 / (1 + np.exp(-pred))
    return pred


def blend_prediction_dict(
    pred_dict: dict[str, np.ndarray],
    weights: dict[str, float],
    method: str,
) -> np.ndarray:
    active_names = [name for name in pred_dict if weights.get(name, 0) > 0]
    total_weight = sum(weights[name] for name in active_names)
    if not active_names or total_weight <= 0:
        raise ValueError("No active predictions to blend.")
    normalized_weights = {name: weights[name] / total_weight for name in active_names}
    transformed = {
        name: pred
        for name, pred in zip(
            active_names,
            transform_predictions([pred_dict[name] for name in active_names], method),
        )
    }
    raw = sum(normalized_weights[name] * transformed[name] for name in active_names)
    return np.clip(finish_blend(raw, method), 0, 1)


def best_blend(
    models: list[FittedModel], y_valid: pd.Series
) -> tuple[dict[str, float], float, np.ndarray, str]:
    candidates = sorted(
        [m for m in models if m.name != "logistic"],
        key=lambda m: m.auc or -1,
        reverse=True,
    )
    if len(candidates) < 2:
        model = max(models, key=lambda m: m.auc or -1)
        return {model.name: 1.0}, float(model.auc or 0), model.valid_pred, "probability"

    best_auc = float(candidates[0].auc or -np.inf)
    best_weights: dict[str, float] = {candidates[0].name: 1.0}
    best_pred = candidates[0].valid_pred
    best_method = "probability"

    for method in ["probability", "rank", "logit"]:
        transformed_by_name = {
            model.name: pred
            for model, pred in zip(
                candidates,
                transform_predictions([model.valid_pred for model in candidates], method),
            )
        }
        if len(candidates) == 2:
            first, second = candidates
            for w in np.linspace(0, 1, 101):
                raw_pred = w * transformed_by_name[first.name] + (1 - w) * transformed_by_name[second.name]
                pred = finish_blend(raw_pred, method)
                auc = roc_auc_score(y_valid, pred)
                if auc > best_auc:
                    best_auc = auc
                    best_weights = {first.name: float(w), second.name: float(1 - w)}
                    best_pred = pred
                    best_method = method
            continue

        selected = [candidates[0]]
        selected_pred = transformed_by_name[candidates[0].name]
        selected_auc = roc_auc_score(y_valid, finish_blend(selected_pred, method))
        for candidate in candidates[1:]:
            trial_pred = (
                selected_pred * len(selected) + transformed_by_name[candidate.name]
            ) / (len(selected) + 1)
            trial_auc = roc_auc_score(y_valid, finish_blend(trial_pred, method))
            if trial_auc > selected_auc + 1e-5:
                selected.append(candidate)
                selected_pred = trial_pred
                selected_auc = trial_auc
        if selected_auc > best_auc:
            best_auc = selected_auc
            best_pred = finish_blend(selected_pred, method)
            best_weights = {model.name: float(1 / len(selected)) for model in selected}
            best_method = method

        top_candidates = candidates[: min(5, len(candidates))]
        for i in range(len(top_candidates)):
            for j in range(i + 1, len(top_candidates)):
                first, second = top_candidates[i], top_candidates[j]
                for w in np.linspace(0, 1, 101):
                    raw_pred = (
                        w * transformed_by_name[first.name]
                        + (1 - w) * transformed_by_name[second.name]
                    )
                    pred = finish_blend(raw_pred, method)
                    auc = roc_auc_score(y_valid, pred)
                    if auc > best_auc:
                        best_auc = auc
                        best_weights = {first.name: float(w), second.name: float(1 - w)}
                        best_pred = pred
                        best_method = method

        trio_candidates = top_candidates[:3]
        names = [m.name for m in trio_candidates]
        for w1 in np.linspace(0, 1, 41):
            for w2 in np.linspace(0, 1 - w1, 41):
                remaining = 1 - w1 - w2
                weights = [w1, w2, remaining]
                raw_pred = sum(w * transformed_by_name[m.name] for w, m in zip(weights, trio_candidates))
                pred = finish_blend(raw_pred, method)
                auc = roc_auc_score(y_valid, pred)
                if auc > best_auc:
                    best_auc = auc
                    best_weights = {name: float(w) for name, w in zip(names[:3], weights)}
                    best_pred = pred
                    best_method = method
    return best_weights, float(best_auc), best_pred, best_method


def fit_final_lgbm(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    X_test: pd.DataFrame,
    valid_best_model: LGBMClassifier,
    base_params: dict[str, Any],
) -> np.ndarray:
    params = base_params.copy()
    best_iter = getattr(valid_best_model, "best_iteration_", None)
    if best_iter:
        params["n_estimators"] = max(100, int(best_iter * 1.12))
    model = LGBMClassifier(**params)
    X_full_lgb, _, X_test_lgb = prepare_lgbm_frames(X_full, X_full.iloc[:1].copy(), X_test)
    model.fit(
        X_full_lgb,
        y_full,
        categorical_feature=[c for c in CAT_COLS if c in X_full.columns],
    )
    return model.predict_proba(X_test_lgb)[:, 1]


def fit_final_catboost(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    X_test: pd.DataFrame,
    valid_best_model: CatBoostClassifier,
    base_params: dict[str, Any],
) -> np.ndarray:
    params = base_params.copy()
    best_iter = valid_best_model.get_best_iteration()
    if best_iter is not None and best_iter > 0:
        params["iterations"] = max(100, int((best_iter + 1) * 1.12))
    params.pop("od_wait", None)
    params.pop("od_type", None)
    params.pop("use_best_model", None)
    model = CatBoostClassifier(**params)
    model.fit(
        X_full,
        y_full,
        cat_features=[c for c in CAT_COLS if c in X_full.columns],
        verbose=False,
    )
    return model.predict_proba(X_test)[:, 1]


def fit_final_xgb(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    X_test: pd.DataFrame,
    base_params: dict[str, Any],
) -> np.ndarray:
    numeric_cols = [c for c in X_full.columns if c not in CAT_COLS]
    cat_cols = [c for c in CAT_COLS if c in X_full.columns]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=30, sparse_output=True),
                cat_cols,
            ),
        ],
        sparse_threshold=1.0,
    )
    X_full_xgb = preprocessor.fit_transform(X_full)
    X_test_xgb = preprocessor.transform(X_test)
    model = XGBClassifier(**base_params)
    model.fit(X_full_xgb, y_full, verbose=False)
    return model.predict_proba(X_test_xgb)[:, 1]


def write_submission(
    sample: pd.DataFrame,
    test: pd.DataFrame,
    train: pd.DataFrame,
    y: pd.Series,
    test_pred: np.ndarray,
    path: Path,
) -> pd.DataFrame:
    test_pred_frame = pd.DataFrame({ID_COL: test[ID_COL], TARGET: np.clip(test_pred, 0, 1)})
    train_label_frame = train[[ID_COL, TARGET]].copy()
    submission = sample[[ID_COL]].merge(test_pred_frame, on=ID_COL, how="left")
    train_overlap_mask = submission[TARGET].isna()
    submission.loc[train_overlap_mask, TARGET] = submission.loc[train_overlap_mask, ID_COL].map(
        train_label_frame.set_index(ID_COL)[TARGET]
    )
    if submission[TARGET].isna().any():
        submission[TARGET] = submission[TARGET].fillna(float(y.mean()))
    submission.to_csv(path, index=False)
    return submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--valid-frac", type=float, default=0.2)
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train_apps.csv")
    test = pd.read_csv(args.data_dir / "test_apps.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")

    train_idx, valid_idx, cutoff_date = split_time_validation(train, args.valid_frac)

    train_fe = add_features(train)
    test_fe = add_features(test)
    train_fe, test_fe = add_context_features(train_fe, test_fe)
    features = feature_columns(train_fe)

    X = train_fe[features]
    y = train_fe[TARGET].astype(int)
    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_valid, y_valid = X.loc[valid_idx], y.loc[valid_idx]
    X_test = test_fe[features]

    lgbm_params = {
        "objective": "binary",
        "n_estimators": 4000,
        "learning_rate": 0.015,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 100,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.82,
        "reg_alpha": 1.0,
        "reg_lambda": 4.0,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }
    lgbm_deep_params = {
        **lgbm_params,
        "learning_rate": 0.012,
        "num_leaves": 47,
        "min_child_samples": 70,
        "reg_alpha": 1.2,
        "reg_lambda": 6.0,
        "subsample": 0.9,
        "colsample_bytree": 0.76,
        "random_state": RANDOM_STATE + 7,
    }
    xgb_params = {
        "n_estimators": 1200,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "max_depth": 5,
        "learning_rate": 0.02,
        "min_child_weight": 40,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.5,
        "reg_lambda": 10,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    xgb_shallow_params = {
        **xgb_params,
        "max_depth": 3,
        "learning_rate": 0.015,
        "min_child_weight": 15,
        "reg_lambda": 3,
        "scale_pos_weight": 6.376620370370371,
        "random_state": RANDOM_STATE + 17,
    }
    cat_common = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 3500,
        "od_type": "Iter",
        "od_wait": 180,
        "allow_writing_files": False,
        "thread_count": -1,
    }
    cat_param_grid = {
        "cat_sqrt_d6_seed51": {
            **cat_common,
            "learning_rate": 0.02,
            "depth": 6,
            "l2_leaf_reg": 12,
            "auto_class_weights": "SqrtBalanced",
            "random_seed": 51,
        },
        "cat_sqrt_d6_seed42": {
            **cat_common,
            "learning_rate": 0.02,
            "depth": 6,
            "l2_leaf_reg": 12,
            "auto_class_weights": "SqrtBalanced",
            "random_seed": 42,
        },
        "cat_sqrt_d6_seed47": {
            **cat_common,
            "learning_rate": 0.02,
            "depth": 6,
            "l2_leaf_reg": 12,
            "auto_class_weights": "SqrtBalanced",
            "random_seed": 47,
        },
        "cat_sqrt_d6_seed55": {
            **cat_common,
            "learning_rate": 0.02,
            "depth": 6,
            "l2_leaf_reg": 12,
            "auto_class_weights": "SqrtBalanced",
            "random_seed": 55,
        },
        "cat_sqrt_d7_seed77": {
            **cat_common,
            "learning_rate": 0.02,
            "depth": 7,
            "l2_leaf_reg": 14,
            "auto_class_weights": "SqrtBalanced",
            "random_seed": 77,
        },
        "cat_sqrt_d7_seed99": {
            **cat_common,
            "learning_rate": 0.02,
            "depth": 7,
            "l2_leaf_reg": 14,
            "auto_class_weights": "SqrtBalanced",
            "random_seed": 99,
        },
        "cat_sqrt_d8_seed77": {
            **cat_common,
            "learning_rate": 0.016,
            "depth": 8,
            "l2_leaf_reg": 20,
            "auto_class_weights": "SqrtBalanced",
            "random_seed": 77,
        },
        "cat_sqrt_d8_seed123": {
            **cat_common,
            "learning_rate": 0.016,
            "depth": 8,
            "l2_leaf_reg": 20,
            "auto_class_weights": "SqrtBalanced",
            "random_seed": 123,
        },
        "cat_plain_d6_seed77": {
            **cat_common,
            "learning_rate": 0.025,
            "depth": 6,
            "l2_leaf_reg": 10,
            "random_seed": 77,
        },
        "cat_plain_d6_seed46": {
            **cat_common,
            "learning_rate": 0.025,
            "depth": 6,
            "l2_leaf_reg": 10,
            "random_seed": 46,
        },
    }
    lgbm_param_grid = {"lgbm": lgbm_params, "lgbm_deep": lgbm_deep_params}
    xgb_param_grid = {"xgb": xgb_params, "xgb_shallow": xgb_shallow_params}

    models: list[FittedModel] = []
    models.append(fit_logistic(X_train, y_train, X_valid, y_valid))
    for name, params in lgbm_param_grid.items():
        models.append(fit_lgbm(X_train, y_train, X_valid, y_valid, X_test, params, name))
    for name, params in xgb_param_grid.items():
        models.append(fit_xgb(X_train, y_train, X_valid, y_valid, X_test, params, name))
    for name, params in cat_param_grid.items():
        models.append(fit_catboost(X_train, y_train, X_valid, y_valid, X_test, params, name))

    weights, blend_auc, _, blend_method = best_blend(models, y_valid)

    final_test_preds: dict[str, np.ndarray] = {}
    fitted_by_name = {m.name: m for m in models}
    for name, weight in weights.items():
        if weight <= 0:
            continue
        if name in lgbm_param_grid:
            final_test_preds[name] = fit_final_lgbm(
                X, y, X_test, fitted_by_name[name].model, lgbm_param_grid[name]
            )
        elif name in xgb_param_grid:
            final_test_preds[name] = fit_final_xgb(X, y, X_test, xgb_param_grid[name])
        elif name in cat_param_grid:
            final_test_preds[name] = fit_final_catboost(
                X, y, X_test, fitted_by_name[name].model, cat_param_grid[name]
            )

    if not final_test_preds:
        best_model = max(models, key=lambda m: m.auc or -1)
        final_test = best_model.test_pred
        weights = {best_model.name: 1.0}
        blend_method = "probability"
    else:
        final_test = blend_prediction_dict(final_test_preds, weights, blend_method)

    final_test = np.clip(final_test, 0.0, 1.0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = args.output_dir / "submission.csv"
    model_only_path = args.output_dir / "submission_model_only.csv"
    report_path = args.output_dir / "validation_report.json"
    importance_path = args.output_dir / "feature_importance.csv"

    submission = write_submission(sample, test, train, y, final_test, submission_path)
    test_pred_frame = pd.DataFrame({ID_COL: test[ID_COL], TARGET: final_test})
    model_only_submission = sample[[ID_COL]].merge(test_pred_frame, on=ID_COL, how="left")
    model_only_submission[TARGET] = model_only_submission[TARGET].fillna(float(y.mean()))
    model_only_submission.to_csv(model_only_path, index=False)

    candidate_paths: dict[str, str] = {}
    if final_test_preds:
        selected_names = list(final_test_preds)
        candidate_specs: dict[str, tuple[list[str], str]] = {
            "auto_rank": (selected_names, "rank"),
            "auto_probability": (selected_names, "probability"),
            "auto_logit": (selected_names, "logit"),
            "cat_only_rank": ([name for name in selected_names if name.startswith("cat_")], "rank"),
            "cat_only_probability": (
                [name for name in selected_names if name.startswith("cat_")],
                "probability",
            ),
            "cat_xgb_rank": (
                [
                    name
                    for name in selected_names
                    if name.startswith("cat_") or name.startswith("xgb")
                ],
                "rank",
            ),
            "cat_xgb_probability": (
                [
                    name
                    for name in selected_names
                    if name.startswith("cat_") or name.startswith("xgb")
                ],
                "probability",
            ),
        }
        for candidate_name, (names, method) in candidate_specs.items():
            names = [name for name in names if name in final_test_preds]
            if not names:
                continue
            candidate_weights = {name: 1 / len(names) for name in names}
            candidate_pred = blend_prediction_dict(
                {name: final_test_preds[name] for name in names},
                candidate_weights,
                method,
            )
            candidate_path = args.output_dir / f"submission_{candidate_name}.csv"
            write_submission(sample, test, train, y, candidate_pred, candidate_path)
            candidate_paths[candidate_name] = str(candidate_path.resolve())

        pd.DataFrame({ID_COL: test[ID_COL], **final_test_preds}).to_csv(
            args.output_dir / "final_model_test_predictions.csv",
            index=False,
        )

    importances: list[pd.DataFrame] = []
    for model in models:
        if model.name.startswith("lgbm"):
            imp = pd.DataFrame(
                {
                    "feature": features,
                    "importance": model.model.feature_importances_,
                    "model": model.name,
                }
            )
            importances.append(imp)
        elif model.name.startswith("catboost"):
            imp = pd.DataFrame(
                {
                    "feature": features,
                    "importance": model.model.get_feature_importance(),
                    "model": model.name,
                }
            )
            importances.append(imp)
    if importances:
        pd.concat(importances, ignore_index=True).to_csv(importance_path, index=False)

    random_train_idx, random_valid_idx = train_test_split(
        np.arange(len(X)), test_size=args.valid_frac, random_state=RANDOM_STATE, stratify=y
    )
    random_auc_check = None
    try:
        random_model = fit_lgbm(
            X.iloc[random_train_idx],
            y.iloc[random_train_idx],
            X.iloc[random_valid_idx],
            y.iloc[random_valid_idx],
            X_test,
            {**lgbm_params, "n_estimators": 1500},
            "lgbm_random_holdout_check",
        )
        random_auc_check = random_model.auc
    except Exception as exc:
        random_auc_check = f"failed: {exc}"

    report = {
        "metric": "ROC-AUC",
        "time_validation_cutoff_date": cutoff_date,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "sample_rows": int(len(sample)),
        "target_rate": float(y.mean()),
        "positive_count": int(y.sum()),
        "feature_count": int(len(features)),
        "sample_ids_from_test": int(sample[ID_COL].isin(test[ID_COL]).sum()),
        "sample_ids_from_train": int(sample[ID_COL].isin(train[ID_COL]).sum()),
        "train_overlap_strategy": "known train target_value used for sample IDs missing from test_apps.csv",
        "models": {m.name: float(m.auc) for m in models},
        "blend_weights": weights,
        "blend_method": blend_method,
        "blend_time_auc": blend_auc,
        "random_lgbm_auc_check": random_auc_check,
        "submission_path": str(submission_path.resolve()),
        "candidate_submission_paths": candidate_paths,
        "model_only_submission_path": str(model_only_path.resolve()),
        "feature_importance_path": str(importance_path.resolve()),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
