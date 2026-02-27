#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature_set_v1 전용 전처리 + 학습 파이프라인

입력:
  - feature_sets/data/feature_set_v1_train_core.csv
  - feature_sets/data/feature_set_v1_test_core.csv
  - feature_sets/data/feature_set_v1_train_core_plus_exp.csv
  - feature_sets/data/feature_set_v1_test_core_plus_exp.csv
  - feature_sets/feature_set_v1_spec.json

출력 (기본):
  model_output_v1/<timestamp>/
    - core/fold_metrics.csv
    - core/model_metrics.csv
    - core/oof_pred_best.csv
    - core/test_pred_best.csv
    - core/submission_best.csv
    - core/oof_pred_blend.csv
    - core/test_pred_blend.csv
    - core/submission_blend.csv
    - core/feature_overview.json
    - core_plus_exp/... (동일 구조)
    - run_summary.json
"""

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def normalize_token(text: str) -> str:
    t = re.sub(r"\s+", " ", str(text)).strip().lower()
    return t


def split_multiselect(value) -> List[str]:
    if pd.isna(value):
        return []
    s = str(value).replace("\n", " ").strip()
    if not s or s.lower() == "nan":
        return []
    parts = re.split(r"\s*,\s*", s)
    out = []
    seen = set()
    for p in parts:
        tok = normalize_token(p)
        if not tok or tok == "nan":
            continue
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


class MultiSelectVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List[str], min_count: int = 5, max_features_per_col: int = 120):
        self.columns = columns
        self.min_count = min_count
        self.max_features_per_col = max_features_per_col
        self.vocab_by_col_: Dict[str, List[str]] = {}
        self.feature_names_: List[str] = []
        self.feature_index_: Dict[Tuple[str, str], int] = {}

    def fit(self, X, y=None):
        X_df = self._to_df(X)
        self.vocab_by_col_ = {}
        self.feature_names_ = []
        self.feature_index_ = {}
        idx = 0

        for c in self.columns:
            if c not in X_df.columns:
                self.vocab_by_col_[c] = []
                continue
            counts = {}
            for v in X_df[c].values:
                for tok in split_multiselect(v):
                    counts[tok] = counts.get(tok, 0) + 1

            toks = [k for k, v in counts.items() if v >= self.min_count]
            toks = sorted(toks, key=lambda t: (-counts[t], t))
            toks = toks[: self.max_features_per_col]
            self.vocab_by_col_[c] = toks

            for tok in toks:
                fname = f"{c}__ms__{tok}"
                self.feature_names_.append(fname)
                self.feature_index_[(c, tok)] = idx
                idx += 1
        return self

    def transform(self, X):
        X_df = self._to_df(X)
        n = X_df.shape[0]
        m = len(self.feature_names_)
        if m == 0:
            return sp.csr_matrix((n, 0), dtype=np.float32)

        rows, cols, data = [], [], []
        for i in range(n):
            for c in self.columns:
                if c not in X_df.columns:
                    continue
                val = X_df.iloc[i][c]
                toks = split_multiselect(val)
                for tok in toks:
                    key = (c, tok)
                    j = self.feature_index_.get(key)
                    if j is not None:
                        rows.append(i)
                        cols.append(j)
                        data.append(1.0)
        return sp.csr_matrix((data, (rows, cols)), shape=(n, m), dtype=np.float32)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_, dtype=object)

    def _to_df(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=self.columns)


def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(X: pd.DataFrame, spec: dict) -> Tuple[ColumnTransformer, dict]:
    multiselect_cols = [c for c in spec.get("multiselect_parse_cols", []) if c in X.columns]
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and c not in multiselect_cols]
    cat_cols = [c for c in X.columns if c not in numeric_cols and c not in multiselect_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_onehot()),
    ])

    ms_pipe = Pipeline([
        ("ms", MultiSelectVectorizer(columns=multiselect_cols, min_count=5, max_features_per_col=120)),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", num_pipe, numeric_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    if multiselect_cols:
        transformers.append(("ms", ms_pipe, multiselect_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.2,
    )

    info = {
        "numeric_cols": numeric_cols,
        "categorical_cols": cat_cols,
        "multiselect_cols": multiselect_cols,
    }
    return preprocessor, info


@dataclass
class ModelSpec:
    name: str
    estimator: str
    kwargs: dict


def model_specs() -> List[ModelSpec]:
    return [
        ModelSpec(
            name="lr_l2_balanced",
            estimator="logreg",
            kwargs={
                "solver": "saga",
                "penalty": "l2",
                "C": 1.0,
                "max_iter": 4000,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
            },
        ),
        ModelSpec(
            name="lr_l1_balanced",
            estimator="logreg",
            kwargs={
                "solver": "saga",
                "penalty": "l1",
                "C": 0.5,
                "max_iter": 5000,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
            },
        ),
        ModelSpec(
            name="lr_l2_unweighted",
            estimator="logreg",
            kwargs={
                "solver": "saga",
                "penalty": "l2",
                "C": 0.8,
                "max_iter": 4000,
                "class_weight": None,
                "random_state": 42,
                "n_jobs": -1,
            },
        ),
        ModelSpec(
            name="sgd_logloss_balanced",
            estimator="sgd",
            kwargs={
                "loss": "log_loss",
                "penalty": "elasticnet",
                "alpha": 1e-4,
                "l1_ratio": 0.15,
                "max_iter": 5000,
                "tol": 1e-4,
                "class_weight": "balanced",
                "random_state": 42,
            },
        ),
        ModelSpec(
            name="sgd_logloss_unweighted",
            estimator="sgd",
            kwargs={
                "loss": "log_loss",
                "penalty": "elasticnet",
                "alpha": 2e-4,
                "l1_ratio": 0.10,
                "max_iter": 5000,
                "tol": 1e-4,
                "class_weight": None,
                "random_state": 42,
            },
        ),
    ]


def make_estimator(m: ModelSpec):
    if m.estimator == "logreg":
        return LogisticRegression(**m.kwargs)
    if m.estimator == "sgd":
        return SGDClassifier(**m.kwargs)
    raise ValueError(f"unknown estimator type: {m.estimator}")


def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_pos_rate: float,
    lambda_rate: float = 0.15,
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    grid = np.linspace(0.05, 0.95, 181)
    q = np.quantile(y_prob, np.linspace(0.05, 0.95, 19))
    cand = np.unique(np.clip(np.concatenate([grid, q]), 0.001, 0.999))

    best_t_f1 = 0.5
    best_f1 = -1.0
    best_t_reg = 0.5
    best_f1_reg = -1.0
    best_reg_obj = -1.0
    best_pos_rate_reg = 0.0

    for t in cand:
        pred = (y_prob >= t).astype(int)
        f = f1_score(y_true, pred, zero_division=0)
        pos_rate = float(np.mean(pred))
        reg_obj = f - lambda_rate * abs(pos_rate - target_pos_rate)

        if (f > best_f1 + 1e-12) or (abs(f - best_f1) <= 1e-12 and abs(t - 0.5) < abs(best_t_f1 - 0.5)):
            best_f1 = float(f)
            best_t_f1 = float(t)

        if (reg_obj > best_reg_obj + 1e-12) or (abs(reg_obj - best_reg_obj) <= 1e-12 and abs(t - 0.5) < abs(best_t_reg - 0.5)):
            best_reg_obj = float(reg_obj)
            best_f1_reg = float(f)
            best_t_reg = float(t)
            best_pos_rate_reg = float(pos_rate)

    return {
        "threshold_f1": best_t_f1,
        "cv_f1": best_f1,
        "threshold_reg": best_t_reg,
        "cv_f1_reg": best_f1_reg,
        "cv_objective_reg": best_reg_obj,
        "pred_pos_rate_reg": best_pos_rate_reg,
        "target_pos_rate": float(target_pos_rate),
    }


def fit_cv_predict(
    X_train: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    spec: dict,
    n_splits: int = 5,
    random_state: int = 42,
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    target_pos_rate = float(np.mean(y))

    fold_rows = []
    model_rows = []
    per_model_oof = {}
    per_model_test = {}
    preproc_overview = None

    for m in model_specs():
        oof = np.zeros(len(X_train), dtype=float)
        test_fold_preds = []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), start=1):
            X_tr = X_train.iloc[tr_idx].copy()
            y_tr = y.iloc[tr_idx].copy()
            X_va = X_train.iloc[va_idx].copy()
            y_va = y.iloc[va_idx].copy()

            preprocessor, pinfo = build_preprocessor(X_tr, spec)
            if preproc_overview is None:
                preproc_overview = pinfo

            clf = make_estimator(m)
            pipe = Pipeline([
                ("prep", preprocessor),
                ("clf", clf),
            ])
            pipe.fit(X_tr, y_tr)

            va_pred = pipe.predict_proba(X_va)[:, 1]
            oof[va_idx] = va_pred
            test_pred = pipe.predict_proba(X_test)[:, 1]
            test_fold_preds.append(test_pred)

            fold_auc = roc_auc_score(y_va, va_pred)
            fold_ap = average_precision_score(y_va, va_pred)
            fold_rows.append({
                "model": m.name,
                "fold": fold,
                "fold_auc": float(fold_auc),
                "fold_ap": float(fold_ap),
                "n_train": int(len(tr_idx)),
                "n_valid": int(len(va_idx)),
            })

        test_mean = np.mean(np.vstack(test_fold_preds), axis=0)
        auc = roc_auc_score(y, oof)
        ap = average_precision_score(y, oof)
        tinfo = optimize_threshold(y.to_numpy(), oof, target_pos_rate=target_pos_rate, lambda_rate=0.15)
        model_rows.append({
            "model": m.name,
            "cv_auc": float(auc),
            "cv_ap": float(ap),
            "cv_f1": float(tinfo["cv_f1"]),
            "threshold_f1": float(tinfo["threshold_f1"]),
            "cv_f1_reg": float(tinfo["cv_f1_reg"]),
            "threshold_reg": float(tinfo["threshold_reg"]),
            "cv_objective_reg": float(tinfo["cv_objective_reg"]),
            "pred_pos_rate_reg": float(tinfo["pred_pos_rate_reg"]),
            "target_pos_rate": float(tinfo["target_pos_rate"]),
            "n_splits": n_splits,
        })
        per_model_oof[m.name] = oof
        per_model_test[m.name] = test_mean

    model_df = pd.DataFrame(model_rows).sort_values(
        ["cv_f1_reg", "cv_auc", "cv_ap"],
        ascending=False
    ).reset_index(drop=True)
    fold_df = pd.DataFrame(fold_rows).sort_values(["model", "fold"]).reset_index(drop=True)

    best_model = model_df.iloc[0]["model"]
    oof_best = per_model_oof[best_model]
    test_best = per_model_test[best_model]

    # simple mean blend
    oof_blend = np.mean(np.vstack([per_model_oof[k] for k in per_model_oof]), axis=0)
    test_blend = np.mean(np.vstack([per_model_test[k] for k in per_model_test]), axis=0)
    blend_auc = roc_auc_score(y, oof_blend)
    blend_ap = average_precision_score(y, oof_blend)
    blend_tinfo = optimize_threshold(y.to_numpy(), oof_blend, target_pos_rate=target_pos_rate, lambda_rate=0.15)

    return {
        "fold_df": fold_df,
        "model_df": model_df,
        "best_model": best_model,
        "oof_best": oof_best,
        "test_best": test_best,
        "oof_blend": oof_blend,
        "test_blend": test_blend,
        "blend_auc": float(blend_auc),
        "blend_ap": float(blend_ap),
        "blend_f1": float(blend_tinfo["cv_f1"]),
        "blend_threshold_f1": float(blend_tinfo["threshold_f1"]),
        "blend_f1_reg": float(blend_tinfo["cv_f1_reg"]),
        "blend_threshold_reg": float(blend_tinfo["threshold_reg"]),
        "blend_objective_reg": float(blend_tinfo["cv_objective_reg"]),
        "blend_pred_pos_rate_reg": float(blend_tinfo["pred_pos_rate_reg"]),
        "preproc_overview": preproc_overview or {},
    }


def run_dataset_mode(
    mode: str,
    train_csv: Path,
    test_csv: Path,
    spec: dict,
    out_dir: Path,
):
    df_tr = pd.read_csv(train_csv)
    df_te = pd.read_csv(test_csv)

    id_col = spec["id_column"]
    target_col = spec["target"]
    if id_col not in df_tr.columns or id_col not in df_te.columns:
        raise ValueError(f"[{mode}] id column not found: {id_col}")
    if target_col not in df_tr.columns:
        raise ValueError(f"[{mode}] target column not found: {target_col}")

    X_train = df_tr.drop(columns=[id_col, target_col])
    y = df_tr[target_col].astype(int)
    X_test = df_te.drop(columns=[id_col], errors="ignore")

    if X_train.columns.tolist() != X_test.columns.tolist():
        raise ValueError(f"[{mode}] train/test feature columns mismatch")

    result = fit_cv_predict(X_train, y, X_test, spec, n_splits=5, random_state=42)

    out_mode = out_dir / mode
    out_mode.mkdir(parents=True, exist_ok=True)

    # metrics
    result["fold_df"].to_csv(out_mode / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    model_df = result["model_df"].copy()
    best_threshold_f1 = float(model_df.iloc[0]["threshold_f1"])
    best_threshold_reg = float(model_df.iloc[0]["threshold_reg"])
    best_f1 = float(model_df.iloc[0]["cv_f1"])
    best_f1_reg = float(model_df.iloc[0]["cv_f1_reg"])
    blend_row = pd.DataFrame([{
        "model": "blend_mean",
        "cv_auc": result["blend_auc"],
        "cv_ap": result["blend_ap"],
        "cv_f1": result["blend_f1"],
        "threshold_f1": result["blend_threshold_f1"],
        "cv_f1_reg": result["blend_f1_reg"],
        "threshold_reg": result["blend_threshold_reg"],
        "cv_objective_reg": result["blend_objective_reg"],
        "pred_pos_rate_reg": result["blend_pred_pos_rate_reg"],
        "target_pos_rate": float(np.mean(y)),
        "n_splits": 5,
    }])
    model_df = pd.concat([model_df, blend_row], ignore_index=True)
    model_df.to_csv(out_mode / "model_metrics.csv", index=False, encoding="utf-8-sig")

    # predictions
    best_model = result["best_model"]
    oof_best = pd.DataFrame({
        id_col: df_tr[id_col],
        target_col: y,
        "pred_proba": result["oof_best"],
        "model": best_model,
    })
    oof_best.to_csv(out_mode / "oof_pred_best.csv", index=False, encoding="utf-8-sig")

    test_best = pd.DataFrame({
        id_col: df_te[id_col],
        "pred_proba": result["test_best"],
        "model": best_model,
    })
    test_best.to_csv(out_mode / "test_pred_best.csv", index=False, encoding="utf-8-sig")

    sub_best = pd.DataFrame({
        id_col: df_te[id_col],
        target_col: (result["test_best"] >= best_threshold_reg).astype(int),
    })
    sub_best.to_csv(out_mode / "submission_best.csv", index=False, encoding="utf-8-sig")

    sub_best_f1 = pd.DataFrame({
        id_col: df_te[id_col],
        target_col: (result["test_best"] >= best_threshold_f1).astype(int),
    })
    sub_best_f1.to_csv(out_mode / "submission_best_f1max.csv", index=False, encoding="utf-8-sig")

    oof_blend = pd.DataFrame({
        id_col: df_tr[id_col],
        target_col: y,
        "pred_proba": result["oof_blend"],
        "model": "blend_mean",
    })
    oof_blend.to_csv(out_mode / "oof_pred_blend.csv", index=False, encoding="utf-8-sig")

    test_blend = pd.DataFrame({
        id_col: df_te[id_col],
        "pred_proba": result["test_blend"],
        "model": "blend_mean",
    })
    test_blend.to_csv(out_mode / "test_pred_blend.csv", index=False, encoding="utf-8-sig")

    sub_blend = pd.DataFrame({
        id_col: df_te[id_col],
        target_col: (result["test_blend"] >= result["blend_threshold_reg"]).astype(int),
    })
    sub_blend.to_csv(out_mode / "submission_blend.csv", index=False, encoding="utf-8-sig")

    sub_blend_f1 = pd.DataFrame({
        id_col: df_te[id_col],
        target_col: (result["test_blend"] >= result["blend_threshold_f1"]).astype(int),
    })
    sub_blend_f1.to_csv(out_mode / "submission_blend_f1max.csv", index=False, encoding="utf-8-sig")

    # 선택 규칙(모드 내): CV F1(reg) 우선, 동률은 AUC/AP 우선
    best_auc = float(result["model_df"].iloc[0]["cv_auc"])
    best_ap = float(result["model_df"].iloc[0]["cv_ap"])
    best_f1 = float(result["model_df"].iloc[0]["cv_f1"])
    best_f1_reg = float(result["model_df"].iloc[0]["cv_f1_reg"])
    blend_auc = float(result["blend_auc"])
    blend_ap = float(result["blend_ap"])
    blend_f1 = float(result["blend_f1"])
    blend_f1_reg = float(result["blend_f1_reg"])

    if (blend_f1_reg > best_f1_reg + 1e-6) or (
        abs(blend_f1_reg - best_f1_reg) <= 1e-6 and (
            (blend_auc > best_auc + 1e-6) or (abs(blend_auc - best_auc) <= 1e-6 and blend_ap > best_ap)
        )
    ):
        selected_variant = "blend"
        selected_cv_f1 = blend_f1
        selected_cv_f1_reg = blend_f1_reg
        selected_cv_auc = blend_auc
        selected_cv_ap = blend_ap
        selected_threshold = float(result["blend_threshold_reg"])
        selected_threshold_f1max = float(result["blend_threshold_f1"])
        selected_submission_path = out_mode / "submission_blend.csv"
        selected_oof_path = out_mode / "oof_pred_blend.csv"
        selected_submission_f1max_path = out_mode / "submission_blend_f1max.csv"
    else:
        selected_variant = "best_single"
        selected_cv_f1 = best_f1
        selected_cv_f1_reg = best_f1_reg
        selected_cv_auc = best_auc
        selected_cv_ap = best_ap
        selected_threshold = best_threshold_reg
        selected_threshold_f1max = best_threshold_f1
        selected_submission_path = out_mode / "submission_best.csv"
        selected_oof_path = out_mode / "oof_pred_best.csv"
        selected_submission_f1max_path = out_mode / "submission_best_f1max.csv"

    # 모드별 selected 파일 별도 고정
    shutil.copyfile(selected_submission_path, out_mode / "submission_selected.csv")
    shutil.copyfile(selected_oof_path, out_mode / "oof_pred_selected.csv")
    shutil.copyfile(selected_submission_f1max_path, out_mode / "submission_selected_f1max.csv")

    overview = {
        "mode": mode,
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "n_train": int(df_tr.shape[0]),
        "n_test": int(df_te.shape[0]),
        "n_features": int(X_train.shape[1]),
        "best_model": best_model,
        "best_cv_auc": best_auc,
        "best_cv_ap": best_ap,
        "best_cv_f1": best_f1,
        "best_cv_f1_reg": best_f1_reg,
        "best_threshold_f1": best_threshold_f1,
        "best_threshold_reg": best_threshold_reg,
        "blend_cv_auc": blend_auc,
        "blend_cv_ap": blend_ap,
        "blend_cv_f1": blend_f1,
        "blend_cv_f1_reg": blend_f1_reg,
        "blend_threshold_f1": float(result["blend_threshold_f1"]),
        "blend_threshold_reg": float(result["blend_threshold_reg"]),
        "selected_variant": selected_variant,
        "selected_cv_f1": selected_cv_f1,
        "selected_cv_f1_reg": selected_cv_f1_reg,
        "selected_cv_auc": selected_cv_auc,
        "selected_cv_ap": selected_cv_ap,
        "selected_threshold_reg": selected_threshold,
        "selected_threshold_f1": selected_threshold_f1max,
        "selected_submission_path": str(out_mode / "submission_selected.csv"),
        "selected_submission_f1max_path": str(out_mode / "submission_selected_f1max.csv"),
        "selected_oof_path": str(out_mode / "oof_pred_selected.csv"),
        "preprocess": result["preproc_overview"],
    }
    (out_mode / "feature_overview.json").write_text(json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")

    return overview


def choose_final_summary(summaries: List[dict]) -> dict:
    if not summaries:
        return {}
    if len(summaries) == 1:
        return summaries[0]

    def nz(v, default=-1.0):
        return default if v is None or (isinstance(v, float) and np.isnan(v)) else v

    # 기본: selected_cv_f1_reg 우선
    ranked = sorted(
        summaries,
        key=lambda x: (nz(x.get("selected_cv_f1_reg")), nz(x.get("selected_cv_auc")), nz(x.get("selected_cv_ap"))),
        reverse=True,
    )
    top = ranked[0]
    second = ranked[1]

    # 안정성 우선 tie-break:
    # core_plus_exp가 아주 근소하게만 높으면(core 대비 +0.005 미만) core 채택
    if top["mode"] == "core_plus_exp" and second["mode"] == "core":
        if (nz(top.get("selected_cv_f1_reg")) - nz(second.get("selected_cv_f1_reg"))) < 0.005:
            return second
    if top["mode"] == "core" and second["mode"] == "core_plus_exp":
        if (nz(second.get("selected_cv_f1_reg")) - nz(top.get("selected_cv_f1_reg"))) < 0.005:
            return top

    return top


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec_path", type=str, default="feature_sets/feature_set_v1_spec.json")
    parser.add_argument("--data_dir", type=str, default="feature_sets/data")
    parser.add_argument("--mode", type=str, default="both", choices=["core", "core_plus_exp", "both"])
    parser.add_argument("--out_dir", type=str, default="model_output_v1")
    args = parser.parse_args()

    spec = json.loads(Path(args.spec_path).read_text(encoding="utf-8"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)

    summaries = []
    if args.mode in ("core", "both"):
        summaries.append(
            run_dataset_mode(
                mode="core",
                train_csv=data_dir / "feature_set_v1_train_core.csv",
                test_csv=data_dir / "feature_set_v1_test_core.csv",
                spec=spec,
                out_dir=run_dir,
            )
        )
    if args.mode in ("core_plus_exp", "both"):
        summaries.append(
            run_dataset_mode(
                mode="core_plus_exp",
                train_csv=data_dir / "feature_set_v1_train_core_plus_exp.csv",
                test_csv=data_dir / "feature_set_v1_test_core_plus_exp.csv",
                spec=spec,
                out_dir=run_dir,
            )
        )

    final_choice = choose_final_summary(summaries)

    # 최종 산출물 고정
    if final_choice:
        final_submission = run_dir / "final_submission.csv"
        final_oof = run_dir / "final_oof.csv"
        shutil.copyfile(final_choice["selected_submission_path"], final_submission)
        shutil.copyfile(final_choice["selected_oof_path"], final_oof)
    else:
        final_submission = run_dir / "final_submission.csv"
        final_oof = run_dir / "final_oof.csv"

    run_summary = {
        "run_dir": str(run_dir),
        "modes_run": [s["mode"] for s in summaries],
        "final_mode": final_choice.get("mode", ""),
        "final_variant": final_choice.get("selected_variant", ""),
        "final_cv_f1": final_choice.get("selected_cv_f1", np.nan),
        "final_cv_f1_reg": final_choice.get("selected_cv_f1_reg", np.nan),
        "final_cv_auc": final_choice.get("selected_cv_auc", np.nan),
        "final_cv_ap": final_choice.get("selected_cv_ap", np.nan),
        "final_threshold_reg": final_choice.get("selected_threshold_reg", np.nan),
        "final_threshold_f1": final_choice.get("selected_threshold_f1", np.nan),
        "final_submission": str(final_submission),
        "final_oof": str(final_oof),
        "summaries": summaries,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "TRAIN FEATURE SET V1 DONE",
        f"run_dir={run_dir}",
        f"modes={[s['mode'] for s in summaries]}",
        f"final_mode={run_summary['final_mode']}",
        f"final_variant={run_summary['final_variant']}",
        f"final_cv_f1={run_summary['final_cv_f1']}",
        f"final_cv_f1_reg={run_summary['final_cv_f1_reg']}",
        f"final_cv_auc={run_summary['final_cv_auc']}",
        f"final_threshold_reg={run_summary['final_threshold_reg']}",
        f"final_threshold_f1={run_summary['final_threshold_f1']}",
        f"final_submission={run_summary['final_submission']}",
    ]
    (run_dir / "99_done.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
