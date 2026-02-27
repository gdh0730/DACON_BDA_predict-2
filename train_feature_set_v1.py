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
    - core/model_metrics_base.csv
    - core/model_metrics.csv
    - core/oof_pred_<candidate>.csv
    - core/test_pred_<candidate>.csv
    - core/submission_<candidate>.csv
    - core/submission_<candidate>_f1max.csv
    - core/selected_threshold_sweep.csv
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

DEFAULT_N_SPLITS = 5
DEFAULT_RANDOM_STATE = 42
DEFAULT_REG_LAMBDA = 0.35


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


def derive_target_pos_rate(train_pos_rate: float) -> float:
    # train prevalence 대비 약간 완화된 비율을 기본 타겟으로 사용
    return float(np.clip(train_pos_rate + 0.10, 0.35, 0.60))


def unit_rank(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if n == 0:
        return x
    order = np.argsort(x, kind="mergesort")
    out = np.empty(n, dtype=float)
    out[order] = np.arange(1, n + 1, dtype=float)
    return out / float(n)


def safe_slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(text)).strip("_").lower()
    return s or "candidate"


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
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_pos_rate = float(np.mean(y))
    target_pos_rate = derive_target_pos_rate(train_pos_rate)

    fold_rows = []
    base_model_rows = []
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
        tinfo = optimize_threshold(
            y.to_numpy(),
            oof,
            target_pos_rate=target_pos_rate,
            lambda_rate=DEFAULT_REG_LAMBDA,
        )
        base_model_rows.append({
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

    base_model_df = pd.DataFrame(base_model_rows).sort_values(
        ["cv_f1_reg", "cv_auc", "cv_ap"],
        ascending=False
    ).reset_index(drop=True)
    fold_df = pd.DataFrame(fold_rows).sort_values(["model", "fold"]).reset_index(drop=True)

    best_model = base_model_df.iloc[0]["model"]
    base_names = list(per_model_oof.keys())
    model_to_ap = {r["model"]: float(r["cv_ap"]) for r in base_model_rows}

    candidate_rows = []
    candidate_pred_map = {}

    def register_candidate(name: str, oof_pred: np.ndarray, test_pred: np.ndarray, source: str):
        auc = roc_auc_score(y, oof_pred)
        ap = average_precision_score(y, oof_pred)
        tinfo = optimize_threshold(
            y.to_numpy(),
            oof_pred,
            target_pos_rate=target_pos_rate,
            lambda_rate=DEFAULT_REG_LAMBDA,
        )
        threshold_f1 = float(tinfo["threshold_f1"])
        threshold_reg = float(tinfo["threshold_reg"])
        test_pos_rate_f1 = float(np.mean((test_pred >= threshold_f1).astype(int)))
        test_pos_rate_reg = float(np.mean((test_pred >= threshold_reg).astype(int)))

        row = {
            "candidate": name,
            "source": source,
            "cv_auc": float(auc),
            "cv_ap": float(ap),
            "cv_f1": float(tinfo["cv_f1"]),
            "threshold_f1": threshold_f1,
            "cv_f1_reg": float(tinfo["cv_f1_reg"]),
            "threshold_reg": threshold_reg,
            "cv_objective_reg": float(tinfo["cv_objective_reg"]),
            "pred_pos_rate_reg": float(tinfo["pred_pos_rate_reg"]),
            "target_pos_rate": float(tinfo["target_pos_rate"]),
            "test_pos_rate_reg": test_pos_rate_reg,
            "test_pos_rate_f1": test_pos_rate_f1,
            "n_splits": n_splits,
        }
        candidate_rows.append(row)
        candidate_pred_map[name] = {
            "oof": oof_pred,
            "test": test_pred,
            "metrics": row,
        }

    for mname in base_names:
        register_candidate(
            name=f"model::{mname}",
            oof_pred=per_model_oof[mname],
            test_pred=per_model_test[mname],
            source="single_model",
        )

    register_candidate(
        name="best_single",
        oof_pred=per_model_oof[best_model],
        test_pred=per_model_test[best_model],
        source=f"model::{best_model}",
    )

    # blend 1: 단순 평균
    oof_stack = np.column_stack([per_model_oof[m] for m in base_names])
    test_stack = np.column_stack([per_model_test[m] for m in base_names])
    register_candidate(
        name="blend_mean",
        oof_pred=np.mean(oof_stack, axis=1),
        test_pred=np.mean(test_stack, axis=1),
        source="blend_mean",
    )

    # blend 2: AP 가중 평균
    ap_weights = np.array([max(model_to_ap.get(m, 0.0), 1e-6) for m in base_names], dtype=float)
    ap_weights = ap_weights / ap_weights.sum()
    register_candidate(
        name="blend_ap_weighted",
        oof_pred=np.average(oof_stack, axis=1, weights=ap_weights),
        test_pred=np.average(test_stack, axis=1, weights=ap_weights),
        source="blend_ap_weighted",
    )

    # blend 3: rank 평균(분포 차이에 견고)
    oof_rank = np.column_stack([unit_rank(per_model_oof[m]) for m in base_names])
    test_rank = np.column_stack([unit_rank(per_model_test[m]) for m in base_names])
    register_candidate(
        name="blend_rank_mean",
        oof_pred=np.mean(oof_rank, axis=1),
        test_pred=np.mean(test_rank, axis=1),
        source="blend_rank_mean",
    )

    # blend 4: OOF 기반 2층 로지스틱 스태킹
    meta_oof = np.zeros(len(X_train), dtype=float)
    meta_test_fold_preds = []
    meta_skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state + 77)
    y_np = y.to_numpy()

    for fold, (tr_idx, va_idx) in enumerate(meta_skf.split(oof_stack, y_np), start=1):
        meta = LogisticRegression(
            solver="liblinear",
            penalty="l2",
            C=1.0,
            max_iter=3000,
            class_weight=None,
            random_state=random_state + fold,
        )
        meta.fit(oof_stack[tr_idx], y_np[tr_idx])
        meta_oof[va_idx] = meta.predict_proba(oof_stack[va_idx])[:, 1]
        meta_test_fold_preds.append(meta.predict_proba(test_stack)[:, 1])

    meta_test = np.mean(np.vstack(meta_test_fold_preds), axis=0)
    register_candidate(
        name="stack_lr",
        oof_pred=meta_oof,
        test_pred=meta_test,
        source="meta_logreg",
    )

    candidate_df = pd.DataFrame(candidate_rows)
    candidate_df["selection_score"] = (
        candidate_df["cv_f1_reg"]
        - np.clip(candidate_df["test_pos_rate_reg"] - 0.80, 0.0, None) * 0.10
        - np.clip(0.20 - candidate_df["test_pos_rate_reg"], 0.0, None) * 0.08
    )
    candidate_df = candidate_df.sort_values(
        ["selection_score", "cv_f1_reg", "cv_auc", "cv_ap"],
        ascending=False
    ).reset_index(drop=True)

    selected_candidate = candidate_df.iloc[0]["candidate"]

    return {
        "fold_df": fold_df,
        "base_model_df": base_model_df,
        "candidate_df": candidate_df,
        "selected_candidate": selected_candidate,
        "best_model": best_model,
        "candidate_pred_map": candidate_pred_map,
        "target_pos_rate_train": train_pos_rate,
        "target_pos_rate_reg": target_pos_rate,
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

    result = fit_cv_predict(
        X_train,
        y,
        X_test,
        spec,
        n_splits=DEFAULT_N_SPLITS,
        random_state=DEFAULT_RANDOM_STATE,
    )

    out_mode = out_dir / mode
    out_mode.mkdir(parents=True, exist_ok=True)

    # metrics
    result["fold_df"].to_csv(out_mode / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    base_model_df = result["base_model_df"].copy()
    candidate_df = result["candidate_df"].copy()
    candidate_pred_map = result["candidate_pred_map"]
    base_model_df.to_csv(out_mode / "model_metrics_base.csv", index=False, encoding="utf-8-sig")
    candidate_df.rename(columns={"candidate": "model"}).to_csv(
        out_mode / "model_metrics.csv", index=False, encoding="utf-8-sig"
    )

    # candidate별 확률/제출 파일 전부 생성
    artifact_by_candidate = {}
    for _, row in candidate_df.iterrows():
        cand = str(row["candidate"])
        slug = safe_slug(cand)
        oof_pred = np.asarray(candidate_pred_map[cand]["oof"], dtype=float)
        test_pred = np.asarray(candidate_pred_map[cand]["test"], dtype=float)
        threshold_reg = float(row["threshold_reg"])
        threshold_f1 = float(row["threshold_f1"])

        oof_path = out_mode / f"oof_pred_{slug}.csv"
        test_path = out_mode / f"test_pred_{slug}.csv"
        sub_reg_path = out_mode / f"submission_{slug}.csv"
        sub_f1_path = out_mode / f"submission_{slug}_f1max.csv"

        pd.DataFrame({
            id_col: df_tr[id_col],
            target_col: y,
            "pred_proba": oof_pred,
            "model": cand,
        }).to_csv(oof_path, index=False, encoding="utf-8-sig")

        pd.DataFrame({
            id_col: df_te[id_col],
            "pred_proba": test_pred,
            "model": cand,
        }).to_csv(test_path, index=False, encoding="utf-8-sig")

        pd.DataFrame({
            id_col: df_te[id_col],
            target_col: (test_pred >= threshold_reg).astype(int),
        }).to_csv(sub_reg_path, index=False, encoding="utf-8-sig")

        pd.DataFrame({
            id_col: df_te[id_col],
            target_col: (test_pred >= threshold_f1).astype(int),
        }).to_csv(sub_f1_path, index=False, encoding="utf-8-sig")

        artifact_by_candidate[cand] = {
            "oof": oof_path,
            "test": test_path,
            "submission_reg": sub_reg_path,
            "submission_f1": sub_f1_path,
        }

    def row_dict(candidate_name: str) -> dict:
        m = candidate_df[candidate_df["candidate"] == candidate_name]
        return {} if m.empty else m.iloc[0].to_dict()

    best_model = result["best_model"]
    best_row = row_dict("best_single")
    blend_row = row_dict("blend_mean")
    selected_variant = str(result["selected_candidate"])
    selected_row = row_dict(selected_variant)

    if "best_single" in artifact_by_candidate:
        shutil.copyfile(artifact_by_candidate["best_single"]["oof"], out_mode / "oof_pred_best.csv")
        shutil.copyfile(artifact_by_candidate["best_single"]["test"], out_mode / "test_pred_best.csv")
        shutil.copyfile(artifact_by_candidate["best_single"]["submission_reg"], out_mode / "submission_best.csv")
        shutil.copyfile(artifact_by_candidate["best_single"]["submission_f1"], out_mode / "submission_best_f1max.csv")

    if "blend_mean" in artifact_by_candidate:
        shutil.copyfile(artifact_by_candidate["blend_mean"]["oof"], out_mode / "oof_pred_blend.csv")
        shutil.copyfile(artifact_by_candidate["blend_mean"]["test"], out_mode / "test_pred_blend.csv")
        shutil.copyfile(artifact_by_candidate["blend_mean"]["submission_reg"], out_mode / "submission_blend.csv")
        shutil.copyfile(artifact_by_candidate["blend_mean"]["submission_f1"], out_mode / "submission_blend_f1max.csv")

    # 모드별 selected 파일 고정
    selected_submission_path = artifact_by_candidate[selected_variant]["submission_reg"]
    selected_submission_f1max_path = artifact_by_candidate[selected_variant]["submission_f1"]
    selected_oof_path = artifact_by_candidate[selected_variant]["oof"]
    shutil.copyfile(selected_submission_path, out_mode / "submission_selected.csv")
    shutil.copyfile(selected_oof_path, out_mode / "oof_pred_selected.csv")
    shutil.copyfile(selected_submission_f1max_path, out_mode / "submission_selected_f1max.csv")

    # selected 후보에 대해 임계값 sweep 제출 파일 생성
    selected_oof_pred = np.asarray(candidate_pred_map[selected_variant]["oof"], dtype=float)
    selected_test_pred = np.asarray(candidate_pred_map[selected_variant]["test"], dtype=float)
    sweep_rows = []
    rate_grid = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    y_np = y.to_numpy()
    for target_rate in rate_grid:
        threshold = float(np.quantile(selected_oof_pred, 1.0 - target_rate))
        oof_bin = (selected_oof_pred >= threshold).astype(int)
        test_bin = (selected_test_pred >= threshold).astype(int)
        oof_f1 = float(f1_score(y_np, oof_bin, zero_division=0))
        oof_pos_rate = float(np.mean(oof_bin))
        test_pos_rate = float(np.mean(test_bin))
        sub_path = out_mode / f"submission_selected_pr{int(round(target_rate * 100)):02d}.csv"
        pd.DataFrame({
            id_col: df_te[id_col],
            target_col: test_bin,
        }).to_csv(sub_path, index=False, encoding="utf-8-sig")
        sweep_rows.append({
            "target_pos_rate": target_rate,
            "threshold": threshold,
            "oof_f1": oof_f1,
            "oof_pos_rate": oof_pos_rate,
            "test_pos_rate": test_pos_rate,
            "submission_path": str(sub_path),
        })
    threshold_sweep_path = out_mode / "selected_threshold_sweep.csv"
    pd.DataFrame(sweep_rows).sort_values("target_pos_rate").to_csv(
        threshold_sweep_path, index=False, encoding="utf-8-sig"
    )

    overview = {
        "mode": mode,
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "n_train": int(df_tr.shape[0]),
        "n_test": int(df_te.shape[0]),
        "n_features": int(X_train.shape[1]),
        "best_model": best_model,
        "best_cv_auc": float(best_row.get("cv_auc", np.nan)),
        "best_cv_ap": float(best_row.get("cv_ap", np.nan)),
        "best_cv_f1": float(best_row.get("cv_f1", np.nan)),
        "best_cv_f1_reg": float(best_row.get("cv_f1_reg", np.nan)),
        "best_threshold_f1": float(best_row.get("threshold_f1", np.nan)),
        "best_threshold_reg": float(best_row.get("threshold_reg", np.nan)),
        "blend_cv_auc": float(blend_row.get("cv_auc", np.nan)),
        "blend_cv_ap": float(blend_row.get("cv_ap", np.nan)),
        "blend_cv_f1": float(blend_row.get("cv_f1", np.nan)),
        "blend_cv_f1_reg": float(blend_row.get("cv_f1_reg", np.nan)),
        "blend_threshold_f1": float(blend_row.get("threshold_f1", np.nan)),
        "blend_threshold_reg": float(blend_row.get("threshold_reg", np.nan)),
        "selected_variant": selected_variant,
        "selected_cv_f1": float(selected_row.get("cv_f1", np.nan)),
        "selected_cv_f1_reg": float(selected_row.get("cv_f1_reg", np.nan)),
        "selected_cv_auc": float(selected_row.get("cv_auc", np.nan)),
        "selected_cv_ap": float(selected_row.get("cv_ap", np.nan)),
        "selected_threshold_reg": float(selected_row.get("threshold_reg", np.nan)),
        "selected_threshold_f1": float(selected_row.get("threshold_f1", np.nan)),
        "selected_test_pos_rate_reg": float(selected_row.get("test_pos_rate_reg", np.nan)),
        "selected_test_pos_rate_f1": float(selected_row.get("test_pos_rate_f1", np.nan)),
        "target_pos_rate_train": float(result.get("target_pos_rate_train", np.nan)),
        "target_pos_rate_reg": float(result.get("target_pos_rate_reg", np.nan)),
        "selected_submission_path": str(out_mode / "submission_selected.csv"),
        "selected_submission_f1max_path": str(out_mode / "submission_selected_f1max.csv"),
        "selected_oof_path": str(out_mode / "oof_pred_selected.csv"),
        "threshold_sweep_path": str(threshold_sweep_path),
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

    ranked = []
    for s in summaries:
        test_pos_rate = nz(s.get("selected_test_pos_rate_reg"), default=0.5)
        score = (
            nz(s.get("selected_cv_f1_reg"))
            - max(0.0, test_pos_rate - 0.80) * 0.10
            - max(0.0, 0.20 - test_pos_rate) * 0.08
        )
        ranked.append((score, nz(s.get("selected_cv_f1_reg")), nz(s.get("selected_cv_ap")), nz(s.get("selected_cv_auc")), s))
    ranked.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
    return ranked[0][4]


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
        "final_test_pos_rate_reg": final_choice.get("selected_test_pos_rate_reg", np.nan),
        "final_test_pos_rate_f1": final_choice.get("selected_test_pos_rate_f1", np.nan),
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
        f"final_test_pos_rate_reg={run_summary['final_test_pos_rate_reg']}",
        f"final_test_pos_rate_f1={run_summary['final_test_pos_rate_f1']}",
        f"final_submission={run_summary['final_submission']}",
    ]
    (run_dir / "99_done.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
