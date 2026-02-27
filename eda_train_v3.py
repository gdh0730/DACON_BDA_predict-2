# -*- coding: utf-8 -*-
"""
가설 기반 고도화 EDA 스크립트 (v3)
- 입력: open/train.csv, open/test.csv(선택)
- 출력: eda_output_v3/YYYYMMDD_HHMMSS/{reports,figures,artifacts}
- 목적:
  1) v2의 광범위 진단 유지
  2) 'completed(수료여부) 예측' 목적에 맞춘 가설 기반 분석 추가
  3) train/test 분포 이동(드리프트)까지 점검하여 실전 예측 안정성 강화
"""

import argparse
import math
import re
import itertools
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 한글 폰트(환경에 없으면 무시)
try:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

# (선택) 통계
try:
    from scipy.stats import chi2_contingency, spearmanr
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# (선택) 모델 기반 지표
try:
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

TOKEN_PAT = re.compile(r"[A-Za-z0-9가-힣]+")


# =========================
# 유틸
# =========================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


def safe_filename(name: str) -> str:
    name = "" if name is None else str(name)
    name = re.sub(r"[^A-Za-z0-9가-힣._-]+", "_", name).strip("_")
    return name or "unnamed"


def safe_series_to_numeric(s: pd.Series) -> pd.Series:
    """문자열/불리언 혼입 가능성을 고려해 숫자로 강제 변환(실패는 NaN)."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype("int8")
    if str(s.dtype).lower() == "boolean":
        return s.astype("Int8")
    return pd.to_numeric(s, errors="coerce")


def is_binary_target(y: pd.Series) -> bool:
    uniq = set(pd.Series(y).dropna().unique())
    return uniq <= {0, 1}


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if not SKLEARN_OK:
        return np.nan
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    mask = np.isfinite(y_true) & np.isfinite(score)
    if mask.sum() < 30:
        return np.nan
    y2 = y_true[mask]
    s2 = score[mask]
    if len(np.unique(y2)) < 2 or len(np.unique(s2)) < 2:
        return np.nan
    try:
        auc = float(roc_auc_score(y2, s2))
    except Exception:
        return np.nan
    # 방향성 제거(0/1 부호 뒤집힘 영향을 줄이기 위함)
    return max(auc, 1.0 - auc)


def normalize_multiselect_tokens(value) -> list:
    if pd.isna(value):
        return []
    s = str(value).replace("\n", " ").strip()
    if not s or s.lower() == "nan":
        return []
    parts = re.split(r"\s*,\s*", s)
    out = []
    seen = set()
    for p in parts:
        p2 = re.sub(r"\s+", " ", p).strip()
        if not p2 or p2.lower() == "nan":
            continue
        if p2 not in seen:
            seen.add(p2)
            out.append(p2)
    return out


def is_multiselect_like(series: pd.Series, min_unique: int = 8, min_comma_ratio: float = 0.15) -> bool:
    s = series.dropna().astype(str)
    if s.empty:
        return False
    uniq = s.nunique(dropna=True)
    comma_ratio = s.str.contains(",", regex=False).mean()
    return (uniq >= min_unique) and (comma_ratio >= min_comma_ratio)


def valid_string_mask(s: pd.Series) -> pd.Series:
    s_obj = s.astype("object")
    return s_obj.notna() & s_obj.astype(str).str.strip().ne("") & s_obj.astype(str).str.lower().ne("nan")


def yes_no_flag(s: pd.Series, yes_tokens: List[str] = None) -> pd.Series:
    yes_tokens = yes_tokens or ["예", "yes", "true", "1"]
    t = s.astype("object").fillna("").astype(str).str.lower()
    return t.apply(lambda x: int(any(tok in x for tok in yes_tokens)))


def parse_bool_like(s: pd.Series) -> pd.Series:
    t = s.astype("object").fillna("").astype(str).str.strip().str.lower()
    mapping = {
        "true": 1, "false": 0,
        "1": 1, "0": 0,
        "yes": 1, "no": 0,
        "예": 1, "아니요": 0,
    }
    out = t.map(mapping)
    # 매핑 실패는 숫자 변환 시도
    if out.isna().any():
        num = pd.to_numeric(t, errors="coerce")
        out = out.fillna(num)
    return out.fillna(0).astype(float)


def token_count(value) -> int:
    return len(normalize_multiselect_tokens(value))


def contains_keyword_count(value, keywords: List[str]) -> int:
    tokens = normalize_multiselect_tokens(value)
    cnt = 0
    for tok in tokens:
        low = tok.lower()
        if any(k.lower() in low for k in keywords):
            cnt += 1
    return cnt


def quantile_bucket(series: pd.Series, q: int = 5, labels: List[str] = None) -> pd.Series:
    x = safe_series_to_numeric(series)
    if x.notna().sum() < max(20, q * 4):
        return pd.Series(["missing_or_sparse"] * len(series), index=series.index)
    try:
        b = pd.qcut(x, q=q, duplicates="drop")
    except Exception:
        return pd.Series(["missing_or_sparse"] * len(series), index=series.index)
    if labels is None:
        return b.astype(str).fillna("missing")
    # qcut bins 수가 q보다 줄어들 수 있으므로 길이 보정
    bins_n = b.cat.categories.size if hasattr(b, "cat") else len(np.unique(b.dropna()))
    used_labels = labels[:bins_n]
    return b.cat.rename_categories(used_labels).astype("object").fillna("missing")


def psi_numeric(train_s: pd.Series, test_s: pd.Series, bins: int = 10) -> float:
    tr = safe_series_to_numeric(train_s)
    te = safe_series_to_numeric(test_s)
    tr = tr.dropna()
    te = te.dropna()
    if tr.empty or te.empty:
        return np.nan

    # train 분위수 기준 bin
    q = min(bins, max(2, int(tr.nunique())))
    try:
        _, edges = pd.qcut(tr, q=q, retbins=True, duplicates="drop")
    except Exception:
        return np.nan
    if len(edges) < 3:
        return np.nan

    tr_bin = pd.cut(tr, bins=edges, include_lowest=True)
    te_bin = pd.cut(te, bins=edges, include_lowest=True)

    tr_dist = tr_bin.value_counts(normalize=True, dropna=False)
    te_dist = te_bin.value_counts(normalize=True, dropna=False)
    idx = tr_dist.index.union(te_dist.index)
    p = tr_dist.reindex(idx, fill_value=0.0).values
    qd = te_dist.reindex(idx, fill_value=0.0).values

    eps = 1e-6
    p = np.clip(p, eps, 1.0)
    qd = np.clip(qd, eps, 1.0)
    return float(np.sum((p - qd) * np.log(p / qd)))


def js_divergence_categorical(train_s: pd.Series, test_s: pd.Series, topk: int = 100) -> float:
    tr = train_s.astype("object").fillna("<<MISSING>>").astype(str)
    te = test_s.astype("object").fillna("<<MISSING>>").astype(str)

    tr_vc = tr.value_counts(normalize=True)
    te_vc = te.value_counts(normalize=True)

    if len(tr_vc) > topk:
        keep = set(tr_vc.head(topk).index.tolist())
        tr = tr.where(tr.isin(keep), "<<OTHER>>")
        te = te.where(te.isin(keep), "<<OTHER>>")
        tr_vc = tr.value_counts(normalize=True)
        te_vc = te.value_counts(normalize=True)

    idx = tr_vc.index.union(te_vc.index)
    p = tr_vc.reindex(idx, fill_value=0.0).values
    qd = te_vc.reindex(idx, fill_value=0.0).values
    m = 0.5 * (p + qd)

    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    qd = np.clip(qd, eps, 1.0)
    m = np.clip(m, eps, 1.0)

    js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(qd * np.log(qd / m))
    return float(js)


def shift_level(metric_value: float, metric_type: str) -> str:
    if not np.isfinite(metric_value):
        return "unknown"
    if metric_type == "psi":
        if metric_value >= 0.25:
            return "high"
        if metric_value >= 0.10:
            return "medium"
        return "low"
    # js
    if metric_value >= 0.10:
        return "high"
    if metric_value >= 0.03:
        return "medium"
    return "low"


def classify_column(series: pd.Series, name: str, target_name: str):
    """
    컬럼 타입 분류:
      - id_like
      - numeric_continuous
      - numeric_discrete
      - categorical_low
      - categorical_high
      - text
      - datetime_like
      - constant
      - target
    """
    if name == target_name:
        return "target"

    nun = series.nunique(dropna=True)
    if nun <= 1:
        return "constant"

    if series.dtype == "object":
        sample = series.dropna().astype(str).head(200)
        if len(sample) > 0:
            date_hits = sample.str.contains(r"(?:\d{4}[-/]\d{1,2}[-/]\d{1,2})|(?:\d{8})", regex=True).mean()
            if date_hits >= 0.6:
                return "datetime_like"

    n = len(series)
    uniq_ratio = nun / max(n, 1)
    if uniq_ratio >= 0.98:
        return "id_like"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric_discrete" if nun <= 30 else "numeric_continuous"

    if series.dtype == "object":
        sn = safe_series_to_numeric(series)
        conv_ratio = sn.notna().mean()
        if conv_ratio >= 0.95:
            nun_num = sn.nunique(dropna=True)
            return "numeric_discrete" if nun_num <= 30 else "numeric_continuous"

    if series.dtype == "object":
        s_str = series.dropna().astype(str)
        avg_len = s_str.map(len).mean() if len(s_str) > 0 else 0
        if nun <= 20 and avg_len <= 40:
            return "categorical_low"
        if nun <= 200 and avg_len <= 80:
            return "categorical_high"
        return "text"

    return "unknown"


def describe_numeric(series: pd.Series):
    s = safe_series_to_numeric(series) if series.dtype == "object" else series
    if pd.api.types.is_bool_dtype(s) or str(s.dtype).lower() == "boolean":
        s = s.astype(float)
    s = pd.Series(s).dropna().astype(float)
    if len(s) == 0:
        return {}
    return {
        "count": int(s.shape[0]),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
        "min": float(s.min()),
        "p1": float(np.percentile(s, 1)),
        "p5": float(np.percentile(s, 5)),
        "p25": float(np.percentile(s, 25)),
        "p50": float(np.percentile(s, 50)),
        "p75": float(np.percentile(s, 75)),
        "p95": float(np.percentile(s, 95)),
        "p99": float(np.percentile(s, 99)),
        "max": float(s.max()),
        "nunique": int(s.nunique()),
    }


def iqr_outlier_mask(x: pd.Series, k: float = 1.5):
    x = safe_series_to_numeric(x) if x.dtype == "object" else x
    if pd.api.types.is_bool_dtype(x) or str(x.dtype).lower() == "boolean":
        x = x.astype(float)
    x = pd.Series(x).dropna().astype(float)
    if len(x) < 8:
        return None, None, None
    q1, q3 = np.percentile(x, 25), np.percentile(x, 75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return lo, hi, (x < lo) | (x > hi)


def cramers_v_from_crosstab(ct: pd.DataFrame) -> float:
    if not SCIPY_OK:
        return np.nan
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan
    try:
        chi2 = chi2_contingency(ct.values, correction=False)[0]
    except Exception:
        return np.nan
    n = ct.values.sum()
    if n <= 1:
        return np.nan
    phi2 = chi2 / n
    r, k = ct.shape
    # bias-corrected Cramer's V
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return np.nan
    return float(np.sqrt(phi2corr / denom))


# =========================
# 시각화
# =========================
def plot_missing_bar(missing_rate: pd.Series, outpath: Path, topk: int = 30):
    mr = missing_rate.sort_values(ascending=False).head(topk)
    plt.figure(figsize=(10, 6))
    plt.barh(mr.index[::-1], mr.values[::-1])
    plt.xlabel("Missing Rate")
    plt.title(f"Top {topk} Missing Rate Columns")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_target_distribution(y: pd.Series, outpath: Path):
    vc = y.value_counts(dropna=False)
    plt.figure(figsize=(5, 4))
    plt.bar(vc.index.astype(str), vc.values)
    plt.title("Target Distribution")
    plt.xlabel("target")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_numeric_hist(series: pd.Series, name: str, outpath: Path):
    x = safe_series_to_numeric(series) if series.dtype == "object" else series
    x = pd.Series(x).dropna().astype(float)
    if len(x) == 0:
        return

    p1, p99 = np.percentile(x, 1), np.percentile(x, 99)

    plt.figure(figsize=(7, 4))
    plt.hist(x, bins=30)
    plt.title(f"Histogram: {name}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    x_clip = x[(x >= p1) & (x <= p99)]
    plt.hist(x_clip, bins=30)
    plt.title(f"Histogram (1-99% clipped): {name}")
    plt.tight_layout()
    plt.savefig(outpath.with_name(outpath.stem + "_clipped.png"), dpi=160)
    plt.close()


def plot_numeric_by_target(df: pd.DataFrame, col: str, target: str, outpath: Path):
    x = safe_series_to_numeric(df[col]) if df[col].dtype == "object" else df[col]
    x = pd.Series(x)
    y = df[target]

    data0 = x[y == 0].dropna().astype(float)
    data1 = x[y == 1].dropna().astype(float)

    if len(data0) + len(data1) == 0:
        return
    plt.figure(figsize=(6, 4))
    plt.boxplot([data0.values, data1.values], tick_labels=["0", "1"], showfliers=False)
    plt.title(f"{col} by {target} (box, no fliers)")
    plt.xlabel(target)
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def _wrap_label(s: str, width: int = 20) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\n", " ").strip()
    if len(s) <= width:
        return s
    chunks = []
    while len(s) > width:
        chunks.append(s[:width])
        s = s[width:]
    if s:
        chunks.append(s)
    return "\n".join(chunks)


def plot_categorical_target_rate(df: pd.DataFrame, col: str, target: str, outpath: Path, topk: int = 15):
    s = df[col].astype("object")
    vc = s.value_counts(dropna=False)
    cats = vc.head(topk).index
    tmp = df[df[col].isin(cats)].copy()
    if tmp.empty:
        return

    rate = tmp.groupby(col, dropna=False)[target].mean().reindex(cats)
    cnt = tmp.groupby(col, dropna=False)[target].size().reindex(cats)

    labels = [_wrap_label(str(c), width=18) for c in rate.index]
    y_pos = np.arange(len(labels))
    fig_h = max(4, 0.9 * len(labels) + 1)

    plt.figure(figsize=(10, fig_h))
    plt.barh(y_pos, rate.values)
    plt.yticks(y_pos, labels)
    plt.xlim(0, 1)
    plt.title(f"{col}: Target Rate (top {topk})")
    plt.xlabel("P(target=1)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, fig_h))
    plt.barh(y_pos, cnt.values)
    plt.yticks(y_pos, labels)
    plt.title(f"{col}: Category Counts (top {topk})")
    plt.xlabel("count")
    plt.tight_layout()
    plt.savefig(outpath.with_name(outpath.stem + "_counts.png"), dpi=160, bbox_inches="tight")
    plt.close()


def correlation_heatmap(df_num: pd.DataFrame, outpath: Path):
    if df_num.shape[1] < 2:
        return
    corr = df_num.corr(numeric_only=True)
    plt.figure(figsize=(min(14, 0.5 * corr.shape[1] + 4), min(14, 0.5 * corr.shape[0] + 4)))
    plt.imshow(corr.values, aspect="auto")
    plt.colorbar()
    plt.xticks(range(corr.shape[1]), corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(corr.shape[0]), corr.index, fontsize=7)
    plt.title("Correlation Heatmap (numeric)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_token_risk_diff(token_df: pd.DataFrame, title: str, outpath: Path, topk: int = 10):
    if token_df.empty:
        return

    pos = token_df.sort_values("risk_diff", ascending=False).head(topk)
    neg = token_df.sort_values("risk_diff", ascending=True).head(topk)
    show = pd.concat([neg, pos], axis=0)
    if show.empty:
        return

    labels = [_wrap_label(x, width=24) for x in show["token"].astype(str)]
    y = np.arange(len(show))
    colors = ["#d95f02" if v > 0 else "#1b9e77" for v in show["risk_diff"].values]

    plt.figure(figsize=(10, max(5, 0.45 * len(show) + 1)))
    plt.barh(y, show["risk_diff"].values, color=colors)
    plt.yticks(y, labels)
    plt.axvline(0, color="gray", linewidth=1)
    plt.title(title)
    plt.xlabel("risk_diff = P(y=1|token) - P(y=1|no token)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=170, bbox_inches="tight")
    plt.close()


# =========================
# 고도화 분석 함수
# =========================
def leakage_suspects_v1_style(df: pd.DataFrame, target: str, col_types: dict) -> pd.DataFrame:
    suspects = []

    keywords = [
        "complete", "completed", "pass", "result", "outcome", "label",
        "score", "grade", "eval", "evaluation", "final", "certificate",
        "수료", "합격", "결과", "점수", "평가", "성적", "완료"
    ]
    for c in df.columns:
        if c == target:
            continue
        lower = c.lower()
        if any(k in lower for k in keywords):
            suspects.append((c, "name_keyword"))

    y = pd.to_numeric(df[target], errors="coerce")
    for c, t in col_types.items():
        if c == target:
            continue
        if t in ("numeric_continuous", "numeric_discrete"):
            x = safe_series_to_numeric(df[c])
            ok = x.notna() & y.notna()
            if ok.sum() >= 50:
                try:
                    corr = np.corrcoef(x[ok].astype(float), y[ok].astype(float))[0, 1]
                except Exception:
                    corr = np.nan
                if np.isfinite(corr) and abs(corr) >= 0.75:
                    suspects.append((c, f"high_corr(|r|={corr:.3f})"))

    for c in df.columns:
        if c == target:
            continue
        miss = df[c].isna().astype(int)
        ok = y.notna()
        if ok.sum() >= 50:
            try:
                corr = np.corrcoef(miss[ok], y[ok].astype(int))[0, 1]
            except Exception:
                corr = np.nan
            if np.isfinite(corr) and abs(corr) >= 0.75:
                suspects.append((c, f"missing_pattern_corr(|r|={corr:.3f})"))

    out = pd.DataFrame(suspects, columns=["column", "reason"]).drop_duplicates()
    if out.empty:
        return pd.DataFrame(columns=["column", "reason"])
    return out.sort_values(["reason", "column"])  # noqa: C408


def categorical_association_summary(df: pd.DataFrame, cat_cols: list, target: str) -> pd.DataFrame:
    rows = []
    for c in cat_cols:
        s = df[c].astype("object").fillna("<<MISSING>>")
        ct = pd.crosstab(s, df[target])
        chi2_p = np.nan
        chi2_stat = np.nan
        cramers_v = np.nan
        dof = np.nan

        if SCIPY_OK and ct.shape[0] >= 2 and ct.shape[1] == 2:
            try:
                chi2_stat, chi2_p, dof, _ = chi2_contingency(ct.values)
            except Exception:
                pass
        cramers_v = cramers_v_from_crosstab(ct)

        rows.append({
            "column": c,
            "nunique": int(df[c].nunique(dropna=True)),
            "missing_rate": float(df[c].isna().mean()),
            "chi2_stat": chi2_stat,
            "chi2_pvalue": chi2_p,
            "chi2_dof": dof,
            "cramers_v": cramers_v,
        })
    if not rows:
        return pd.DataFrame(columns=["column", "nunique", "missing_rate", "chi2_stat", "chi2_pvalue", "chi2_dof", "cramers_v"])
    out = pd.DataFrame(rows)
    return out.sort_values(["cramers_v", "chi2_stat"], ascending=[False, False])


def analyze_multiselect_tokens(df: pd.DataFrame, columns: list, target: str,
                               artifacts_dir: Path, figures_dir: Path,
                               min_count: int = 8) -> pd.DataFrame:
    if not is_binary_target(df[target]):
        return pd.DataFrame(columns=["column", "token", "token_count", "token_rate", "non_token_rate", "risk_diff", "lift_vs_base"])

    y = df[target].astype(int).to_numpy()
    base_rate = float(np.mean(y))
    all_rows = []

    for c in columns:
        token_lists = df[c].apply(normalize_multiselect_tokens)
        token_to_idx = defaultdict(list)

        for i, toks in enumerate(token_lists):
            for tok in toks:
                token_to_idx[tok].append(i)

        rows = []
        n = len(df)
        for tok, idx_list in token_to_idx.items():
            cnt = len(idx_list)
            if cnt < min_count:
                continue
            present = np.zeros(n, dtype=bool)
            present[idx_list] = True
            y_present = y[present]
            y_absent = y[~present]
            if len(y_present) < min_count or len(y_absent) < min_count:
                continue

            token_rate = float(np.mean(y_present))
            non_token_rate = float(np.mean(y_absent))
            risk_diff = token_rate - non_token_rate
            lift = token_rate / base_rate if base_rate > 0 else np.nan

            rows.append({
                "column": c,
                "token": tok,
                "token_count": cnt,
                "token_rate": token_rate,
                "non_token_rate": non_token_rate,
                "risk_diff": risk_diff,
                "lift_vs_base": lift,
            })

        if not rows:
            continue

        col_df = pd.DataFrame(rows).sort_values(["risk_diff", "token_count"], ascending=[False, False])
        col_df.to_csv(artifacts_dir / f"multiselect_token_signal__{safe_filename(c)}.csv", index=False, encoding="utf-8-sig")

        plot_token_risk_diff(
            col_df,
            title=f"{c}: token risk diff (min_count={min_count})",
            outpath=figures_dir / f"multiselect_token_riskdiff__{safe_filename(c)}.png",
            topk=10,
        )

        all_rows.extend(rows)

    if not all_rows:
        return pd.DataFrame(columns=["column", "token", "token_count", "token_rate", "non_token_rate", "risk_diff", "lift_vs_base"])

    out = pd.DataFrame(all_rows)
    out["abs_risk_diff"] = out["risk_diff"].abs()
    out = out.sort_values(["abs_risk_diff", "token_count"], ascending=[False, False])
    return out.drop(columns=["abs_risk_diff"])


def numeric_binned_trend(df: pd.DataFrame, numeric_cols: list, target: str, artifacts_dir: Path, max_bins: int = 10):
    rows = []
    per_bin_frames = []

    y = pd.to_numeric(df[target], errors="coerce")

    for c in numeric_cols:
        x = safe_series_to_numeric(df[c])
        ok = x.notna() & y.notna()
        if ok.sum() < 80:
            continue

        x2 = x[ok].astype(float)
        y2 = y[ok].astype(int)

        n_unique = int(x2.nunique(dropna=True))
        if n_unique < 4:
            continue

        q = min(max_bins, n_unique)
        try:
            bins = pd.qcut(x2, q=q, duplicates="drop")
        except Exception:
            continue

        tmp = pd.DataFrame({"x": x2, "y": y2, "bin": bins.astype(str)})
        agg = tmp.groupby("bin", dropna=False).agg(
            bin_count=("y", "size"),
            target_rate=("y", "mean"),
            x_median=("x", "median"),
            x_min=("x", "min"),
            x_max=("x", "max"),
        ).reset_index()

        agg = agg.sort_values("x_median").reset_index(drop=True)
        agg.insert(0, "column", c)
        agg.insert(1, "bin_order", np.arange(len(agg)))
        per_bin_frames.append(agg)

        rho = np.nan
        pval = np.nan
        if SCIPY_OK and len(agg) >= 3:
            try:
                rho, pval = spearmanr(agg["bin_order"].values, agg["target_rate"].values)
            except Exception:
                rho, pval = np.nan, np.nan

        rows.append({
            "column": c,
            "n_valid": int(ok.sum()),
            "n_bins": int(len(agg)),
            "spearman_rho_bin_vs_target_rate": rho,
            "spearman_pvalue": pval,
            "target_rate_min_bin": float(agg["target_rate"].min()),
            "target_rate_max_bin": float(agg["target_rate"].max()),
            "target_rate_range": float(agg["target_rate"].max() - agg["target_rate"].min()),
        })

    trend_df = pd.DataFrame(rows)
    if not trend_df.empty:
        trend_df = trend_df.sort_values(["target_rate_range", "spearman_rho_bin_vs_target_rate"], ascending=[False, False])

    if per_bin_frames:
        per_bin = pd.concat(per_bin_frames, axis=0, ignore_index=True)
        per_bin.to_csv(artifacts_dir / "numeric_binned_trend_detail.csv", index=False, encoding="utf-8-sig")

    return trend_df


def text_token_logodds(df: pd.DataFrame, text_cols: list, target: str, min_df: int = 8) -> pd.DataFrame:
    if not is_binary_target(df[target]):
        return pd.DataFrame(columns=["column", "token", "doc_freq", "doc_freq_y1", "doc_freq_y0", "p_token_given_y1", "p_token_given_y0", "log_odds_y1_vs_y0"])

    y = df[target].astype(int).to_numpy()
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    out_rows = []

    if n1 == 0 or n0 == 0:
        return pd.DataFrame(columns=["column", "token", "doc_freq", "doc_freq_y1", "doc_freq_y0", "p_token_given_y1", "p_token_given_y0", "log_odds_y1_vs_y0"])

    for c in text_cols:
        s = df[c].astype("object").fillna("").astype(str)
        cnt1 = Counter()
        cnt0 = Counter()

        for i, text in enumerate(s):
            toks = set(TOKEN_PAT.findall(text.lower()))
            if not toks:
                continue
            if y[i] == 1:
                cnt1.update(toks)
            else:
                cnt0.update(toks)

        vocab = set(cnt1.keys()) | set(cnt0.keys())
        if not vocab:
            continue

        for tok in vocab:
            c1 = cnt1.get(tok, 0)
            c0 = cnt0.get(tok, 0)
            dfreq = c1 + c0
            if dfreq < min_df:
                continue

            p1 = c1 / n1
            p0 = c0 / n0
            # 단순 Laplace smoothing 로그 오즈
            log_odds = math.log((c1 + 1) / (n1 + 2)) - math.log((c0 + 1) / (n0 + 2))

            out_rows.append({
                "column": c,
                "token": tok,
                "doc_freq": int(dfreq),
                "doc_freq_y1": int(c1),
                "doc_freq_y0": int(c0),
                "p_token_given_y1": float(p1),
                "p_token_given_y0": float(p0),
                "log_odds_y1_vs_y0": float(log_odds),
            })

    if not out_rows:
        return pd.DataFrame(columns=["column", "token", "doc_freq", "doc_freq_y1", "doc_freq_y0", "p_token_given_y1", "p_token_given_y0", "log_odds_y1_vs_y0"])

    out = pd.DataFrame(out_rows)
    out["abs_log_odds"] = out["log_odds_y1_vs_y0"].abs()
    out = out.sort_values(["abs_log_odds", "doc_freq"], ascending=[False, False]).drop(columns=["abs_log_odds"])
    return out


def high_corr_numeric_pairs(df: pd.DataFrame, numeric_cols: list, threshold: float = 0.85) -> pd.DataFrame:
    if len(numeric_cols) < 2:
        return pd.DataFrame(columns=["col_a", "col_b", "corr", "abs_corr"])

    dfn = pd.DataFrame({c: safe_series_to_numeric(df[c]) for c in numeric_cols})
    corr = dfn.corr(numeric_only=True)
    rows = []
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            r = corr.loc[a, b]
            if np.isfinite(r) and abs(r) >= threshold:
                rows.append({"col_a": a, "col_b": b, "corr": float(r), "abs_corr": float(abs(r))})

    if not rows:
        return pd.DataFrame(columns=["col_a", "col_b", "corr", "abs_corr"])

    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False)


def high_assoc_categorical_pairs(df: pd.DataFrame, cat_cols: list, threshold: float = 0.85, max_nunique: int = 20) -> pd.DataFrame:
    if len(cat_cols) < 2:
        return pd.DataFrame(columns=["col_a", "col_b", "cramers_v", "n_a", "n_b"])

    candidates = [c for c in cat_cols if df[c].nunique(dropna=True) <= max_nunique]
    rows = []

    for a, b in itertools.combinations(candidates, 2):
        s1 = df[a].astype("object").fillna("<<MISSING>>")
        s2 = df[b].astype("object").fillna("<<MISSING>>")
        ct = pd.crosstab(s1, s2)
        v = cramers_v_from_crosstab(ct)
        if np.isfinite(v) and v >= threshold:
            rows.append({
                "col_a": a,
                "col_b": b,
                "cramers_v": float(v),
                "n_a": int(df[a].nunique(dropna=True)),
                "n_b": int(df[b].nunique(dropna=True)),
            })

    if not rows:
        return pd.DataFrame(columns=["col_a", "col_b", "cramers_v", "n_a", "n_b"])

    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False)


def missing_pattern_groups(df: pd.DataFrame, target: str) -> pd.DataFrame:
    pattern_to_cols = defaultdict(list)
    for c in df.columns:
        key = df[c].isna().to_numpy(dtype=np.uint8).tobytes()
        pattern_to_cols[key].append(c)

    rows = []
    y = pd.to_numeric(df[target], errors="coerce") if target in df.columns else None

    for cols in pattern_to_cols.values():
        if len(cols) < 2:
            continue
        rep = cols[0]
        miss = df[rep].isna().astype(int)
        miss_rate = float(miss.mean())
        corr = np.nan
        if y is not None:
            ok = y.notna()
            if ok.sum() >= 50:
                try:
                    corr = np.corrcoef(miss[ok], y[ok].astype(float))[0, 1]
                except Exception:
                    corr = np.nan

        rows.append({
            "group_size": len(cols),
            "missing_rate": miss_rate,
            "corr_with_target": corr,
            "columns": " | ".join(cols),
        })

    if not rows:
        return pd.DataFrame(columns=["group_size", "missing_rate", "corr_with_target", "columns"])
    return pd.DataFrame(rows).sort_values(["group_size", "missing_rate"], ascending=[False, False])


def single_feature_auc_cv(df: pd.DataFrame, target: str, col_types: dict,
                          n_splits: int = 5, random_state: int = 42) -> pd.DataFrame:
    if not SKLEARN_OK:
        return pd.DataFrame(columns=["column", "type", "train_auc", "oof_auc", "auc_gap", "is_suspect", "suspect_reason"])

    y_raw = pd.to_numeric(df[target], errors="coerce")
    if not is_binary_target(y_raw):
        return pd.DataFrame(columns=["column", "type", "train_auc", "oof_auc", "auc_gap", "is_suspect", "suspect_reason"])

    valid_idx = y_raw.notna()
    y = y_raw[valid_idx].astype(int).to_numpy()
    work_df = df.loc[valid_idx].reset_index(drop=True)

    if len(np.unique(y)) < 2:
        return pd.DataFrame(columns=["column", "type", "train_auc", "oof_auc", "auc_gap", "is_suspect", "suspect_reason"])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []

    for c in work_df.columns:
        if c == target:
            continue

        t = col_types.get(c, "unknown")

        train_auc = np.nan
        oof_auc = np.nan

        if t in ("numeric_continuous", "numeric_discrete"):
            x = safe_series_to_numeric(work_df[c]).astype(float)
            if x.notna().sum() < 50 or x.nunique(dropna=True) < 2:
                continue

            med_all = float(np.nanmedian(x.to_numpy())) if np.isfinite(np.nanmedian(x.to_numpy())) else 0.0
            pred_train = x.fillna(med_all).to_numpy()
            train_auc = safe_auc(y, pred_train)

            oof = np.full(len(x), np.nan)
            for tr, va in skf.split(np.zeros(len(y)), y):
                med = float(np.nanmedian(x.iloc[tr].to_numpy()))
                if not np.isfinite(med):
                    med = 0.0
                oof[va] = x.iloc[va].fillna(med).to_numpy()
            oof_auc = safe_auc(y, oof)

        else:
            s = work_df[c].astype("object").fillna("<<MISSING>>").astype(str)
            if s.nunique(dropna=True) < 2:
                continue

            full_map = pd.DataFrame({"k": s, "y": y}).groupby("k")["y"].mean()
            base = float(np.mean(y))
            pred_train = s.map(full_map).fillna(base).to_numpy()
            train_auc = safe_auc(y, pred_train)

            oof = np.full(len(s), np.nan)
            for tr, va in skf.split(np.zeros(len(y)), y):
                tr_map = pd.DataFrame({"k": s.iloc[tr].values, "y": y[tr]}).groupby("k")["y"].mean()
                tr_base = float(np.mean(y[tr]))
                oof[va] = s.iloc[va].map(tr_map).fillna(tr_base).to_numpy()
            oof_auc = safe_auc(y, oof)

        gap = train_auc - oof_auc if np.isfinite(train_auc) and np.isfinite(oof_auc) else np.nan

        reasons = []
        suspect = False
        if np.isfinite(oof_auc) and oof_auc >= 0.85:
            suspect = True
            reasons.append("very_high_oof_auc")
        if np.isfinite(train_auc) and np.isfinite(gap) and train_auc >= 0.72 and gap >= 0.12:
            suspect = True
            reasons.append("high_overfit_gap")

        rows.append({
            "column": c,
            "type": t,
            "train_auc": train_auc,
            "oof_auc": oof_auc,
            "auc_gap": gap,
            "is_suspect": int(suspect),
            "suspect_reason": "|".join(reasons),
        })

    if not rows:
        return pd.DataFrame(columns=["column", "type", "train_auc", "oof_auc", "auc_gap", "is_suspect", "suspect_reason"])

    out = pd.DataFrame(rows)
    return out.sort_values(["is_suspect", "oof_auc", "auc_gap"], ascending=[False, False, False])


def hypothesis_catalog() -> pd.DataFrame:
    rows = [
        {
            "hypothesis_id": "H1",
            "title": "사전 학습경험이 높을수록 수료율이 높다",
            "business_meaning": "이전 학습 경험/재등록 이력이 있는 지원자는 완주 가능성이 높다",
            "related_raw_columns": "re_registration|previous_class_3~8|class1~3",
        },
        {
            "hypothesis_id": "H2",
            "title": "데이터 커리어 정합성이 높을수록 수료율이 높다",
            "business_meaning": "전공/희망직무/자격증 목표가 데이터 직무와 맞을수록 동기 일관성이 높다",
            "related_raw_columns": "major_data|desired_job|desired_certificate|certificate_acquisition|major_field",
        },
        {
            "hypothesis_id": "H3",
            "title": "투입 가능 시간과 참여 방식이 적극적일수록 수료율이 높다",
            "business_meaning": "시간 투자 가능성과 오프라인/팀 선호가 학회 활동 지속성에 영향을 준다",
            "related_raw_columns": "time_input|project_type|hope_for_group|incumbents_lecture_type",
        },
        {
            "hypothesis_id": "H4",
            "title": "응답 구체성/성실도가 높을수록 수료율이 높다",
            "business_meaning": "자유서술 길이와 다중선택 응답 폭은 참여 의지/진지함의 프록시가 된다",
            "related_raw_columns": "incumbents_lecture_scale_reason|interested_company|desired_job|expected_domain|onedayclass_topic",
        },
        {
            "hypothesis_id": "H5",
            "title": "모집채널/동기 유형별로 완주 확률이 다르다",
            "business_meaning": "유입 경로와 지원 동기가 세그먼트별 이탈 리스크를 만든다",
            "related_raw_columns": "inflow_route|whyBDA|what_to_gain|re_registration",
        },
    ]
    return pd.DataFrame(rows)


def build_hypothesis_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    prev_cols = [c for c in df.columns if c.startswith("previous_class_")]
    prev_cols = sorted(prev_cols, key=lambda x: int(re.findall(r"\d+", x)[-1]) if re.findall(r"\d+", x) else 0)
    if prev_cols:
        prev_non_missing = pd.concat([valid_string_mask(df[c]).rename(c) for c in prev_cols], axis=1)
        out["h1_prev_class_count"] = prev_non_missing.sum(axis=1).astype(float)
        out["h1_previous_any"] = (out["h1_prev_class_count"] > 0).astype(float)
    else:
        out["h1_prev_class_count"] = 0.0
        out["h1_previous_any"] = 0.0

    if "re_registration" in df.columns:
        out["h1_reregistration_yes"] = yes_no_flag(df["re_registration"], yes_tokens=["예", "yes", "true", "1"]).astype(float)
    else:
        out["h1_reregistration_yes"] = 0.0

    out["h1_experience_score"] = out["h1_prev_class_count"] + out["h1_reregistration_yes"]

    if "major_data" in df.columns:
        out["h2_major_data_flag"] = parse_bool_like(df["major_data"])
    else:
        out["h2_major_data_flag"] = 0.0

    if "desired_job" in df.columns:
        out["h2_desired_job_data_cnt"] = df["desired_job"].apply(
            lambda v: contains_keyword_count(v, ["데이터", "인공지능", "ai", "머신러닝"])
        ).astype(float)
    else:
        out["h2_desired_job_data_cnt"] = 0.0

    def cert_count_clean(v):
        toks = normalize_multiselect_tokens(v)
        return sum(1 for t in toks if not any(x in t.lower() for x in ["없음", "준비중", "nan"]))

    if "desired_certificate" in df.columns:
        out["h2_desired_cert_cnt"] = df["desired_certificate"].apply(cert_count_clean).astype(float)
    else:
        out["h2_desired_cert_cnt"] = 0.0

    if "certificate_acquisition" in df.columns:
        out["h2_acquired_cert_cnt"] = df["certificate_acquisition"].apply(cert_count_clean).astype(float)
    else:
        out["h2_acquired_cert_cnt"] = 0.0

    out["h2_alignment_score"] = (
        out["h2_major_data_flag"]
        + np.minimum(out["h2_desired_job_data_cnt"], 2.0)
        + (out["h2_desired_cert_cnt"] > 0).astype(float)
    )

    if "time_input" in df.columns:
        time_num = safe_series_to_numeric(df["time_input"])
    else:
        time_num = pd.Series(np.nan, index=df.index)
    out["h3_time_input"] = time_num
    out["h3_time_high"] = (time_num >= 4).astype(float)

    if "project_type" in df.columns:
        out["h3_project_team"] = df["project_type"].astype("object").fillna("").astype(str).str.contains("팀").astype(float)
    else:
        out["h3_project_team"] = 0.0

    if "hope_for_group" in df.columns:
        s = df["hope_for_group"].astype("object").fillna("").astype(str)
        out["h3_hope_offline"] = s.str.contains("오프라인").astype(float)
        out["h3_hope_online"] = s.str.contains("온라인").astype(float)
        out["h3_hope_individual"] = s.str.contains("개인").astype(float)
    else:
        out["h3_hope_offline"] = 0.0
        out["h3_hope_online"] = 0.0
        out["h3_hope_individual"] = 0.0

    out["h3_engagement_score"] = out["h3_time_high"] + out["h3_project_team"] + out["h3_hope_offline"]

    reason_len = df["incumbents_lecture_scale_reason"].astype("object").fillna("").astype(str).str.len() if "incumbents_lecture_scale_reason" in df.columns else pd.Series(0, index=df.index)
    company_len = df["interested_company"].astype("object").fillna("").astype(str).str.len() if "interested_company" in df.columns else pd.Series(0, index=df.index)
    out["h4_text_len_reason"] = reason_len.astype(float)
    out["h4_text_len_company"] = company_len.astype(float)
    out["h4_text_len_total"] = (reason_len + company_len).astype(float)

    breadth_cols = [c for c in ["desired_job", "expected_domain", "onedayclass_topic", "desired_job_except_data"] if c in df.columns]
    breadth = pd.Series(0, index=df.index, dtype=float)
    for c in breadth_cols:
        breadth += df[c].apply(token_count).astype(float)
    out["h4_choice_breadth"] = breadth
    out["h4_response_complexity_score"] = np.log1p(out["h4_text_len_total"]) + out["h4_choice_breadth"]

    q70 = out["h4_response_complexity_score"].quantile(0.70)
    out["h4_complexity_high"] = (out["h4_response_complexity_score"] >= q70).astype(float)

    if "what_to_gain" in df.columns:
        out["h5_goal_item_cnt"] = df["what_to_gain"].apply(token_count).astype(float)
    else:
        out["h5_goal_item_cnt"] = 0.0

    if "inflow_route" in df.columns:
        inflow = df["inflow_route"].astype("object").fillna("").astype(str)
        out["h5_inflow_referral"] = inflow.str.contains("지인|기존 학회원").astype(float)
        out["h5_inflow_organic"] = inflow.str.contains("에브리타임|인스타").astype(float)
    else:
        out["h5_inflow_referral"] = 0.0
        out["h5_inflow_organic"] = 0.0

    return out


def hypothesis_feature_meta() -> Dict[str, Dict[str, str]]:
    return {
        "h1_prev_class_count": {"hypothesis_id": "H1", "expected_direction": "up"},
        "h1_previous_any": {"hypothesis_id": "H1", "expected_direction": "up"},
        "h1_reregistration_yes": {"hypothesis_id": "H1", "expected_direction": "up"},
        "h1_experience_score": {"hypothesis_id": "H1", "expected_direction": "up"},
        "h2_major_data_flag": {"hypothesis_id": "H2", "expected_direction": "up"},
        "h2_desired_job_data_cnt": {"hypothesis_id": "H2", "expected_direction": "up"},
        "h2_desired_cert_cnt": {"hypothesis_id": "H2", "expected_direction": "up"},
        "h2_acquired_cert_cnt": {"hypothesis_id": "H2", "expected_direction": "up"},
        "h2_alignment_score": {"hypothesis_id": "H2", "expected_direction": "up"},
        "h3_time_input": {"hypothesis_id": "H3", "expected_direction": "up"},
        "h3_time_high": {"hypothesis_id": "H3", "expected_direction": "up"},
        "h3_project_team": {"hypothesis_id": "H3", "expected_direction": "up"},
        "h3_hope_offline": {"hypothesis_id": "H3", "expected_direction": "up"},
        "h3_hope_online": {"hypothesis_id": "H3", "expected_direction": "mixed"},
        "h3_hope_individual": {"hypothesis_id": "H3", "expected_direction": "down"},
        "h3_engagement_score": {"hypothesis_id": "H3", "expected_direction": "up"},
        "h4_text_len_reason": {"hypothesis_id": "H4", "expected_direction": "mixed"},
        "h4_text_len_company": {"hypothesis_id": "H4", "expected_direction": "mixed"},
        "h4_text_len_total": {"hypothesis_id": "H4", "expected_direction": "mixed"},
        "h4_choice_breadth": {"hypothesis_id": "H4", "expected_direction": "up"},
        "h4_response_complexity_score": {"hypothesis_id": "H4", "expected_direction": "up"},
        "h4_complexity_high": {"hypothesis_id": "H4", "expected_direction": "up"},
        "h5_goal_item_cnt": {"hypothesis_id": "H5", "expected_direction": "up"},
        "h5_inflow_referral": {"hypothesis_id": "H5", "expected_direction": "mixed"},
        "h5_inflow_organic": {"hypothesis_id": "H5", "expected_direction": "mixed"},
    }


def single_series_oof_auc(x: pd.Series, y: pd.Series, n_splits: int = 5, random_state: int = 42) -> float:
    if not SKLEARN_OK:
        return np.nan
    y_num = pd.to_numeric(y, errors="coerce")
    ok = y_num.notna()
    x = x.loc[ok]
    yv = y_num.loc[ok].astype(int).to_numpy()
    if len(np.unique(yv)) < 2 or len(x) < 80:
        return np.nan

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.full(len(x), np.nan)

    if pd.api.types.is_numeric_dtype(x):
        xs = safe_series_to_numeric(x).astype(float)
        for tr, va in skf.split(np.zeros(len(yv)), yv):
            med = float(np.nanmedian(xs.iloc[tr].to_numpy()))
            if not np.isfinite(med):
                med = 0.0
            oof[va] = xs.iloc[va].fillna(med).to_numpy()
        return safe_auc(yv, oof)

    s = x.astype("object").fillna("<<MISSING>>").astype(str)
    for tr, va in skf.split(np.zeros(len(yv)), yv):
        tr_map = pd.DataFrame({"k": s.iloc[tr].values, "y": yv[tr]}).groupby("k")["y"].mean()
        tr_base = float(np.mean(yv[tr]))
        oof[va] = s.iloc[va].map(tr_map).fillna(tr_base).to_numpy()
    return safe_auc(yv, oof)


def hypothesis_univariate_signal(hfeat: pd.DataFrame, y: pd.Series, meta: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    rows = []
    y_num = pd.to_numeric(y, errors="coerce")
    base = float(np.nanmean(y_num))

    for c in hfeat.columns:
        s = hfeat[c]
        miss = float(s.isna().mean())
        nun = int(s.nunique(dropna=True))

        if pd.api.types.is_numeric_dtype(s):
            uniq = set(pd.Series(s.dropna().unique()))
            ftype = "binary" if uniq <= {0, 1} else "numeric"
        else:
            ftype = "categorical"

        if ftype == "numeric" and nun >= 8:
            gcol = quantile_bucket(s, q=5)
        else:
            gcol = s.astype("object").fillna("<<MISSING>>").astype(str)

        tmp = pd.DataFrame({"g": gcol, "y": y_num}).dropna(subset=["y"])
        grp = tmp.groupby("g", dropna=False)["y"].agg(["size", "mean"]).reset_index()
        grp = grp.rename(columns={"size": "count", "mean": "target_rate"})
        grp2 = grp[grp["count"] >= max(12, int(0.02 * len(tmp)))]
        if grp2.empty:
            grp2 = grp
        grp2 = grp2.sort_values("target_rate")

        worst = grp2.iloc[0]
        best = grp2.iloc[-1]
        effect = float(best["target_rate"] - worst["target_rate"])
        oof_auc = single_series_oof_auc(s, y_num)

        rows.append({
            "feature": c,
            "hypothesis_id": meta.get(c, {}).get("hypothesis_id", "HX"),
            "expected_direction": meta.get(c, {}).get("expected_direction", "mixed"),
            "feature_type": ftype,
            "missing_rate": miss,
            "nunique": nun,
            "oof_auc": oof_auc,
            "best_segment": str(best["g"]),
            "best_segment_count": int(best["count"]),
            "best_segment_rate": float(best["target_rate"]),
            "worst_segment": str(worst["g"]),
            "worst_segment_count": int(worst["count"]),
            "worst_segment_rate": float(worst["target_rate"]),
            "effect_range": effect,
            "lift_best_vs_base": float(best["target_rate"] / base) if base > 0 else np.nan,
        })

    out = pd.DataFrame(rows)
    return out.sort_values(["oof_auc", "effect_range"], ascending=[False, False])


def hypothesis_segment_effects(hfeat: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    y_num = pd.to_numeric(y, errors="coerce")
    base = float(np.nanmean(y_num))
    rows = []

    exp = hfeat["h1_prev_class_count"].copy()
    exp_seg = pd.Series(np.where(exp <= 0, "0", np.where(exp <= 1, "1", "2+")), index=exp.index)

    align = hfeat["h2_alignment_score"].copy()
    align_seg = pd.Series(np.where(align <= 1, "low", np.where(align <= 2, "mid", "high")), index=align.index)

    engage = hfeat["h3_engagement_score"].copy()
    engage_seg = pd.Series(np.where(engage <= 1, "low", np.where(engage <= 2, "mid", "high")), index=engage.index)

    complexity = quantile_bucket(hfeat["h4_response_complexity_score"], q=3, labels=["low", "mid", "high"])

    seg_map = {
        "H1_experience_bucket": exp_seg,
        "H2_alignment_bucket": align_seg,
        "H3_engagement_bucket": engage_seg,
        "H4_complexity_bucket": complexity,
    }

    for seg_name, seg_series in seg_map.items():
        tmp = pd.DataFrame({"seg": seg_series, "y": y_num}).dropna(subset=["y"])
        grp = tmp.groupby("seg")["y"].agg(["size", "mean"]).reset_index()
        for _, r in grp.iterrows():
            rows.append({
                "segment_feature": seg_name,
                "segment_value": r["seg"],
                "count": int(r["size"]),
                "target_rate": float(r["mean"]),
                "lift_vs_base": float(r["mean"] / base) if base > 0 else np.nan,
            })

    out = pd.DataFrame(rows)
    return out.sort_values(["segment_feature", "target_rate"], ascending=[True, False])


def hypothesis_interactions(hfeat: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    y_num = pd.to_numeric(y, errors="coerce")
    base = float(np.nanmean(y_num))
    candidate = [
        "h1_previous_any", "h1_reregistration_yes", "h2_major_data_flag",
        "h3_time_high", "h3_project_team", "h3_hope_offline", "h4_complexity_high",
    ]
    cols = [c for c in candidate if c in hfeat.columns]
    rows = []

    for a, b in itertools.combinations(cols, 2):
        tmp = pd.DataFrame({
            "a": (hfeat[a] > 0).astype(int),
            "b": (hfeat[b] > 0).astype(int),
            "y": y_num,
        }).dropna(subset=["y"])
        if len(tmp) < 80:
            continue
        grp = tmp.groupby(["a", "b"])["y"].agg(["size", "mean"]).reset_index()
        lookup = {(int(r["a"]), int(r["b"])): (int(r["size"]), float(r["mean"])) for _, r in grp.iterrows()}
        n11, r11 = lookup.get((1, 1), (0, np.nan))
        n10, r10 = lookup.get((1, 0), (0, np.nan))
        n01, r01 = lookup.get((0, 1), (0, np.nan))
        n00, r00 = lookup.get((0, 0), (0, np.nan))
        if n11 < 15:
            continue
        best_single = np.nanmax([r10, r01]) if np.isfinite(r10) or np.isfinite(r01) else np.nan
        synergy = r11 - best_single if np.isfinite(r11) and np.isfinite(best_single) else np.nan
        rows.append({
            "feature_a": a,
            "feature_b": b,
            "n11": n11,
            "rate11": r11,
            "n10": n10,
            "rate10": r10,
            "n01": n01,
            "rate01": r01,
            "n00": n00,
            "rate00": r00,
            "lift11_vs_base": (r11 / base) if np.isfinite(r11) and base > 0 else np.nan,
            "synergy_delta_vs_best_single": synergy,
        })
    if not rows:
        return pd.DataFrame(columns=[
            "feature_a", "feature_b", "n11", "rate11", "n10", "rate10",
            "n01", "rate01", "n00", "rate00", "lift11_vs_base", "synergy_delta_vs_best_single",
        ])
    return pd.DataFrame(rows).sort_values(["synergy_delta_vs_best_single", "rate11"], ascending=[False, False])


def hypothesis_feature_drift(hfeat_train: pd.DataFrame, hfeat_test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in hfeat_train.columns:
        tr = hfeat_train[c]
        te = hfeat_test[c] if c in hfeat_test.columns else pd.Series(np.nan, index=hfeat_train.index)
        if pd.api.types.is_numeric_dtype(tr) and tr.nunique(dropna=True) >= 8:
            metric = psi_numeric(tr, te, bins=10)
            mtype = "psi"
        else:
            metric = js_divergence_categorical(tr, te, topk=100)
            mtype = "js"
        rows.append({
            "feature": c,
            "shift_metric_type": mtype,
            "shift_metric_value": metric,
            "shift_level": shift_level(metric, mtype),
            "train_mean": float(pd.to_numeric(tr, errors="coerce").mean()) if pd.api.types.is_numeric_dtype(tr) else np.nan,
            "test_mean": float(pd.to_numeric(te, errors="coerce").mean()) if pd.api.types.is_numeric_dtype(tr) else np.nan,
            "train_missing_rate": float(tr.isna().mean()),
            "test_missing_rate": float(te.isna().mean()),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["shift_level", "shift_metric_value"], ascending=[True, False])


def summarize_hypothesis_evidence(univariate: pd.DataFrame, drift: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    rows = []
    drift_map = {}
    if drift is not None and not drift.empty:
        drift_map = drift.set_index("feature").to_dict("index")

    for _, h in catalog.iterrows():
        hid = h["hypothesis_id"]
        part = univariate[univariate["hypothesis_id"] == hid]
        if part.empty:
            rows.append({
                "hypothesis_id": hid,
                "title": h["title"],
                "n_features": 0,
                "best_feature": "",
                "best_oof_auc": np.nan,
                "best_effect_range": np.nan,
                "high_shift_ratio": np.nan,
                "evidence_level": "insufficient",
                "note": "해당 가설 피처 없음",
            })
            continue

        best = part.sort_values(["oof_auc", "effect_range"], ascending=[False, False]).iloc[0]
        high_shift = 0
        known_shift = 0
        for f in part["feature"].tolist():
            d = drift_map.get(f)
            if not d:
                continue
            known_shift += 1
            if d.get("shift_level") == "high":
                high_shift += 1
        high_shift_ratio = (high_shift / known_shift) if known_shift > 0 else np.nan

        best_auc = float(best["oof_auc"]) if np.isfinite(best["oof_auc"]) else np.nan
        best_effect = float(best["effect_range"]) if np.isfinite(best["effect_range"]) else np.nan

        level = "weak"
        if (np.isfinite(best_auc) and best_auc >= 0.57) or (np.isfinite(best_effect) and best_effect >= 0.16):
            level = "strong"
        elif (np.isfinite(best_auc) and best_auc >= 0.54) or (np.isfinite(best_effect) and best_effect >= 0.10):
            level = "moderate"
        if np.isfinite(high_shift_ratio) and high_shift_ratio >= 0.4:
            level = f"{level}_with_shift_risk"

        rows.append({
            "hypothesis_id": hid,
            "title": h["title"],
            "n_features": int(part.shape[0]),
            "best_feature": best["feature"],
            "best_oof_auc": best_auc,
            "best_effect_range": best_effect,
            "high_shift_ratio": high_shift_ratio,
            "evidence_level": level,
            "note": f"best={best['feature']}",
        })

    return pd.DataFrame(rows)


def build_hypothesis_feature_blueprint(univariate: pd.DataFrame, drift: pd.DataFrame, leak_v2: pd.DataFrame) -> pd.DataFrame:
    d = univariate.copy()
    if drift is not None and not drift.empty:
        d = d.merge(
            drift[["feature", "shift_metric_type", "shift_metric_value", "shift_level"]],
            on="feature",
            how="left",
        )
    else:
        d["shift_metric_type"] = ""
        d["shift_metric_value"] = np.nan
        d["shift_level"] = "unknown"

    leak_cols = set(leak_v2["column"].tolist()) if leak_v2 is not None and not leak_v2.empty else set()

    penalties = d["shift_level"].map({"high": 0.08, "medium": 0.03, "low": 0.0, "unknown": 0.02}).fillna(0.02)
    d["priority_score"] = (d["oof_auc"].fillna(0.5) - 0.5) * 2.0 + d["effect_range"].fillna(0.0) - penalties
    d["is_leakage_review"] = d["feature"].isin(leak_cols).astype(int)

    actions = []
    for _, r in d.iterrows():
        if int(r["is_leakage_review"]) == 1:
            actions.append("REVIEW_LEAKAGE_FIRST")
        elif r["shift_level"] == "high" and (not np.isfinite(r["oof_auc"]) or r["oof_auc"] < 0.54):
            actions.append("LOW_PRIORITY_OR_REGULARIZE")
        elif np.isfinite(r["oof_auc"]) and r["oof_auc"] >= 0.56 and r["shift_level"] in ("low", "medium"):
            actions.append("HIGH_PRIORITY_FEATURE")
        elif np.isfinite(r["effect_range"]) and r["effect_range"] >= 0.10:
            actions.append("TRY_WITH_REGULARIZATION")
        else:
            actions.append("BACKUP_FEATURE")
    d["blueprint_action"] = actions

    cols = [
        "feature", "hypothesis_id", "expected_direction", "feature_type",
        "oof_auc", "effect_range", "shift_level", "shift_metric_type", "shift_metric_value",
        "is_leakage_review", "priority_score", "blueprint_action",
        "best_segment", "best_segment_rate", "worst_segment", "worst_segment_rate",
    ]
    return d[cols].sort_values(["is_leakage_review", "priority_score"], ascending=[False, False])


# =========================
# 메인 EDA
# =========================
def run_eda(
    train_path: str,
    target: str = "completed",
    id_col: str = "ID",
    outdir: str = "eda_output_v3",
    test_path: str = "open/test.csv",
):
    train_path = Path(train_path)
    if not train_path.exists():
        raise FileNotFoundError(f"파일이 없습니다: {train_path}")
    test_path_p = Path(test_path) if test_path else None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(outdir) / ts
    reports = base / "reports"
    figures = base / "figures"
    artifacts = base / "artifacts"
    ensure_dir(reports)
    ensure_dir(figures)
    ensure_dir(artifacts)

    # 1) 로드
    df = pd.read_csv(train_path)
    n_rows, n_cols = df.shape
    df_test = None
    if test_path_p is not None and test_path_p.exists():
        df_test = pd.read_csv(test_path_p)

    # 2) 기본 요약
    summary_lines = [
        f"[DATA] path={train_path}",
        f"[SHAPE] rows={n_rows}, cols={n_cols}",
        f"[COLUMNS] {list(df.columns)}",
    ]
    if df_test is not None:
        summary_lines += [
            "",
            f"[TEST] path={test_path_p}",
            f"[TEST_SHAPE] rows={df_test.shape[0]}, cols={df_test.shape[1]}",
        ]

    if target not in df.columns:
        raise ValueError(f"target 컬럼 '{target}' 이(가) 없습니다. 실제 타깃명을 확인하세요.")

    y = df[target]
    y_counts = y.value_counts(dropna=False)
    y_pos_rate = float((y == 1).mean()) if is_binary_target(y) else float("nan")

    summary_lines += [
        "",
        "[TARGET]",
        str(y_counts),
        f"pos_rate(target=1)={y_pos_rate:.6f}",
    ]

    if id_col in df.columns:
        id_unique = df[id_col].nunique(dropna=True)
        summary_lines += [
            "",
            "[ID CHECK]",
            f"ID unique={id_unique} / rows={n_rows} (unique_ratio={id_unique / max(n_rows, 1):.6f})",
        ]

    save_text(reports / "00_overall_summary.txt", "\n".join(summary_lines))

    # 3) 스키마
    missing_rate = df.isna().mean().sort_values(ascending=False)
    nunique = df.nunique(dropna=True).sort_values(ascending=False)
    dtypes = df.dtypes.astype(str)

    pd.DataFrame(
        {
            "dtype": dtypes,
            "nunique": nunique.reindex(df.columns).values,
            "missing_rate": missing_rate.reindex(df.columns).values,
            "missing_count": df.isna().sum().reindex(df.columns).values,
        },
        index=df.columns,
    ).to_csv(reports / "01_schema_missing_unique.csv", encoding="utf-8-sig")

    dup_rows = int(df.duplicated().sum())
    save_text(reports / "02_duplicates.txt", f"duplicate_rows={dup_rows}")

    plot_missing_bar(missing_rate, figures / "missing_top30.png", topk=30)
    plot_target_distribution(y, figures / "target_distribution.png")

    # 4) 타입 분류
    col_types = {c: classify_column(df[c], c, target) for c in df.columns}
    pd.DataFrame({"column": list(col_types.keys()), "type": list(col_types.values())}).to_csv(
        reports / "03_column_types.csv", index=False, encoding="utf-8-sig"
    )

    # 5) 그룹
    numeric_cols = [c for c, t in col_types.items() if t in ("numeric_continuous", "numeric_discrete")]
    cat_low_cols = [c for c, t in col_types.items() if t == "categorical_low"]
    cat_high_cols = [c for c, t in col_types.items() if t == "categorical_high"]
    text_cols = [c for c, t in col_types.items() if t == "text"]
    constant_cols = [c for c, t in col_types.items() if t == "constant"]
    id_like_cols = [c for c, t in col_types.items() if t == "id_like"]
    datetime_like_cols = [c for c, t in col_types.items() if t == "datetime_like"]

    save_text(
        reports / "04_column_groups.txt",
        "\n".join(
            [
                f"numeric_cols({len(numeric_cols)}): {numeric_cols}",
                f"cat_low_cols({len(cat_low_cols)}): {cat_low_cols}",
                f"cat_high_cols({len(cat_high_cols)}): {cat_high_cols}",
                f"text_cols({len(text_cols)}): {text_cols}",
                f"datetime_like_cols({len(datetime_like_cols)}): {datetime_like_cols}",
                f"id_like_cols({len(id_like_cols)}): {id_like_cols}",
                f"constant_cols({len(constant_cols)}): {constant_cols}",
            ]
        ),
    )

    # 6) 수치형 기술통계 + 이상치
    num_desc_rows = []
    outlier_rows = []
    for c in numeric_cols:
        desc = describe_numeric(df[c])
        if desc:
            desc["column"] = c
            num_desc_rows.append(desc)

        x = safe_series_to_numeric(df[c]) if df[c].dtype == "object" else df[c]
        lo, hi, mask = iqr_outlier_mask(x, k=1.5)
        if mask is not None:
            out_cnt = int(mask.sum())
            out_rate = float(out_cnt / max(len(pd.Series(x).dropna()), 1))
            outlier_rows.append(
                {
                    "column": c,
                    "iqr_low": float(lo),
                    "iqr_high": float(hi),
                    "outlier_count": out_cnt,
                    "outlier_rate": out_rate,
                }
            )

            out_idx = pd.Series(x).dropna().index[mask]
            cols = ([id_col] if id_col in df.columns else []) + [c, target]
            df.loc[out_idx, cols].head(10).to_csv(
                artifacts / f"outlier_samples__{safe_filename(c)}.csv", index=False, encoding="utf-8-sig"
            )

        plot_numeric_hist(df[c], c, figures / f"hist__{safe_filename(c)}.png")
        plot_numeric_by_target(df, c, target, figures / f"box__{safe_filename(c)}_by_target.png")

    if num_desc_rows:
        pd.DataFrame(num_desc_rows).set_index("column").to_csv(reports / "05_numeric_describe.csv", encoding="utf-8-sig")
    if outlier_rows:
        pd.DataFrame(outlier_rows).sort_values("outlier_rate", ascending=False).to_csv(
            reports / "06_numeric_outliers_iqr.csv", index=False, encoding="utf-8-sig"
        )

    # 7) 범주형 기본 + chi2(기존)
    cat_rows = []
    for c in (cat_low_cols + cat_high_cols):
        s = df[c].astype("object")
        vc = s.value_counts(dropna=False)
        top = vc.head(20)

        cat_rows.append(
            {
                "column": c,
                "nunique": int(s.nunique(dropna=True)),
                "missing_rate": float(s.isna().mean()),
                "top20_values": "; ".join([f"{str(k)}:{int(v)}" for k, v in top.items()]),
            }
        )

        g = df.groupby(c, dropna=False)[target].agg(["count", "mean"]).reset_index()
        g = g.sort_values(["mean", "count"], ascending=[False, False])
        g.to_csv(artifacts / f"cat_target_table__{safe_filename(c)}.csv", index=False, encoding="utf-8-sig")

        plot_categorical_target_rate(df, c, target, figures / f"cat_target_rate__{safe_filename(c)}.png", topk=15)

        if SCIPY_OK and df[c].nunique(dropna=True) <= 50:
            s2 = df[c].astype("object").fillna("<<MISSING>>")
            ct = pd.crosstab(s2, df[target])
            if ct.shape[0] >= 2 and ct.shape[1] == 2:
                try:
                    chi2, p, dof, _ = chi2_contingency(ct.values)
                    save_text(
                        artifacts / f"chi2__{safe_filename(c)}.txt",
                        f"chi2={chi2:.6f}\np={p:.6e}\ndof={dof}\nshape={ct.shape}",
                    )
                except Exception as e:
                    save_text(artifacts / f"chi2__{safe_filename(c)}.txt", f"failed: {repr(e)}")

    if cat_rows:
        pd.DataFrame(cat_rows).sort_values(["nunique", "missing_rate"], ascending=[False, False]).to_csv(
            reports / "07_categorical_summary.csv", index=False, encoding="utf-8-sig"
        )

    # 8) 텍스트형 기본 진단
    text_rows = []
    for c in text_cols:
        s = df[c].astype("object")
        s_non = s.dropna().astype(str)
        lengths = s_non.map(len)
        empty_rate = float((s_non.str.strip() == "").mean()) if len(s_non) > 0 else 0.0

        sample = s_non.head(500)
        tokens = []
        for line in sample:
            tokens.extend(TOKEN_PAT.findall(line.lower()))
        top_tokens = pd.Series(tokens).value_counts().head(30) if len(tokens) > 0 else pd.Series(dtype=int)

        text_rows.append(
            {
                "column": c,
                "missing_rate": float(s.isna().mean()),
                "avg_len": float(lengths.mean()) if len(lengths) > 0 else 0.0,
                "p50_len": float(lengths.median()) if len(lengths) > 0 else 0.0,
                "p95_len": float(np.percentile(lengths, 95)) if len(lengths) > 0 else 0.0,
                "empty_rate(non-null)": empty_rate,
                "nunique": int(s.nunique(dropna=True)),
                "top_tokens(sample500)": "; ".join([f"{k}:{int(v)}" for k, v in top_tokens.items()]),
            }
        )

        if len(lengths) > 0:
            plt.figure(figsize=(7, 4))
            plt.hist(lengths, bins=30)
            plt.title(f"Text Length Histogram: {c}")
            plt.xlabel("length")
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(figures / f"text_len__{safe_filename(c)}.png", dpi=160)
            plt.close()

    if text_rows:
        pd.DataFrame(text_rows).sort_values("avg_len", ascending=False).to_csv(
            reports / "08_text_columns_profile.csv", index=False, encoding="utf-8-sig"
        )

    # 9) 결측 플래그 신호
    miss_signal = []
    y_int = pd.to_numeric(df[target], errors="coerce")
    for c in df.columns:
        if c == target:
            continue
        miss = df[c].isna().astype(int)
        ok = y_int.notna()
        if ok.sum() >= 50:
            try:
                r = np.corrcoef(miss[ok], y_int[ok])[0, 1]
            except Exception:
                r = np.nan
            miss_signal.append({"column": c, "missing_rate": float(df[c].isna().mean()), "corr_with_target": r})

    miss_signal_df = pd.DataFrame(miss_signal)
    if miss_signal_df.empty:
        miss_signal_df = pd.DataFrame(columns=["column", "missing_rate", "corr_with_target"])
    else:
        miss_signal_df = miss_signal_df.sort_values("corr_with_target", key=lambda x: x.abs(), ascending=False)
    miss_signal_df.to_csv(reports / "09_missing_flag_signal.csv", index=False, encoding="utf-8-sig")

    # 10) 수치형 상관
    df_num = pd.DataFrame()
    for c in numeric_cols:
        df_num[c] = safe_series_to_numeric(df[c]) if df[c].dtype == "object" else df[c]
    df_num[target] = pd.to_numeric(df[target], errors="coerce")

    if df_num.shape[1] >= 3:
        correlation_heatmap(df_num.drop(columns=[target], errors="ignore"), figures / "corr_heatmap_numeric.png")

    corr_rows = []
    for c in numeric_cols:
        x = pd.to_numeric(df_num[c], errors="coerce")
        ok = x.notna() & df_num[target].notna()
        if ok.sum() >= 50:
            try:
                r = np.corrcoef(x[ok], df_num[target][ok])[0, 1]
            except Exception:
                r = np.nan
            corr_rows.append({"column": c, "corr_with_target": r, "missing_rate": float(df[c].isna().mean())})

    corr_df = pd.DataFrame(corr_rows)
    if corr_df.empty:
        corr_df = pd.DataFrame(columns=["column", "corr_with_target", "missing_rate"])
    else:
        corr_df = corr_df.sort_values("corr_with_target", key=lambda x: x.abs(), ascending=False)
    corr_df.to_csv(reports / "10_numeric_corr_with_target.csv", index=False, encoding="utf-8-sig")

    # 11) MI
    if SKLEARN_OK and is_binary_target(df[target]):
        X_cols = [c for c in df.columns if c != target]
        X = df[X_cols].copy().drop(columns=text_cols, errors="ignore")

        for c in X.columns:
            if pd.api.types.is_numeric_dtype(X[c]):
                continue
            sn = safe_series_to_numeric(X[c])
            if sn.notna().mean() >= 0.95:
                X[c] = sn
            else:
                X[c] = X[c].astype("object").fillna("<<MISSING>>")
                le = LabelEncoder()
                X[c] = le.fit_transform(X[c].astype(str))

        X = X.fillna(-999)
        y_mi = df[target].astype(int)
        try:
            mi = mutual_info_classif(X, y_mi, discrete_features="auto", random_state=42)
            mi_df = pd.DataFrame({"column": X.columns, "mutual_info": mi}).sort_values("mutual_info", ascending=False)
            mi_df.to_csv(reports / "11_mutual_info_rank.csv", index=False, encoding="utf-8-sig")
        except Exception as e:
            save_text(reports / "11_mutual_info_rank.txt", f"mutual_info failed: {repr(e)}")

    # 12) 누수 후보(v1)
    leak_v1 = leakage_suspects_v1_style(df, target, col_types)
    leak_v1.to_csv(reports / "12_leakage_suspects.csv", index=False, encoding="utf-8-sig")

    # 13) 전처리 초안(v1 호환)
    decision_rows = []
    for c in df.columns:
        if c == target:
            continue
        miss = float(df[c].isna().mean())
        t = col_types.get(c, "unknown")

        drop_reason = ""
        recommend = ""

        if miss >= 0.999:
            drop_reason = "missing_~100%"
            recommend = "DROP"
        elif t == "constant":
            drop_reason = "constant"
            recommend = "DROP"
        elif t == "id_like" or c == id_col:
            drop_reason = "id_like"
            recommend = "EXCLUDE_FROM_FEATURES"
        elif t in ("numeric_continuous", "numeric_discrete"):
            recommend = "NUMERIC(consider clip/impute/scale/binning)"
        elif t == "categorical_low":
            recommend = "CATEGORICAL_LOW(onehot + missing_as_category)"
        elif t == "categorical_high":
            recommend = "CATEGORICAL_HIGH(target/impact encoding or hashing; handle unseen)"
        elif t == "text":
            recommend = "TEXT(tfidf/char-ngram/embedding; clean/normalize)"
        elif t == "datetime_like":
            recommend = "DATETIME(parse -> year/month/day/elapsed)"
        else:
            recommend = "REVIEW"

        decision_rows.append(
            {
                "column": c,
                "type": t,
                "missing_rate": miss,
                "nunique": int(df[c].nunique(dropna=True)),
                "recommendation": recommend,
                "drop_reason": drop_reason,
            }
        )

    decision_df_v1 = pd.DataFrame(decision_rows).sort_values(
        ["drop_reason", "missing_rate", "nunique"], ascending=[False, False, False]
    )
    decision_df_v1.to_csv(reports / "13_preprocessing_decision_draft.csv", index=False, encoding="utf-8-sig")

    # =========================
    # v2 고도화 블록
    # =========================
    all_cat_cols = cat_low_cols + cat_high_cols

    # 14) 범주형 연관강도 요약 (chi2 + Cramer's V)
    cat_assoc = categorical_association_summary(df, all_cat_cols, target)
    cat_assoc.to_csv(reports / "14_categorical_association_strength.csv", index=False, encoding="utf-8-sig")

    # 15) 멀티선택 토큰 신호
    multiselect_cols = [c for c in all_cat_cols if is_multiselect_like(df[c])]
    multi_token_df = analyze_multiselect_tokens(df, multiselect_cols, target, artifacts_dir=artifacts, figures_dir=figures, min_count=8)
    multi_token_df.to_csv(reports / "15_multiselect_token_signal.csv", index=False, encoding="utf-8-sig")

    # 16) 수치형 bin 추세
    num_trend_df = numeric_binned_trend(df, numeric_cols, target, artifacts_dir=artifacts, max_bins=10)
    num_trend_df.to_csv(reports / "16_numeric_binned_target_trend.csv", index=False, encoding="utf-8-sig")

    # 17) 텍스트 토큰 log-odds
    text_sig_df = text_token_logodds(df, text_cols, target, min_df=8)
    text_sig_df.to_csv(reports / "17_text_token_logodds.csv", index=False, encoding="utf-8-sig")

    # 18) 수치형 고상관 쌍
    num_pair_df = high_corr_numeric_pairs(df, numeric_cols, threshold=0.85)
    num_pair_df.to_csv(reports / "18_high_corr_pairs_numeric.csv", index=False, encoding="utf-8-sig")

    # 19) 범주형 고연관 쌍
    cat_pair_df = high_assoc_categorical_pairs(df, all_cat_cols, threshold=0.85, max_nunique=20)
    cat_pair_df.to_csv(reports / "19_high_assoc_pairs_categorical.csv", index=False, encoding="utf-8-sig")

    # 20) 결측패턴 동일 그룹
    miss_group_df = missing_pattern_groups(df, target)
    miss_group_df.to_csv(reports / "20_missing_pattern_groups.csv", index=False, encoding="utf-8-sig")

    # 21) 단일 피처 OOF AUC
    auc_scan_df = single_feature_auc_cv(df, target, col_types, n_splits=5, random_state=42)
    auc_scan_df.to_csv(reports / "21_single_feature_auc_cv.csv", index=False, encoding="utf-8-sig")

    # 22) 누수 의심(v2 통합)
    leak_rows = []
    if not leak_v1.empty:
        for _, r in leak_v1.iterrows():
            leak_rows.append({"column": r["column"], "reason": f"v1:{r['reason']}"})

    if not auc_scan_df.empty:
        sus = auc_scan_df[auc_scan_df["is_suspect"] == 1]
        for _, r in sus.iterrows():
            reason = f"auc_cv:{r['suspect_reason']}|oof_auc={r['oof_auc']:.3f}|gap={r['auc_gap']:.3f}"
            leak_rows.append({"column": r["column"], "reason": reason})

    leak_v2 = pd.DataFrame(leak_rows).drop_duplicates() if leak_rows else pd.DataFrame(columns=["column", "reason"])
    if not leak_v2.empty:
        leak_v2 = leak_v2.sort_values(["column", "reason"])  # noqa: C408
    leak_v2.to_csv(reports / "22_leakage_suspects_v2.csv", index=False, encoding="utf-8-sig")

    # 23) 전처리 의사결정 v2
    auc_map = {}
    if not auc_scan_df.empty:
        auc_map = auc_scan_df.set_index("column")[["oof_auc", "auc_gap", "is_suspect", "suspect_reason"]].to_dict("index")

    # 결측그룹 id 부여
    miss_group_map = {}
    if not miss_group_df.empty:
        for gid, cols_str in enumerate(miss_group_df["columns"].tolist(), start=1):
            cols = [x.strip() for x in str(cols_str).split("|")]
            for c in cols:
                miss_group_map[c] = gid

    leak_cols_v2 = set(leak_v2["column"].tolist()) if not leak_v2.empty else set()
    multi_cols = set(multiselect_cols)

    decision_v2_rows = []
    for c in df.columns:
        if c == target:
            continue

        miss = float(df[c].isna().mean())
        nun = int(df[c].nunique(dropna=True))
        t = col_types.get(c, "unknown")

        auc_info = auc_map.get(c, {})
        oof_auc = auc_info.get("oof_auc", np.nan)
        auc_gap = auc_info.get("auc_gap", np.nan)
        auc_sus = int(auc_info.get("is_suspect", 0))
        auc_reason = auc_info.get("suspect_reason", "")

        drop_reason = ""
        recommend = ""

        if miss >= 0.999:
            drop_reason = "missing_~100%"
            recommend = "DROP"
        elif t == "constant":
            drop_reason = "constant"
            recommend = "DROP"
        elif t == "id_like" or c == id_col:
            drop_reason = "id_like"
            recommend = "EXCLUDE_FROM_FEATURES"
        elif miss >= 0.98:
            drop_reason = "very_high_missing"
            recommend = "DROP_OR_MISSING_FLAG_ONLY"
        elif c in leak_cols_v2:
            drop_reason = "leakage_review_required"
            recommend = "REVIEW_LEAKAGE_BEFORE_MODELING"
        elif c in multi_cols:
            recommend = "MULTISELECT(split tokens -> multi-hot/target-encode)"
        elif t in ("numeric_continuous", "numeric_discrete"):
            recommend = "NUMERIC(impute + clip + optional binning)"
        elif t == "categorical_low":
            recommend = "CATEGORICAL_LOW(onehot + rare merge + missing_as_category)"
        elif t == "categorical_high":
            recommend = "CATEGORICAL_HIGH(target/impact encoding + rare merge)"
        elif t == "text":
            recommend = "TEXT(tfidf/embedding + custom normalization)"
        elif t == "datetime_like":
            recommend = "DATETIME(parse -> calendar/elapsed features)"
        else:
            recommend = "REVIEW"

        decision_v2_rows.append(
            {
                "column": c,
                "type": t,
                "missing_rate": miss,
                "nunique": nun,
                "is_multiselect_like": int(c in multi_cols),
                "missing_pattern_group_id": miss_group_map.get(c, ""),
                "single_feature_oof_auc": oof_auc,
                "single_feature_auc_gap": auc_gap,
                "single_feature_auc_suspect": auc_sus,
                "single_feature_auc_suspect_reason": auc_reason,
                "recommendation_v2": recommend,
                "drop_reason_v2": drop_reason,
            }
        )

    decision_v2 = pd.DataFrame(decision_v2_rows).sort_values(
        ["drop_reason_v2", "single_feature_oof_auc", "missing_rate", "nunique"],
        ascending=[False, False, False, False],
    )
    decision_v2.to_csv(reports / "23_preprocessing_decision_v2.csv", index=False, encoding="utf-8-sig")

    # =========================
    # v3 가설 기반 블록
    # =========================
    catalog = hypothesis_catalog()
    catalog.to_csv(reports / "24_hypothesis_catalog.csv", index=False, encoding="utf-8-sig")

    hfeat_train = build_hypothesis_features(df)
    hmeta = hypothesis_feature_meta()
    hfeat_train.to_csv(artifacts / "hypothesis_features_train.csv", index=False, encoding="utf-8-sig")

    # 피처 스냅샷
    snapshot_rows = []
    for c in hfeat_train.columns:
        s = hfeat_train[c]
        snapshot_rows.append({
            "feature": c,
            "hypothesis_id": hmeta.get(c, {}).get("hypothesis_id", "HX"),
            "expected_direction": hmeta.get(c, {}).get("expected_direction", "mixed"),
            "dtype": str(s.dtype),
            "train_missing_rate": float(s.isna().mean()),
            "train_nunique": int(s.nunique(dropna=True)),
            "train_mean": float(pd.to_numeric(s, errors="coerce").mean()) if pd.api.types.is_numeric_dtype(s) else np.nan,
            "train_std": float(pd.to_numeric(s, errors="coerce").std()) if pd.api.types.is_numeric_dtype(s) else np.nan,
        })
    pd.DataFrame(snapshot_rows).to_csv(reports / "25_hypothesis_feature_snapshot.csv", index=False, encoding="utf-8-sig")

    # 단일 효과(OOF + 세그먼트 range)
    h_uni = hypothesis_univariate_signal(hfeat_train, df[target], hmeta)
    h_uni.to_csv(reports / "26_hypothesis_univariate_signal.csv", index=False, encoding="utf-8-sig")

    # 가설 세그먼트 분석
    h_seg = hypothesis_segment_effects(hfeat_train, df[target])
    h_seg.to_csv(reports / "27_hypothesis_segment_effects.csv", index=False, encoding="utf-8-sig")

    # 핵심 상호작용
    h_inter = hypothesis_interactions(hfeat_train, df[target])
    h_inter.to_csv(reports / "28_hypothesis_interactions.csv", index=False, encoding="utf-8-sig")

    # train/test 드리프트
    if df_test is not None:
        hfeat_test = build_hypothesis_features(df_test)
        hfeat_test.to_csv(artifacts / "hypothesis_features_test.csv", index=False, encoding="utf-8-sig")
        h_drift = hypothesis_feature_drift(hfeat_train, hfeat_test)
    else:
        h_drift = pd.DataFrame(columns=[
            "feature", "shift_metric_type", "shift_metric_value", "shift_level",
            "train_mean", "test_mean", "train_missing_rate", "test_missing_rate",
        ])
    h_drift.to_csv(reports / "29_hypothesis_feature_drift_train_test.csv", index=False, encoding="utf-8-sig")

    # 가설별 증거 요약
    h_evidence = summarize_hypothesis_evidence(h_uni, h_drift, catalog)
    h_evidence.to_csv(reports / "30_hypothesis_evidence_summary.csv", index=False, encoding="utf-8-sig")

    # 모델링 직전 피처 우선순위 권고안
    h_blueprint = build_hypothesis_feature_blueprint(h_uni, h_drift, leak_v2)
    h_blueprint.to_csv(reports / "31_hypothesis_feature_blueprint.csv", index=False, encoding="utf-8-sig")

    # 짧은 텍스트 요약
    top_feats = h_blueprint.head(10)["feature"].tolist() if not h_blueprint.empty else []
    top_hyp = (
        h_evidence.sort_values(["best_oof_auc", "best_effect_range"], ascending=[False, False]).head(5)["hypothesis_id"].tolist()
        if not h_evidence.empty else []
    )
    save_text(
        reports / "32_hypothesis_key_findings.txt",
        "\n".join([
            "[GOAL] completed(수료여부) 예측 성능에 직접 기여할 가설 우선 검증",
            f"[TOP_FEATURES] {top_feats}",
            f"[HYPOTHESIS_EVIDENCE] {top_hyp}",
            "[NOTE] shift_level=high 인 피처는 강한 정규화/인코딩 안전장치 없이 직접 사용 금지",
        ]),
    )

    # 99) 완료 로그
    done_msg = [
        "EDA V3 DONE",
        f"output_dir={base}",
        f"reports={reports}",
        f"figures={figures}",
        f"artifacts={artifacts}",
        f"scipy={SCIPY_OK}, sklearn={SKLEARN_OK}",
        f"multiselect_cols={len(multiselect_cols)}",
        f"test_loaded={df_test is not None}",
        f"hypothesis_features={hfeat_train.shape[1]}",
    ]
    save_text(reports / "99_done.txt", "\n".join(done_msg))

    print("\n".join(done_msg))
    return base


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="open/train.csv")
    parser.add_argument("--test_path", type=str, default="open/test.csv")
    parser.add_argument("--target", type=str, default="completed")
    parser.add_argument("--id_col", type=str, default="ID")
    parser.add_argument("--outdir", type=str, default="eda_output_v3")
    args = parser.parse_args()

    run_eda(
        train_path=args.train_path,
        test_path=args.test_path,
        target=args.target,
        id_col=args.id_col,
        outdir=args.outdir,
    )
