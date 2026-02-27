# -*- coding: utf-8 -*-
"""
완전 EDA 스크립트 (train.csv 기준)
- 경로: /mnt/data/train.csv
- 산출물: eda_output/YYYYMMDD_HHMMSS/ 아래 reports/, figures/, artifacts/
- 의존성: pandas, numpy, matplotlib, scipy (선택), scikit-learn(선택)
"""

import os
import re
import json
import math
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# (선택) 통계검정/모델 기반 중요도
try:
    from scipy.stats import chi2_contingency, pointbiserialr
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# =========================
# 유틸
# =========================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


def safe_series_to_numeric(s: pd.Series) -> pd.Series:
    """문자열 혼입 가능성을 고려해 숫자로 강제 변환(실패는 NaN)."""
    return pd.to_numeric(s, errors="coerce")


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

    # 상수열
    nun = series.nunique(dropna=True)
    if nun <= 1:
        return "constant"

    # datetime like: 문자열/숫자 섞여 있을 수 있으니 휴리스틱
    if series.dtype == "object":
        # 날짜 패턴(YYYY-MM-DD / YYYY/MM/DD / YYYYMMDD 등)
        sample = series.dropna().astype(str).head(200)
        if len(sample) > 0:
            date_hits = sample.str.contains(r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})|(\d{8})", regex=True).mean()
            if date_hits >= 0.6:
                return "datetime_like"

    # ID like: 유니크가 매우 크고(대부분 유니크) 문자열/정수 형태
    #   - 행 수 대비 유니크 비율이 높고
    #   - 값 길이가 일정/해시/UUID/숫자 코드일 가능성
    n = len(series)
    uniq_ratio = nun / max(n, 1)
    if uniq_ratio >= 0.98:
        # 거의 대부분 유니크이면 ID일 가능성 높음
        return "id_like"

    # numeric 판별
    if pd.api.types.is_numeric_dtype(series):
        # 이산/연속 휴리스틱: 유니크 수가 작으면 이산
        if nun <= 30:
            return "numeric_discrete"
        return "numeric_continuous"

    # object인데 숫자처럼 보일 때
    if series.dtype == "object":
        sn = safe_series_to_numeric(series)
        # 숫자 변환 성공률이 높으면 numeric 후보
        conv_ratio = sn.notna().mean()
        if conv_ratio >= 0.95:
            # 변환 가능한데 유니크가 작으면 이산 코드일 가능성
            nun_num = sn.nunique(dropna=True)
            if nun_num <= 30:
                return "numeric_discrete"
            return "numeric_continuous"

    # 범주 vs 텍스트
    if series.dtype == "object":
        # 평균 길이, 유니크 비율
        s_str = series.dropna().astype(str)
        avg_len = s_str.map(len).mean() if len(s_str) > 0 else 0
        nun = series.nunique(dropna=True)
        # 저/고 카디널
        if nun <= 20 and avg_len <= 40:
            return "categorical_low"
        if nun <= 200 and avg_len <= 60:
            return "categorical_high"

        # 길이가 길고 유니크가 큰 경우 텍스트로 간주
        return "text"

    # 기본 fallback
    return "unknown"


def describe_numeric(series: pd.Series):
    s = safe_series_to_numeric(series) if series.dtype == "object" else series
    s = s.dropna()
    if len(s) == 0:
        return {}
    desc = {
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
    return desc


def iqr_outlier_mask(x: pd.Series, k: float = 1.5):
    """IQR 기반 이상치 마스크"""
    x = x.dropna()
    if len(x) < 8:
        return None, None, None
    q1, q3 = np.percentile(x, 25), np.percentile(x, 75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return lo, hi, (x < lo) | (x > hi)


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
    plt.xlabel("completed")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_numeric_hist(series: pd.Series, name: str, outpath: Path):
    x = safe_series_to_numeric(series) if series.dtype == "object" else series
    x = x.dropna()
    if len(x) == 0:
        return
    # 극단값 때문에 히스토그램이 망가지지 않게 1~99 분위로 제한한 버전도 저장
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
    y = df[target]
    # boxplot: target별 분포
    data0 = x[y == 0].dropna()
    data1 = x[y == 1].dropna()
    if len(data0) + len(data1) == 0:
        return
    plt.figure(figsize=(6, 4))
    plt.boxplot([data0.values, data1.values], labels=["0", "1"], showfliers=False)
    plt.title(f"{col} by {target} (box, no fliers)")
    plt.xlabel(target)
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_categorical_target_rate(df: pd.DataFrame, col: str, target: str, outpath: Path, topk: int = 15):
    # 상위 빈도 카테고리(topk)만 시각화
    s = df[col].astype("object")
    vc = s.value_counts(dropna=False)
    cats = vc.head(topk).index
    tmp = df[df[col].isin(cats)].copy()
    if tmp.empty:
        return
    rate = tmp.groupby(col)[target].mean().reindex(cats)
    cnt = tmp.groupby(col)[target].size().reindex(cats)

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(rate)), rate.values)
    plt.xticks(range(len(rate)), [str(c) for c in rate.index], rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.title(f"{col}: Target Rate (top {topk} categories)")
    plt.ylabel("P(completed=1)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

    # count도 함께 저장
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(cnt)), cnt.values)
    plt.xticks(range(len(cnt)), [str(c) for c in cnt.index], rotation=45, ha="right")
    plt.title(f"{col}: Category Counts (top {topk})")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(outpath.with_name(outpath.stem + "_counts.png"), dpi=160)
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


def leakage_suspects(df: pd.DataFrame, target: str, col_types: dict):
    """
    누수 후보 컬럼을 '자동'으로 걸러내기 위한 휴리스틱:
    1) 컬럼명 키워드(complete, pass, result, score 등)
    2) target과의 과도한 상관/분리
    3) 결측 패턴이 target과 거의 동일
    """
    suspects = []

    # 1) 이름 기반
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

    # 2) 수치형 상관/point-biserial
    y = df[target]
    for c, t in col_types.items():
        if c == target:
            continue
        if t in ("numeric_continuous", "numeric_discrete"):
            x = safe_series_to_numeric(df[c])
            # 결측 제거 후
            ok = x.notna() & y.notna()
            if ok.sum() >= 50:
                # 상관(단순)
                try:
                    corr = np.corrcoef(x[ok].astype(float), y[ok].astype(float))[0, 1]
                except Exception:
                    corr = np.nan
                if np.isfinite(corr) and abs(corr) >= 0.75:
                    suspects.append((c, f"high_corr(|r|={corr:.3f})"))

    # 3) 결측 패턴이 타깃과 강한 연관
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

    # 중복 제거
    out = pd.DataFrame(suspects, columns=["column", "reason"]).drop_duplicates()
    return out.sort_values(["reason", "column"])


# =========================
# 메인 EDA
# =========================
def run_eda(train_path: str, target: str = "completed", id_col: str = "ID", outdir: str = "eda_output"):
    train_path = Path(train_path)
    assert train_path.exists(), f"파일이 없습니다: {train_path}"

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

    # 2) 기본 요약
    summary_lines = []
    summary_lines.append(f"[DATA] path={train_path}")
    summary_lines.append(f"[SHAPE] rows={n_rows}, cols={n_cols}")
    summary_lines.append(f"[COLUMNS] {list(df.columns)}")

    # 타깃 확인
    if target not in df.columns:
        raise ValueError(f"target 컬럼 '{target}' 이(가) 없습니다. 실제 타깃명을 확인하세요.")
    y = df[target]
    y_counts = y.value_counts(dropna=False)
    y_pos_rate = float((y == 1).mean()) if set(y.dropna().unique()) <= {0, 1} else float("nan")

    summary_lines.append("\n[TARGET]")
    summary_lines.append(str(y_counts))
    summary_lines.append(f"pos_rate(completed=1)={y_pos_rate:.6f}")

    # ID 유일성
    if id_col in df.columns:
        id_unique = df[id_col].nunique(dropna=True)
        summary_lines.append("\n[ID CHECK]")
        summary_lines.append(f"ID unique={id_unique} / rows={n_rows} (unique_ratio={id_unique/max(n_rows,1):.6f})")

    save_text(reports / "00_overall_summary.txt", "\n".join(summary_lines))

    # 3) 결측/중복/유니크/타입 리포트
    missing_rate = df.isna().mean().sort_values(ascending=False)
    nunique = df.nunique(dropna=True).sort_values(ascending=False)
    dtypes = df.dtypes.astype(str)

    pd.DataFrame({
        "dtype": dtypes,
        "nunique": nunique.reindex(df.columns).values,
        "missing_rate": missing_rate.reindex(df.columns).values,
        "missing_count": df.isna().sum().reindex(df.columns).values,
    }, index=df.columns).to_csv(reports / "01_schema_missing_unique.csv", encoding="utf-8-sig")

    # 중복 체크
    dup_rows = df.duplicated().sum()
    save_text(reports / "02_duplicates.txt", f"duplicate_rows={dup_rows}")

    # 결측 시각화
    plot_missing_bar(missing_rate, figures / "missing_top30.png", topk=30)
    plot_target_distribution(y, figures / "target_distribution.png")

    # 4) 컬럼 타입 분류
    col_types = {}
    for c in df.columns:
        col_types[c] = classify_column(df[c], c, target)

    col_type_df = pd.DataFrame({"column": list(col_types.keys()), "type": list(col_types.values())})
    col_type_df.to_csv(reports / "03_column_types.csv", index=False, encoding="utf-8-sig")

    # 5) 수치형/범주형/텍스트형 리스트
    numeric_cols = [c for c, t in col_types.items() if t in ("numeric_continuous", "numeric_discrete")]
    cat_low_cols = [c for c, t in col_types.items() if t == "categorical_low"]
    cat_high_cols = [c for c, t in col_types.items() if t == "categorical_high"]
    text_cols = [c for c, t in col_types.items() if t == "text"]
    constant_cols = [c for c, t in col_types.items() if t == "constant"]
    id_like_cols = [c for c, t in col_types.items() if t == "id_like"]
    datetime_like_cols = [c for c, t in col_types.items() if t == "datetime_like"]

    save_text(
        reports / "04_column_groups.txt",
        "\n".join([
            f"numeric_cols({len(numeric_cols)}): {numeric_cols}",
            f"cat_low_cols({len(cat_low_cols)}): {cat_low_cols}",
            f"cat_high_cols({len(cat_high_cols)}): {cat_high_cols}",
            f"text_cols({len(text_cols)}): {text_cols}",
            f"datetime_like_cols({len(datetime_like_cols)}): {datetime_like_cols}",
            f"id_like_cols({len(id_like_cols)}): {id_like_cols}",
            f"constant_cols({len(constant_cols)}): {constant_cols}",
        ])
    )

    # 6) 수치형 상세 기술통계 + 이상치(IQR) + 히스토그램 저장
    num_desc_rows = []
    outlier_rows = []
    for c in numeric_cols:
        desc = describe_numeric(df[c])
        if not desc:
            continue
        desc["column"] = c
        num_desc_rows.append(desc)

        # 이상치(IQR)
        x = safe_series_to_numeric(df[c]) if df[c].dtype == "object" else df[c]
        lo, hi, mask = iqr_outlier_mask(x, k=1.5)
        if mask is not None:
            out_cnt = int(mask.sum())
            out_rate = float(out_cnt / max(len(x.dropna()), 1))
            outlier_rows.append({
                "column": c,
                "iqr_low": float(lo),
                "iqr_high": float(hi),
                "outlier_count": out_cnt,
                "outlier_rate": out_rate,
            })

            # 이상치 샘플 추출(원본 인덱스 기준 상위 10개)
            out_idx = x.dropna().index[mask]
            sample = df.loc[out_idx, [id_col] if id_col in df.columns else [] + [c, target]].head(10)
            sample.to_csv(artifacts / f"outlier_samples__{c}.csv", index=False, encoding="utf-8-sig")

        # 분포 그림
        plot_numeric_hist(df[c], c, figures / f"hist__{c}.png")

        # 타깃별 boxplot
        plot_numeric_by_target(df, c, target, figures / f"box__{c}_by_target.png")

    if num_desc_rows:
        pd.DataFrame(num_desc_rows).set_index("column").to_csv(reports / "05_numeric_describe.csv", encoding="utf-8-sig")
    if outlier_rows:
        pd.DataFrame(outlier_rows).sort_values("outlier_rate", ascending=False).to_csv(
            reports / "06_numeric_outliers_iqr.csv", index=False, encoding="utf-8-sig"
        )

    # 7) 범주형 분석(저/고카디널) + 타깃 비율표 + 시각화
    def cat_summary(col: str):
        s = df[col].astype("object")
        vc = s.value_counts(dropna=False)
        top = vc.head(20)
        nun = s.nunique(dropna=True)
        miss = float(s.isna().mean())
        return nun, miss, top

    cat_rows = []
    for c in (cat_low_cols + cat_high_cols):
        nun, miss, top = cat_summary(c)
        cat_rows.append({
            "column": c,
            "nunique": nun,
            "missing_rate": miss,
            "top20_values": "; ".join([f"{str(k)}:{int(v)}" for k, v in top.items()])
        })

        # 타깃 비율표(카테고리별)
        g = df.groupby(c, dropna=False)[target].agg(["count", "mean"]).reset_index()
        g = g.sort_values(["mean", "count"], ascending=[False, False])
        g.to_csv(artifacts / f"cat_target_table__{c}.csv", index=False, encoding="utf-8-sig")

        # 시각화(상위 빈도 topk)
        plot_categorical_target_rate(df, c, target, figures / f"cat_target_rate__{c}.png", topk=15)

        # (선택) 카이제곱 검정: 저카디널 중심
        if SCIPY_OK and df[c].nunique(dropna=True) <= 50:
            # 결측 포함하여 범주화
            s2 = df[c].astype("object").fillna("<<MISSING>>")
            ct = pd.crosstab(s2, df[target])
            if ct.shape[0] >= 2 and ct.shape[1] == 2:
                try:
                    chi2, p, dof, _ = chi2_contingency(ct.values)
                    save_text(artifacts / f"chi2__{c}.txt", f"chi2={chi2:.6f}\np={p:.6e}\ndof={dof}\nshape={ct.shape}")
                except Exception as e:
                    save_text(artifacts / f"chi2__{c}.txt", f"failed: {repr(e)}")

    if cat_rows:
        pd.DataFrame(cat_rows).sort_values(["nunique", "missing_rate"], ascending=[False, False]).to_csv(
            reports / "07_categorical_summary.csv", index=False, encoding="utf-8-sig"
        )

    # 8) 텍스트형(자유서술) 기본 진단: 길이/토큰/빈 문자열/상위 키워드(아주 기본)
    text_rows = []
    token_pat = re.compile(r"[A-Za-z0-9가-힣]+")
    for c in text_cols:
        s = df[c].astype("object")
        s_non = s.dropna().astype(str)
        lengths = s_non.map(len)
        empty_rate = float((s_non.str.strip() == "").mean()) if len(s_non) > 0 else 0.0

        # 상위 토큰(빈도) - 너무 큰 비용 방지 위해 샘플링
        sample = s_non.head(500)  # 필요시 늘려도 됨
        tokens = []
        for line in sample:
            tokens.extend(token_pat.findall(line.lower()))
        top_tokens = pd.Series(tokens).value_counts().head(30) if len(tokens) > 0 else pd.Series(dtype=int)
        top_token_str = "; ".join([f"{k}:{int(v)}" for k, v in top_tokens.items()])

        text_rows.append({
            "column": c,
            "missing_rate": float(s.isna().mean()),
            "avg_len": float(lengths.mean()) if len(lengths) > 0 else 0.0,
            "p50_len": float(lengths.median()) if len(lengths) > 0 else 0.0,
            "p95_len": float(np.percentile(lengths, 95)) if len(lengths) > 0 else 0.0,
            "empty_rate(non-null)": empty_rate,
            "nunique": int(s.nunique(dropna=True)),
            "top_tokens(sample500)": top_token_str
        })

        # 길이 분포 시각화
        if len(lengths) > 0:
            plt.figure(figsize=(7, 4))
            plt.hist(lengths, bins=30)
            plt.title(f"Text Length Histogram: {c}")
            plt.xlabel("length")
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(figures / f"text_len__{c}.png", dpi=160)
            plt.close()

    if text_rows:
        pd.DataFrame(text_rows).sort_values("avg_len", ascending=False).to_csv(
            reports / "08_text_columns_profile.csv", index=False, encoding="utf-8-sig"
        )

    # 9) 결측 여부 자체가 타깃과 얼마나 연관되는지(결측 플래그 신호)
    miss_signal = []
    y_int = df[target].astype(float)
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
    miss_signal_df = pd.DataFrame(miss_signal).sort_values("corr_with_target", key=lambda x: x.abs(), ascending=False)
    miss_signal_df.to_csv(reports / "09_missing_flag_signal.csv", index=False, encoding="utf-8-sig")

    # 10) 수치형 상관 및 타깃과의 단순 상관(참고용)
    # target 포함 numeric dataframe 구성
    df_num = pd.DataFrame()
    for c in numeric_cols:
        df_num[c] = safe_series_to_numeric(df[c]) if df[c].dtype == "object" else df[c]
    df_num[target] = df[target].astype(float)

    # 상관 heatmap
    if df_num.shape[1] >= 3:
        correlation_heatmap(df_num.drop(columns=[target], errors="ignore"), figures / "corr_heatmap_numeric.png")

    # target과의 상관 테이블
    corr_rows = []
    for c in numeric_cols:
        x = df_num[c]
        ok = x.notna() & df_num[target].notna()
        if ok.sum() >= 50:
            try:
                r = np.corrcoef(x[ok], df_num[target][ok])[0, 1]
            except Exception:
                r = np.nan
            corr_rows.append({"column": c, "corr_with_target": r, "missing_rate": float(df[c].isna().mean())})
    pd.DataFrame(corr_rows).sort_values("corr_with_target", key=lambda x: x.abs(), ascending=False).to_csv(
        reports / "10_numeric_corr_with_target.csv", index=False, encoding="utf-8-sig"
    )

    # 11) (선택) Mutual Information 기반 중요도(간단히)
    if SKLEARN_OK:
        # 간단 인코딩: 범주형은 라벨 인코딩(누수 없는 정교한 방식은 전처리 단계에서 fold 내 처리)
        # 여기서는 EDA 목적의 "신호 스캐닝"만 한다.
        X_cols = [c for c in df.columns if c not in (target,)]
        X = df[X_cols].copy()

        # 너무 고차원 텍스트는 제외(EDA 신호 스캐닝용)
        X = X.drop(columns=text_cols, errors="ignore")

        # 숫자화
        for c in X.columns:
            if pd.api.types.is_numeric_dtype(X[c]):
                continue
            # 숫자로 변환 가능한 건 숫자로
            sn = safe_series_to_numeric(X[c])
            if sn.notna().mean() >= 0.95:
                X[c] = sn
            else:
                # 범주형은 문자열로 통일 후 라벨인코딩
                X[c] = X[c].astype("object").fillna("<<MISSING>>")
                le = LabelEncoder()
                X[c] = le.fit_transform(X[c].astype(str))

        # 결측은 간단히 -999로(EDA 용)
        X = X.fillna(-999)

        y_mi = df[target].astype(int)
        try:
            mi = mutual_info_classif(X, y_mi, discrete_features="auto", random_state=42)
            mi_df = pd.DataFrame({"column": X.columns, "mutual_info": mi}).sort_values("mutual_info", ascending=False)
            mi_df.to_csv(reports / "11_mutual_info_rank.csv", index=False, encoding="utf-8-sig")
        except Exception as e:
            save_text(reports / "11_mutual_info_rank.txt", f"mutual_info failed: {repr(e)}")

    # 12) 누수 후보 자동 스캔
    leak_df = leakage_suspects(df, target, col_types)
    leak_df.to_csv(reports / "12_leakage_suspects.csv", index=False, encoding="utf-8-sig")

    # 13) 전처리 의사결정에 바로 쓰는 '컬럼 처리 테이블(초안)' 생성
    # - drop 후보: constant, 100% missing
    # - id_like: 모델 입력 제외(보통)
    # - text: 텍스트 파이프라인 필요 표시
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

        decision_rows.append({
            "column": c,
            "type": t,
            "missing_rate": miss,
            "nunique": int(df[c].nunique(dropna=True)),
            "recommendation": recommend,
            "drop_reason": drop_reason
        })

    decision_df = pd.DataFrame(decision_rows).sort_values(
        ["drop_reason", "missing_rate", "nunique"],
        ascending=[False, False, False]
    )
    decision_df.to_csv(reports / "13_preprocessing_decision_draft.csv", index=False, encoding="utf-8-sig")

    # 14) 최종 로그
    done_msg = [
        "EDA DONE",
        f"output_dir={base}",
        f"reports={reports}",
        f"figures={figures}",
        f"artifacts={artifacts}",
        f"scipy={SCIPY_OK}, sklearn={SKLEARN_OK}",
    ]
    save_text(reports / "99_done.txt", "\n".join(done_msg))

    print("\n".join(done_msg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="/mnt/data/train.csv")
    parser.add_argument("--target", type=str, default="completed")
    parser.add_argument("--id_col", type=str, default="ID")
    parser.add_argument("--outdir", type=str, default="eda_output")
    args = parser.parse_args()

    run_eda(
        train_path=args.train_path,
        target=args.target,
        id_col=args.id_col,
        outdir=args.outdir
    )
