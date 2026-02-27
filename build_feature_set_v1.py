#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature_set_v1_spec.json 기반 학습용 데이터셋 생성 스크립트.

출력:
  - feature_sets/data/feature_set_v1_train_core.csv
  - feature_sets/data/feature_set_v1_test_core.csv
  - feature_sets/data/feature_set_v1_train_core_plus_exp.csv
  - feature_sets/data/feature_set_v1_test_core_plus_exp.csv
  - feature_sets/data/feature_set_v1_manifest.json
  - feature_sets/data/feature_set_v1_build_summary.txt
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from eda_train_v3 import build_hypothesis_features


def dedup_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def validate_columns(df: pd.DataFrame, columns: list, label: str):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"[{label}] missing columns: {missing}")


def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def build_feature_set(
    spec_path: Path,
    train_path: Path,
    test_path: Path,
    out_dir: Path,
):
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    target_col = spec["target"]
    id_col = spec["id_column"]

    raw_keep = spec["raw_features_keep_core"]
    raw_drop = set(spec["raw_features_drop_core"])
    eng_core = spec["engineered_features_keep_core"]
    eng_exp = spec["engineered_features_experimental"]

    # 1) load
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if target_col not in train_df.columns:
        raise ValueError(f"train target column not found: {target_col}")
    if target_col in test_df.columns:
        raise ValueError(f"test must not contain target column: {target_col}")
    if id_col not in train_df.columns or id_col not in test_df.columns:
        raise ValueError(f"id column not found in train/test: {id_col}")

    validate_columns(train_df, raw_keep, "train/raw_keep")
    validate_columns(test_df, raw_keep, "test/raw_keep")

    # 2) engineered features (v3 정의와 동일)
    train_eng = build_hypothesis_features(train_df)
    test_eng = build_hypothesis_features(test_df)

    validate_columns(train_eng, eng_core, "train/engineered_core")
    validate_columns(test_eng, eng_core, "test/engineered_core")
    validate_columns(train_eng, eng_exp, "train/engineered_exp")
    validate_columns(test_eng, eng_exp, "test/engineered_exp")

    core_feature_cols = dedup_keep_order(raw_keep + eng_core)
    core_plus_exp_cols = dedup_keep_order(raw_keep + eng_core + eng_exp)

    # 3) assemble
    train_core_feat = pd.concat([train_df[raw_keep].reset_index(drop=True), train_eng[eng_core].reset_index(drop=True)], axis=1)
    test_core_feat = pd.concat([test_df[raw_keep].reset_index(drop=True), test_eng[eng_core].reset_index(drop=True)], axis=1)

    train_exp_feat = pd.concat(
        [train_df[raw_keep].reset_index(drop=True), train_eng[eng_core + eng_exp].reset_index(drop=True)], axis=1
    )
    test_exp_feat = pd.concat(
        [test_df[raw_keep].reset_index(drop=True), test_eng[eng_core + eng_exp].reset_index(drop=True)], axis=1
    )

    # dedup order 강제
    train_core_feat = train_core_feat[core_feature_cols]
    test_core_feat = test_core_feat[core_feature_cols]
    train_exp_feat = train_exp_feat[core_plus_exp_cols]
    test_exp_feat = test_exp_feat[core_plus_exp_cols]

    train_core_out = pd.concat([train_df[[id_col, target_col]].reset_index(drop=True), train_core_feat], axis=1)
    test_core_out = pd.concat([test_df[[id_col]].reset_index(drop=True), test_core_feat], axis=1)
    train_exp_out = pd.concat([train_df[[id_col, target_col]].reset_index(drop=True), train_exp_feat], axis=1)
    test_exp_out = pd.concat([test_df[[id_col]].reset_index(drop=True), test_exp_feat], axis=1)

    # 4) save
    out_dir.mkdir(parents=True, exist_ok=True)
    p_train_core = out_dir / "feature_set_v1_train_core.csv"
    p_test_core = out_dir / "feature_set_v1_test_core.csv"
    p_train_exp = out_dir / "feature_set_v1_train_core_plus_exp.csv"
    p_test_exp = out_dir / "feature_set_v1_test_core_plus_exp.csv"

    save_csv(train_core_out, p_train_core)
    save_csv(test_core_out, p_test_core)
    save_csv(train_exp_out, p_train_exp)
    save_csv(test_exp_out, p_test_exp)

    manifest = {
        "name": spec.get("name", "feature_set_v1"),
        "source_spec": str(spec_path),
        "source_train": str(train_path),
        "source_test": str(test_path),
        "target": target_col,
        "id_column": id_col,
        "core_feature_count": len(core_feature_cols),
        "core_plus_exp_feature_count": len(core_plus_exp_cols),
        "core_feature_columns": core_feature_cols,
        "core_plus_exp_feature_columns": core_plus_exp_cols,
        "raw_drop_columns": sorted(raw_drop),
        "outputs": {
            "train_core": str(p_train_core),
            "test_core": str(p_test_core),
            "train_core_plus_exp": str(p_train_exp),
            "test_core_plus_exp": str(p_test_exp),
        },
    }

    p_manifest = out_dir / "feature_set_v1_manifest.json"
    p_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_lines = [
        "[BUILD] feature_set_v1",
        f"train_rows={train_df.shape[0]}, test_rows={test_df.shape[0]}",
        f"core_feature_count={len(core_feature_cols)}",
        f"core_plus_exp_feature_count={len(core_plus_exp_cols)}",
        f"train_core_shape={train_core_out.shape}",
        f"test_core_shape={test_core_out.shape}",
        f"train_core_plus_exp_shape={train_exp_out.shape}",
        f"test_core_plus_exp_shape={test_exp_out.shape}",
        f"output_manifest={p_manifest}",
    ]
    (out_dir / "feature_set_v1_build_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n".join(summary_lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec_path", type=str, default="feature_sets/feature_set_v1_spec.json")
    parser.add_argument("--train_path", type=str, default="open/train.csv")
    parser.add_argument("--test_path", type=str, default="open/test.csv")
    parser.add_argument("--out_dir", type=str, default="feature_sets/data")
    args = parser.parse_args()

    build_feature_set(
        spec_path=Path(args.spec_path),
        train_path=Path(args.train_path),
        test_path=Path(args.test_path),
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()

