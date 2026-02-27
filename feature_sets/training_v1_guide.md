# Training Guide (feature_set_v1)

## 1) Build Feature Matrices
```bash
python3 build_feature_set_v1.py
```

## 2) Train / Validate / Predict
```bash
python3 train_feature_set_v1.py --mode both
```

`--mode` options:
- `core`
- `core_plus_exp`
- `both` (default)

## 3) Outputs
`model_output_v1/<timestamp>/`
- `core/model_metrics_base.csv` (단일 모델 성능)
- `core/model_metrics.csv`
- `core/fold_metrics.csv`
- `core/oof_pred_<candidate>.csv`
- `core/test_pred_<candidate>.csv`
- `core/submission_<candidate>.csv`
- `core/submission_<candidate>_f1max.csv`
- `core/submission_best.csv`
- `core/submission_best_f1max.csv`
- `core/submission_blend.csv`
- `core/submission_blend_f1max.csv`
- `core/submission_selected.csv` (모드 내부 자동 선택)
- `core/submission_selected_f1max.csv` (순수 F1 최대화 임계값 버전)
- `core/selected_threshold_sweep.csv` (selected 후보의 목표 양성비율별 제출 파일 목록)
- `core_plus_exp/...` (if run)
- `core_plus_exp/submission_selected.csv` (if run)
- `final_submission.csv` (모드 간 최종 자동 선택 결과, `completed`는 0/1 정수)
- `final_oof.csv`
- `run_summary.json`

`run_summary.json`에 `final_mode`, `final_variant`, `final_cv_f1_reg`, `final_threshold_reg`, `final_test_pos_rate_reg`가 기록됩니다.
`core/feature_overview.json`에는 `selected_test_pos_rate_reg`, `target_pos_rate_reg`, `threshold_sweep_path`가 추가로 기록됩니다.
