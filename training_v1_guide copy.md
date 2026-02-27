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
- `core/model_metrics.csv`
- `core/fold_metrics.csv`
- `core/submission_best.csv`
- `core/submission_best_f1max.csv`
- `core/submission_blend.csv`
- `core/submission_blend_f1max.csv`
- `core/submission_selected.csv` (모드 내부 자동 선택)
- `core/submission_selected_f1max.csv` (순수 F1 최대화 임계값 버전)
- `core/test_pred_best.csv` (확률)
- `core/test_pred_blend.csv` (확률)
- `core_plus_exp/...` (if run)
- `core_plus_exp/submission_selected.csv` (if run)
- `final_submission.csv` (모드 간 최종 자동 선택 결과, `completed`는 0/1 정수)
- `final_oof.csv`
- `run_summary.json`

`run_summary.json`에 `final_mode`, `final_variant`, `final_cv_f1_reg`, `final_threshold_reg`가 기록됩니다.
